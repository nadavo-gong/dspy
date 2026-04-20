"""DSPy LM backend using LangChain chat models instead of LiteLLM.

Subclasses :class:`dspy.clients.base_lm.BaseLM` so that DSPy modules,
optimizers, adapters, and callbacks work transparently.  LLM calls are
routed through a user-supplied LangChain
:class:`langchain_core.language_models.BaseChatModel` instance.

Usage::

    from langchain_openai import ChatOpenAI
    from dspy.clients.langchain_lm import LangChainLM

    lc_model = ChatOpenAI(model="gpt-4o-mini")
    lm = LangChainLM(model="gpt-4o-mini", langchain_model=lc_model)
    dspy.configure(lm=lm)
"""

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage, SystemMessage

from dspy.clients.base_lm import BaseLM
from dspy.clients.langchain_openai_compat import langchain_to_openai_completion
from dspy.utils.exceptions import ContextWindowExceededError

logger = logging.getLogger(__name__)

ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": LCAIMessage,
}

# DSPy kwargs that should be forwarded to LangChain's .bind() at call time.
_LANGCHAIN_BIND_PARAMS = {
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "n",
    "tools",
    "response_format",
    "reasoning_effort",
}


class LangChainLM(BaseLM):
    """DSPy LM backend using a LangChain ``BaseChatModel``.

    Overrides ``BaseLM.forward()`` to route calls through LangChain instead of
    LiteLLM.  Capability properties are overridden to inform DSPy adapters
    about what the underlying model supports (per PR #9516 interface).

    Args:
        model: Model identifier string (e.g. ``"vertex_ai/gemini-2.5-flash"``).
        langchain_model: A pre-configured LangChain ``BaseChatModel`` instance.
        cache: Whether to enable DSPy's request cache.
        **kwargs: Passed through to ``BaseLM.__init__`` and stored in ``self.kwargs``.
    """

    def __init__(
        self,
        model: str,
        langchain_model: BaseChatModel,
        cache: bool = True,
        **kwargs,
    ):
        super().__init__(model=model, cache=cache, **kwargs)
        self._langchain_model = langchain_model

    # -- Core LM interface -----------------------------------------------------

    def forward(self, prompt=None, messages=None, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        lc_messages = _to_langchain_messages(messages)

        bind_params = _extract_langchain_params(merged_kwargs)
        bind_params = _translate_params_for_provider(bind_params, self.model)
        model = self._langchain_model.bind(**bind_params) if bind_params else self._langchain_model

        try:
            response = model.invoke(lc_messages)
        except Exception as e:
            if _is_context_window_error(e):
                raise ContextWindowExceededError(model=self.model, message=str(e)) from e
            raise

        return langchain_to_openai_completion(response, model=self.model)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        lc_messages = _to_langchain_messages(messages)

        bind_params = _extract_langchain_params(merged_kwargs)
        bind_params = _translate_params_for_provider(bind_params, self.model)
        model = self._langchain_model.bind(**bind_params) if bind_params else self._langchain_model

        try:
            response = await model.ainvoke(lc_messages)
        except Exception as e:
            if _is_context_window_error(e):
                raise ContextWindowExceededError(model=self.model, message=str(e)) from e
            raise

        return langchain_to_openai_completion(response, model=self.model)

    # -- Capability properties (override BaseLM defaults, PR #9516) ------------

    @property
    def supports_function_calling(self) -> bool:
        return True

    @property
    def supports_reasoning(self) -> bool:
        model_lower = self.model.lower()
        return any(
            x in model_lower
            for x in [
                "o1",
                "o3",
                "o4",
                "claude-sonnet-4",
                "claude-3-7",
                "gemini-2.5",
                "gpt-5",
            ]
        )

    @property
    def supports_response_schema(self) -> bool:
        return True

    @property
    def supported_params(self) -> set[str]:
        return {"response_format", "temperature", "max_tokens", "top_p", "stop", "tools"}

    # -- Serialization ---------------------------------------------------------

    def copy(self, **kwargs):
        new_kwargs = {**self.kwargs}
        init_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(self, key) and key not in ("kwargs",):
                init_kwargs[key] = value
            elif value is None:
                new_kwargs.pop(key, None)
            else:
                new_kwargs[key] = value

        return LangChainLM(
            model=init_kwargs.get("model", self.model),
            langchain_model=self._langchain_model,
            cache=init_kwargs.get("cache", self.cache),
            **new_kwargs,
        )

    def dump_state(self):
        state = super().dump_state() if hasattr(super(), "dump_state") else {}
        state["model"] = self.model
        state["model_type"] = self.model_type
        return state


# -- Helper functions ----------------------------------------------------------


def _to_langchain_messages(messages: list[dict[str, Any]]) -> list:
    """Convert DSPy message dicts to LangChain message objects."""
    lc_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        cls = ROLE_MAP.get(role)
        if cls is None:
            raise ValueError(f"Unknown message role: {role!r}")
        lc_messages.append(cls(content=content))
    return lc_messages


def _extract_langchain_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract parameters compatible with LangChain's ``.bind()`` from DSPy kwargs."""
    return {k: v for k, v in kwargs.items() if k in _LANGCHAIN_BIND_PARAMS and v is not None}


def _translate_params_for_provider(params: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Map generic DSPy parameter names to provider-specific names.

    Different LangChain providers expect different parameter names:
    - Google Gemini: ``max_output_tokens`` instead of ``max_tokens``
    - Azure OpenAI (newer models): ``max_completion_tokens`` instead of ``max_tokens``
    - Bedrock: does not accept ``reasoning_effort`` or ``response_format``
      (Claude's thinking config is handled at the LangChain model level)
    """
    params = {**params}  # shallow copy to avoid mutation
    if model_name.startswith("vertex_ai/"):
        if "max_tokens" in params:
            params["max_output_tokens"] = params.pop("max_tokens")
    elif model_name.startswith("azure/"):
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
    elif model_name.startswith("bedrock/"):
        params.pop("reasoning_effort", None)
        params.pop("response_format", None)
    return params


# Pattern that matches common context-window / input-length errors across providers.
_CONTEXT_WINDOW_PATTERNS = re.compile(
    r"context.window|context.length|too.many.tokens|input.too.long|"
    r"maximum.*token|token.limit|exceeds.*length|prompt.is.too.long|"
    r"request.too.large|content.too.large",
    re.IGNORECASE,
)


def _is_context_window_error(exc: Exception) -> bool:
    """Detect whether *exc* is a provider's context-window-exceeded error."""
    msg = str(exc)
    if _CONTEXT_WINDOW_PATTERNS.search(msg):
        return True
    cls_name = type(exc).__name__
    if "ContextWindow" in cls_name or "ContentTooLarge" in cls_name:
        return True
    return False
