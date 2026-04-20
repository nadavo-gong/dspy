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

The class aims to be provider-agnostic: capability flags can be
introspected from the LangChain model or supplied explicitly, and
parameter translation is delegated to caller-supplied callables (see
``translate_vertex`` / ``translate_azure`` / ``translate_bedrock`` for
examples).
"""

import json
import logging
import re
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from dspy.clients.base_lm import BaseLM
from dspy.clients.langchain_openai_compat import langchain_to_openai_completion
from dspy.utils.exceptions import ContextWindowExceededError

logger = logging.getLogger(__name__)

ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": LCAIMessage,
    "tool": ToolMessage,
}

# Default DSPy kwargs forwarded to LangChain's .bind() at call time.  Callers
# can supply a custom set via ``LangChainLM(bind_params=...)`` to add
# provider-specific knobs (e.g. ``reasoning_effort`` for OpenAI reasoning
# models).
_DEFAULT_BIND_PARAMS = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "stop",
        "n",
        "tools",
        "response_format",
    }
)

# Backwards-compat alias (existing callers may still import this name).
_LANGCHAIN_BIND_PARAMS = _DEFAULT_BIND_PARAMS


ParamTranslator = Callable[[dict[str, Any]], dict[str, Any]]


class LangChainLM(BaseLM):
    """DSPy LM backend using a LangChain ``BaseChatModel``.

    Overrides ``BaseLM.forward()`` to route calls through LangChain instead of
    LiteLLM.  Capability properties default to values introspected from the
    underlying LangChain model and can be overridden via constructor args
    (per PR #9516 interface).

    Args:
        model: Model identifier string (e.g. ``"vertex_ai/gemini-2.5-flash"``).
        langchain_model: A pre-configured LangChain ``BaseChatModel`` instance.
        cache: Whether to enable DSPy's request cache.
        supports_function_calling: If ``None``, detected via
            ``hasattr(langchain_model, "bind_tools")``.
        supports_response_schema: If ``None``, detected via
            ``hasattr(langchain_model, "with_structured_output")``.
        supports_reasoning: Whether the model emits reasoning tokens / content
            (defaults to ``False``; callers that know the model supports it
            should pass ``True``).
        bind_params: Iterable of DSPy kwarg names that should be forwarded to
            ``BaseChatModel.bind()``.  Defaults to ``_DEFAULT_BIND_PARAMS``.
        param_translator: Optional callable that takes a dict of bind-params
            and returns a (possibly renamed / filtered) dict suitable for the
            underlying provider.  Useful for mapping ``max_tokens`` to
            ``max_output_tokens`` (Vertex) or ``max_completion_tokens``
            (Azure).  Defaults to the identity function.
        **kwargs: Passed through to ``BaseLM.__init__`` and stored in
            ``self.kwargs``.
    """

    def __init__(
        self,
        model: str,
        langchain_model: BaseChatModel,
        cache: bool = True,
        *,
        supports_function_calling: Optional[bool] = None,
        supports_response_schema: Optional[bool] = None,
        supports_reasoning: bool = False,
        bind_params: Optional[set[str]] = None,
        param_translator: Optional[ParamTranslator] = None,
        **kwargs,
    ):
        super().__init__(model=model, cache=cache, **kwargs)
        self._langchain_model = langchain_model
        self._supports_function_calling = (
            supports_function_calling
            if supports_function_calling is not None
            else hasattr(langchain_model, "bind_tools")
        )
        self._supports_response_schema = (
            supports_response_schema
            if supports_response_schema is not None
            else hasattr(langchain_model, "with_structured_output")
        )
        self._supports_reasoning = supports_reasoning
        self._bind_params = set(bind_params) if bind_params is not None else set(_DEFAULT_BIND_PARAMS)
        self._param_translator: ParamTranslator = param_translator or (lambda p: p)

    # -- Core LM interface -----------------------------------------------------

    def forward(self, prompt=None, messages=None, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        lc_messages = _to_langchain_messages(messages)

        bind_params = _extract_langchain_params(merged_kwargs, self._bind_params)
        bind_params = self._param_translator(bind_params)
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

        bind_params = _extract_langchain_params(merged_kwargs, self._bind_params)
        bind_params = self._param_translator(bind_params)
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
        return self._supports_function_calling

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def supports_response_schema(self) -> bool:
        return self._supports_response_schema

    @property
    def supported_params(self) -> set[str]:
        return set(self._bind_params)

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
            supports_function_calling=self._supports_function_calling,
            supports_response_schema=self._supports_response_schema,
            supports_reasoning=self._supports_reasoning,
            bind_params=self._bind_params,
            param_translator=self._param_translator,
            **new_kwargs,
        )

    def dump_state(self):
        state = super().dump_state()
        state["model"] = self.model
        state["model_type"] = self.model_type
        return state


# -- Helper functions ----------------------------------------------------------


def _to_langchain_messages(messages: list[dict[str, Any]]) -> list:
    """Convert DSPy message dicts to LangChain message objects.

    Handles ``system``/``user``/``assistant``/``tool`` roles.  Assistant
    messages carrying ``tool_calls`` (OpenAI format) are round-tripped into
    LangChain's unified tool-call shape on ``AIMessage.tool_calls``.
    """
    out = []
    for m in messages:
        role = m["role"]
        content = m.get("content") or ""
        if role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
            continue
        if role == "assistant" and m.get("tool_calls"):
            out.append(
                LCAIMessage(
                    content=content,
                    tool_calls=[
                        {
                            "id": tc.get("id") or "",
                            "name": tc["function"]["name"],
                            "args": _loads_or_passthrough(tc["function"].get("arguments", "{}")),
                        }
                        for tc in m["tool_calls"]
                    ],
                )
            )
            continue
        cls = ROLE_MAP.get(role)
        if cls is None:
            raise ValueError(f"Unknown message role: {role!r}")
        out.append(cls(content=content))
    return out


def _loads_or_passthrough(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value or {}


def _extract_langchain_params(kwargs: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Extract parameters compatible with LangChain's ``.bind()`` from DSPy kwargs."""
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


# -- Built-in parameter translators -------------------------------------------
#
# Named helpers for common LangChain providers.  Callers compose these (or
# supply their own) via ``LangChainLM(param_translator=...)``.


def translate_vertex(params: dict[str, Any]) -> dict[str, Any]:
    """Translate generic bind-params for Google Vertex (``ChatVertexAI``)."""
    out = dict(params)
    if "max_tokens" in out:
        out["max_output_tokens"] = out.pop("max_tokens")
    return out


def translate_azure(params: dict[str, Any]) -> dict[str, Any]:
    """Translate generic bind-params for Azure OpenAI (newer reasoning models)."""
    out = dict(params)
    if "max_tokens" in out:
        out["max_completion_tokens"] = out.pop("max_tokens")
    return out


def translate_bedrock(params: dict[str, Any]) -> dict[str, Any]:
    """Translate generic bind-params for AWS Bedrock chat models.

    Bedrock does not accept ``reasoning_effort`` (Claude's thinking config is
    handled at the LangChain model level) nor ``response_format``.
    """
    out = dict(params)
    out.pop("reasoning_effort", None)
    out.pop("response_format", None)
    return out


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
