"""Convert LangChain AIMessage responses to OpenAI ChatCompletion objects.

DSPy's ``BaseLM._process_lm_response()`` expects responses that conform to
the `OpenAI ChatCompletion format <https://platform.openai.com/docs/api-reference/chat/object>`_.
Specifically, it accesses both attribute (``response.choices``) and bracket
(``response["choices"]``) notation, so plain dicts are insufficient --
we use actual ``openai.types`` Pydantic models.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from langchain_core.messages import AIMessage
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from openai.types.completion_usage import CompletionTokensDetails


@dataclass(frozen=True)
class ReasoningContent:
    """Structured view of a provider's reasoning / thinking output.

    Providers differ in what they expose:

    - **Claude (Bedrock)** emits only a full thinking trace; no condensed
      summary is surfaced → ``summary=None, full=<joined thinking>``.
    - **Gemini (Vertex) 2.5+** may emit both a short summary via
      ``additional_kwargs["reasoning_summary"]`` and a full trace as
      ``content`` blocks with ``type="thinking"`` →
      ``summary=<summary>, full=<full>``.
    - **GPT-5 / Azure** emits ``additional_kwargs["reasoning_content"]``.
      In default (summarized) mode this is a condensed summary; in
      verbose mode it's the full trace. We conservatively treat it as
      ``summary`` — callers that want the full trace can opt into it
      via provider config and it will be picked up the same way.
    - Non-reasoning providers → both ``None``.

    ``__str__`` returns the best-available text (summary preferred over
    full) so existing callers that expect a string still work.
    """

    summary: Optional[str] = None
    full: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.summary) or bool(self.full)

    def __str__(self) -> str:
        return self.summary or self.full or ""


def langchain_to_openai_completion(message: AIMessage, model: str) -> ChatCompletion:
    """Convert a LangChain ``AIMessage`` to an OpenAI ``ChatCompletion``.

    Args:
        message: The response from a LangChain ``BaseChatModel.invoke()`` call.
        model: The model identifier string (e.g. ``"vertex_ai/gemini-2.5-flash"``).

    Returns:
        An ``openai.types.chat.ChatCompletion`` object consumable by
        ``BaseLM._process_lm_response()``.
    """
    usage_meta = message.usage_metadata or {}

    reasoning_tokens = usage_meta.get("output_token_details", {}).get("reasoning", 0) or 0
    completion_tokens_details = (
        CompletionTokensDetails(reasoning_tokens=reasoning_tokens) if reasoning_tokens > 0 else None
    )

    tool_calls = _extract_tool_calls(message)
    text_content = _extract_text_content(message)

    completion = ChatCompletion(
        id=f"langchain-{uuid4().hex[:8]}",
        model=model,
        object="chat.completion",
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                finish_reason=_extract_finish_reason(message, has_tool_calls=bool(tool_calls)),
                message=ChatCompletionMessage(
                    role="assistant",
                    content=text_content or None,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=usage_meta.get("input_tokens", 0) or 0,
            completion_tokens=usage_meta.get("output_tokens", 0) or 0,
            total_tokens=usage_meta.get("total_tokens", 0) or 0,
            completion_tokens_details=completion_tokens_details,
        ),
    )

    # ``_hidden_params`` + ``cache_hit`` are read by DSPy's
    # BaseLM._process_lm_response() for cost tracking and cache detection.
    completion._hidden_params = {}
    completion.cache_hit = False

    reasoning = _extract_reasoning_content(message)
    if reasoning:
        completion.choices[0].message.reasoning_content = str(reasoning)
    # Always stash the structured object (even when empty) so downstream
    # helpers that want the ``summary`` / ``full`` split can retrieve it
    # from ``lm.history[-1]["response"]`` without re-parsing the
    # AIMessage. DSPy's ``BaseLM._process_lm_response`` flattens
    # reasoning down to a single string on the output dict — this extra
    # attribute is the only lossless path.
    completion._reasoning_content_detailed = reasoning

    return completion


def _extract_text_content(message: AIMessage) -> str:
    """Extract text content from a LangChain AIMessage.

    LangChain providers return content in different formats:
    - **String**: Most providers return ``message.content`` as a plain string.
    - **List of blocks**: Claude (Bedrock) and Gemini (Vertex) may return a list
      of content blocks, e.g. ``[{"type": "text", "text": "..."}]``, especially
      when thinking/reasoning is enabled.

    Returns:
        The concatenated text content as a string.
    """
    content = message.content
    if isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts) if parts else ""
    return ""


def _extract_tool_calls(message: AIMessage) -> Optional[list[ChatCompletionMessageToolCall]]:
    """Extract tool calls from a LangChain AIMessage in OpenAI format.

    LangChain unifies tool calls across providers on ``AIMessage.tool_calls``
    as ``[{"id": str, "name": str, "args": dict}, ...]``.  We translate that
    into OpenAI's ``ChatCompletionMessageToolCall`` with a JSON-serialized
    ``arguments`` field so DSPy's tool-using modules (ReAct, etc.) can
    consume it directly.

    Returns:
        A list of tool-call objects, or ``None`` if the message carries none.
    """
    raw = getattr(message, "tool_calls", None)
    if not raw:
        return None
    return [
        ChatCompletionMessageToolCall(
            id=tc.get("id") or f"call_{uuid4().hex[:8]}",
            type="function",
            function=Function(
                name=tc["name"],
                arguments=json.dumps(tc.get("args") or {}),
            ),
        )
        for tc in raw
    ]


# Map provider-specific finish reason values to OpenAI-compatible ones.
_FINISH_REASON_MAP = {
    "stop": "stop",
    "STOP": "stop",
    "end_turn": "stop",
    "length": "length",
    "MAX_TOKENS": "length",
    "max_tokens": "length",
    "SAFETY": "content_filter",
    "safety": "content_filter",
    "RECITATION": "content_filter",
    "tool_calls": "tool_calls",
    "function_call": "function_call",
    "TOOL_CALLS": "tool_calls",
    "tool_use": "tool_calls",
}


def _extract_finish_reason(message: AIMessage, has_tool_calls: bool = False) -> str:
    """Extract the finish reason from a LangChain AIMessage.

    Different providers expose this in different locations within
    ``response_metadata`` and use different naming conventions.
    Normalizes to OpenAI-compatible values.  When the message carries tool
    calls and the provider did not mark it as such explicitly, coerce to
    ``"tool_calls"`` so downstream OpenAI-shaped consumers dispatch correctly.
    """
    meta = message.response_metadata or {}
    raw = meta.get("finish_reason") or meta.get("stop_reason")
    if raw is None:
        return "tool_calls" if has_tool_calls else "stop"
    mapped = _FINISH_REASON_MAP.get(raw, "stop")
    if has_tool_calls and mapped == "stop":
        return "tool_calls"
    return mapped


def _extract_reasoning_content(message: AIMessage) -> ReasoningContent:
    """Extract reasoning / thinking content from a LangChain AIMessage.

    Returns a :class:`ReasoningContent` carrying the provider-emitted
    summary (when available) and the full thinking trace (when
    available) as separate fields. Callers that just want "some text"
    can ``str(...)`` the result and get a summary-preferred-over-full
    string (empty string when neither field is set).

    Provider-specific extraction:

    - **Claude via Bedrock** — ``additional_kwargs["thinking"]`` holds
      a list of thinking blocks (the full trace). Anthropic's API does
      not currently emit a separate summary, so we set
      ``summary=None, full=<joined thinking>``.
    - **Gemini via Vertex** — content blocks with ``type="thinking"``
      carry the full trace; when the provider emits a condensed
      summary it shows up under
      ``additional_kwargs["reasoning_summary"]``.
    - **GPT-5 / Azure** — ``additional_kwargs["reasoning_content"]``
      is treated as the summary (matches Azure's default summarized
      mode; callers that enable verbose mode get the full trace in
      the same field and it is still treated as summary, which is
      acceptable given the caller's fallback chain).
    - Non-reasoning providers → both fields ``None``.

    Returns:
        A :class:`ReasoningContent` with ``summary`` and ``full``
        fields populated according to what the provider emitted.
    """
    extra = message.additional_kwargs or {}

    summary: Optional[str] = None
    full: Optional[str] = None

    # Claude via Bedrock: thinking blocks. Full trace only.
    thinking_blocks = extra.get("thinking")
    if isinstance(thinking_blocks, list) and thinking_blocks:
        texts = [b.get("text", "") for b in thinking_blocks if isinstance(b, dict)]
        combined = "\n".join(t for t in texts if t)
        if combined:
            full = combined

    # Gemini via Vertex: content blocks with type="thinking" (full trace).
    # Take the first non-empty hit; ``text`` / ``thinking`` / ``reasoning``
    # are all in use across LangChain versions, so try all three.
    if full is None and isinstance(message.content, list):
        thinking_parts = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text = block.get("text") or block.get("thinking") or block.get("reasoning") or ""
                if text:
                    thinking_parts.append(text)
        if thinking_parts:
            full = "\n".join(thinking_parts)

    # Gemini reasoning summary (when provider emits one alongside the trace).
    gemini_summary = extra.get("reasoning_summary")
    if gemini_summary:
        summary = gemini_summary

    # GPT-5 / Azure: reasoning_content — treat as summary.
    # Only take it when a Gemini-style summary hasn't already been set so
    # Vertex keeps its own summary when both somehow appear.
    if summary is None:
        azure_reasoning = extra.get("reasoning_content")
        if azure_reasoning:
            summary = azure_reasoning

    return ReasoningContent(summary=summary, full=full)
