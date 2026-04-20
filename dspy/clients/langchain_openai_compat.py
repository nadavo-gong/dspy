"""Convert LangChain AIMessage responses to OpenAI ChatCompletion objects.

DSPy's ``BaseLM._process_lm_response()`` expects responses that conform to
the `OpenAI ChatCompletion format <https://platform.openai.com/docs/api-reference/chat/object>`_.
Specifically, it accesses both attribute (``response.choices``) and bracket
(``response["choices"]``) notation, so plain dicts are insufficient --
we use actual ``openai.types`` Pydantic models.
"""

import time
from typing import Optional
from uuid import uuid4

from langchain_core.messages import AIMessage
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails


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

    completion = ChatCompletion(
        id=f"langchain-{uuid4().hex[:8]}",
        model=model,
        object="chat.completion",
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                finish_reason=_extract_finish_reason(message),
                message=ChatCompletionMessage(
                    role="assistant",
                    content=_extract_text_content(message),
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

    # DSPy reads these for cost tracking and cache detection.
    completion._hidden_params = {}
    completion.cache_hit = False

    reasoning = _extract_reasoning_content(message)
    if reasoning:
        completion.choices[0].message.reasoning_content = reasoning

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
}


def _extract_finish_reason(message: AIMessage) -> str:
    """Extract the finish reason from a LangChain AIMessage.

    Different providers expose this in different locations within
    ``response_metadata`` and use different naming conventions.
    Normalizes to OpenAI-compatible values.
    """
    meta = message.response_metadata or {}
    raw = meta.get("finish_reason") or meta.get("stop_reason") or "stop"
    return _FINISH_REASON_MAP.get(raw, "stop")


def _extract_reasoning_content(message: AIMessage) -> Optional[str]:
    """Extract reasoning / thinking content from a LangChain AIMessage.

    Provider-specific extraction:
    - **Claude (Bedrock)**: ``additional_kwargs["thinking"]`` list of blocks.
    - **Gemini (Vertex)**: Content blocks with ``type="thinking"``.
    - **GPT (Azure)**: ``additional_kwargs["reasoning_content"]``.

    Returns:
        The reasoning text, or ``None`` if not present.
    """
    extra = message.additional_kwargs or {}

    # Claude via Bedrock: thinking blocks
    thinking_blocks = extra.get("thinking")
    if isinstance(thinking_blocks, list) and thinking_blocks:
        texts = [b.get("text", "") for b in thinking_blocks if isinstance(b, dict)]
        combined = "\n".join(t for t in texts if t)
        if combined:
            return combined

    # GPT via Azure: reasoning_content
    reasoning = extra.get("reasoning_content")
    if reasoning:
        return reasoning

    # Gemini via Vertex: content blocks with type="thinking"
    # LangChain surfaces these in content when content is a list of blocks.
    if isinstance(message.content, list):
        thinking_parts = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_parts.append(block.get("thinking", ""))
        if thinking_parts:
            return "\n".join(thinking_parts)

    return None
