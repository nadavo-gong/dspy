"""Unit tests for ``dspy.clients.langchain_lm`` and ``langchain_openai_compat``.

Uses mocked ``BaseChatModel`` instances -- no real LangChain provider is
invoked.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from openai.types.chat import ChatCompletion

from dspy.clients.langchain_lm import (
    LangChainLM,
    _extract_langchain_params,
    _is_context_window_error,
    _to_langchain_messages,
    _translate_params_for_provider,
)
from dspy.clients.langchain_openai_compat import (
    _extract_finish_reason,
    _extract_reasoning_content,
    _extract_text_content,
    langchain_to_openai_completion,
)
from dspy.utils.exceptions import ContextWindowExceededError


def _make_mock_chat_model(response: AIMessage) -> BaseChatModel:
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = response
    model.bind.return_value = model
    return model


# ---------------------------------------------------------------------------
# langchain_to_openai_completion
# ---------------------------------------------------------------------------


def test_basic_conversion_preserves_text_and_model():
    msg = AIMessage(
        content="hello",
        usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    )
    resp = langchain_to_openai_completion(msg, model="openai/gpt-4o")

    assert isinstance(resp, ChatCompletion)
    assert resp.model == "openai/gpt-4o"
    assert resp.choices[0].message.content == "hello"
    assert resp.usage.prompt_tokens == 3
    assert resp.usage.completion_tokens == 1
    assert resp.usage.total_tokens == 4
    # DSPy reads these two attributes for cost tracking + cache detection.
    assert resp._hidden_params == {}
    assert resp.cache_hit is False


def test_reasoning_tokens_surface_in_completion_tokens_details():
    msg = AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_token_details": {"reasoning": 15},
        },
    )
    resp = langchain_to_openai_completion(msg, model="azure/gpt-5")

    assert resp.usage.completion_tokens_details is not None
    assert resp.usage.completion_tokens_details.reasoning_tokens == 15


def test_zero_reasoning_tokens_omits_details():
    msg = AIMessage(
        content="answer",
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    resp = langchain_to_openai_completion(msg, model="openai/gpt-4o")
    assert resp.usage.completion_tokens_details is None


# ---------------------------------------------------------------------------
# _extract_text_content
# ---------------------------------------------------------------------------


def test_extract_text_from_string_content():
    assert _extract_text_content(AIMessage(content="plain")) == "plain"


def test_extract_text_from_block_list():
    msg = AIMessage(
        content=[
            {"type": "text", "text": "first"},
            {"type": "thinking", "thinking": "ignored"},
            {"type": "text", "text": "second"},
        ]
    )
    assert _extract_text_content(msg) == "first\nsecond"


def test_extract_text_returns_empty_for_missing_content():
    assert _extract_text_content(AIMessage(content="")) == ""
    assert _extract_text_content(AIMessage(content=[])) == ""


# ---------------------------------------------------------------------------
# _extract_finish_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "stop"),
        ("end_turn", "stop"),
        ("STOP", "stop"),
        ("length", "length"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("tool_calls", "tool_calls"),
        ("unknown_value", "stop"),
    ],
)
def test_finish_reason_normalization(raw, expected):
    msg = AIMessage(content="x", response_metadata={"finish_reason": raw})
    assert _extract_finish_reason(msg) == expected


def test_finish_reason_falls_back_to_stop_reason():
    msg = AIMessage(content="x", response_metadata={"stop_reason": "end_turn"})
    assert _extract_finish_reason(msg) == "stop"


# ---------------------------------------------------------------------------
# _extract_reasoning_content
# ---------------------------------------------------------------------------


def test_reasoning_from_claude_thinking_blocks():
    msg = AIMessage(
        content="x",
        additional_kwargs={
            "thinking": [
                {"type": "thinking", "text": "step 1"},
                {"type": "thinking", "text": "step 2"},
            ]
        },
    )
    assert _extract_reasoning_content(msg) == "step 1\nstep 2"


def test_reasoning_from_azure_reasoning_content():
    msg = AIMessage(content="x", additional_kwargs={"reasoning_content": "chain of thought"})
    assert _extract_reasoning_content(msg) == "chain of thought"


def test_reasoning_from_gemini_block_content():
    msg = AIMessage(
        content=[
            {"type": "text", "text": "final"},
            {"type": "thinking", "thinking": "inner monologue"},
        ]
    )
    assert _extract_reasoning_content(msg) == "inner monologue"


def test_reasoning_returns_none_when_absent():
    assert _extract_reasoning_content(AIMessage(content="x")) is None


def test_reasoning_content_surfaces_on_completion_message():
    msg = AIMessage(content="answer", additional_kwargs={"reasoning_content": "why"})
    resp = langchain_to_openai_completion(msg, model="azure/gpt-5")
    assert resp.choices[0].message.reasoning_content == "why"


# ---------------------------------------------------------------------------
# _to_langchain_messages
# ---------------------------------------------------------------------------


def test_to_langchain_messages_maps_roles():
    result = _to_langchain_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert [type(m).__name__ for m in result] == ["SystemMessage", "HumanMessage", "AIMessage"]
    assert [m.content for m in result] == ["sys", "hi", "hello"]


def test_to_langchain_messages_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown message role"):
        _to_langchain_messages([{"role": "tool", "content": "x"}])


# ---------------------------------------------------------------------------
# _extract_langchain_params / _translate_params_for_provider
# ---------------------------------------------------------------------------


def test_extract_langchain_params_filters_unknown_and_none():
    extracted = _extract_langchain_params(
        {"temperature": 0.2, "max_tokens": 100, "nonsense": "x", "stop": None}
    )
    assert extracted == {"temperature": 0.2, "max_tokens": 100}


def test_translate_params_for_vertex_renames_max_tokens():
    out = _translate_params_for_provider({"max_tokens": 100}, "vertex_ai/gemini-2.5-flash")
    assert out == {"max_output_tokens": 100}


def test_translate_params_for_azure_renames_max_tokens():
    out = _translate_params_for_provider({"max_tokens": 100}, "azure/gpt-5")
    assert out == {"max_completion_tokens": 100}


def test_translate_params_for_bedrock_drops_unsupported():
    out = _translate_params_for_provider(
        {"max_tokens": 100, "reasoning_effort": "high", "response_format": {"type": "json"}},
        "bedrock/anthropic.claude-sonnet-4",
    )
    assert out == {"max_tokens": 100}


def test_translate_params_does_not_mutate_input():
    params = {"max_tokens": 100}
    _translate_params_for_provider(params, "vertex_ai/gemini-2.5-flash")
    assert params == {"max_tokens": 100}


# ---------------------------------------------------------------------------
# _is_context_window_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Context window exceeded",
        "prompt is too long",
        "Request too large for model",
        "Input too long: 200000 tokens",
        "exceeds maximum length of 32000 tokens",
    ],
)
def test_context_window_error_detected_by_message(message):
    assert _is_context_window_error(Exception(message)) is True


def test_context_window_error_detected_by_class_name():
    class ContextWindowError(Exception):
        pass

    assert _is_context_window_error(ContextWindowError("anything")) is True


def test_context_window_error_rejects_unrelated():
    assert _is_context_window_error(ValueError("bad api key")) is False


# ---------------------------------------------------------------------------
# LangChainLM.forward / aforward
# ---------------------------------------------------------------------------


def test_forward_routes_through_langchain_and_returns_chat_completion():
    ai = AIMessage(
        content="response-text",
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    mock_model = _make_mock_chat_model(ai)

    lm = LangChainLM(model="openai/gpt-4o", langchain_model=mock_model, cache=False)
    resp = lm.forward(messages=[{"role": "user", "content": "hi"}], temperature=0.1)

    # Per-call temperature overrides BaseLM's default; max_tokens defaults to 1000.
    mock_model.bind.assert_called_once()
    bind_kwargs = mock_model.bind.call_args.kwargs
    assert bind_kwargs["temperature"] == 0.1
    mock_model.invoke.assert_called_once()
    assert isinstance(resp, ChatCompletion)
    assert resp.choices[0].message.content == "response-text"


def test_forward_binds_default_params_from_base_lm():
    ai = AIMessage(content="x", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    mock_model = _make_mock_chat_model(ai)

    lm = LangChainLM(model="openai/gpt-4o", langchain_model=mock_model, cache=False)
    lm.forward(messages=[{"role": "user", "content": "hi"}])

    # BaseLM seeds self.kwargs with temperature + max_tokens, so bind is always called.
    mock_model.bind.assert_called_once()
    bind_kwargs = mock_model.bind.call_args.kwargs
    assert "temperature" in bind_kwargs
    assert "max_tokens" in bind_kwargs
    mock_model.invoke.assert_called_once()


def test_forward_raises_context_window_exceeded():
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind.return_value = mock_model
    mock_model.invoke.side_effect = RuntimeError("Context window exceeded: 200000 tokens")

    lm = LangChainLM(model="openai/gpt-4o", langchain_model=mock_model, cache=False)
    with pytest.raises(ContextWindowExceededError):
        lm.forward(messages=[{"role": "user", "content": "hi"}])


def test_forward_preserves_unrelated_exceptions():
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind.return_value = mock_model
    mock_model.invoke.side_effect = RuntimeError("invalid api key")

    lm = LangChainLM(model="openai/gpt-4o", langchain_model=mock_model, cache=False)
    with pytest.raises(RuntimeError, match="invalid api key"):
        lm.forward(messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# LangChainLM capabilities + copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("openai/gpt-5", True),
        ("vertex_ai/gemini-2.5-flash", True),
        ("bedrock/anthropic.claude-sonnet-4", True),
        ("openai/o3-mini", True),
        ("openai/gpt-4o", False),
    ],
)
def test_supports_reasoning_matches_known_models(model_name, expected):
    lm = LangChainLM(model=model_name, langchain_model=MagicMock(spec=BaseChatModel))
    assert lm.supports_reasoning is expected


def test_copy_preserves_langchain_model_and_updates_kwargs():
    mock_model = MagicMock(spec=BaseChatModel)
    lm = LangChainLM(model="openai/gpt-4o", langchain_model=mock_model, temperature=0.5)
    lm2 = lm.copy(temperature=0.9)

    assert lm2 is not lm
    assert lm2._langchain_model is mock_model
    assert lm2.kwargs["temperature"] == 0.9
    # original should be unchanged
    assert lm.kwargs["temperature"] == 0.5
