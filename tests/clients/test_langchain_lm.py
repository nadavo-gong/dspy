"""Unit tests for ``dspy.clients.langchain_lm`` and ``langchain_openai_compat``.

Uses mocked ``BaseChatModel`` instances -- no real LangChain provider is
invoked.
"""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai.types.chat import ChatCompletion

from dspy.clients.langchain_lm import (
    LangChainLM,
    _DEFAULT_BIND_PARAMS,
    _extract_langchain_params,
    _is_context_window_error,
    _to_langchain_messages,
    translate_azure,
    translate_bedrock,
    translate_vertex,
)
from dspy.clients.langchain_openai_compat import (
    _extract_finish_reason,
    _extract_reasoning_content,
    _extract_text_content,
    _extract_tool_calls,
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
# _extract_tool_calls + tool-call round-tripping
# ---------------------------------------------------------------------------


def test_extract_tool_calls_translates_langchain_shape_to_openai():
    msg = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "get_weather", "args": {"city": "Tokyo"}}],
    )
    tool_calls = _extract_tool_calls(msg)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "c1"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"city": "Tokyo"}


def test_extract_tool_calls_returns_none_when_absent():
    assert _extract_tool_calls(AIMessage(content="x")) is None


def test_tool_calls_surface_on_completion_and_finish_reason_is_tool_calls():
    msg = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "f", "args": {"x": 1}}],
    )
    resp = langchain_to_openai_completion(msg, model="openai/gpt-4o")
    choice = resp.choices[0]
    assert choice.message.tool_calls is not None
    assert choice.message.tool_calls[0].function.name == "f"
    assert choice.finish_reason == "tool_calls"


def test_tool_calls_missing_id_gets_synthetic_id():
    # LangChain itself requires ``id`` (possibly ``None``), so emulate the
    # blank-id case that can arise when a provider returns an empty string.
    msg = AIMessage(
        content="",
        tool_calls=[{"id": "", "name": "f", "args": {}}],
    )
    tool_calls = _extract_tool_calls(msg)
    assert tool_calls[0].id.startswith("call_")


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
        ("tool_use", "tool_calls"),
        ("unknown_value", "stop"),
    ],
)
def test_finish_reason_normalization(raw, expected):
    msg = AIMessage(content="x", response_metadata={"finish_reason": raw})
    assert _extract_finish_reason(msg) == expected


def test_finish_reason_falls_back_to_stop_reason():
    msg = AIMessage(content="x", response_metadata={"stop_reason": "end_turn"})
    assert _extract_finish_reason(msg) == "stop"


def test_finish_reason_coerces_to_tool_calls_when_tool_calls_present():
    msg = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    assert _extract_finish_reason(msg, has_tool_calls=True) == "tool_calls"


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


def test_reasoning_from_gemini_text_block_under_thinking_type():
    # Real Vertex output puts the text under "text", not "thinking".
    msg = AIMessage(
        content=[
            {"type": "text", "text": "final"},
            {"type": "thinking", "text": "inner monologue"},
        ]
    )
    assert _extract_reasoning_content(msg) == "inner monologue"


def test_reasoning_from_gemini_legacy_thinking_key():
    msg = AIMessage(
        content=[
            {"type": "text", "text": "final"},
            {"type": "thinking", "thinking": "legacy key"},
        ]
    )
    assert _extract_reasoning_content(msg) == "legacy key"


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


def test_to_langchain_messages_accepts_tool_role():
    result = _to_langchain_messages(
        [{"role": "tool", "content": "42", "tool_call_id": "c1"}]
    )
    assert isinstance(result[0], ToolMessage)
    assert result[0].content == "42"
    assert result[0].tool_call_id == "c1"


def test_to_langchain_messages_converts_assistant_tool_calls():
    result = _to_langchain_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                    }
                ],
            }
        ]
    )
    ai = result[0]
    assert ai.tool_calls[0]["id"] == "c1"
    assert ai.tool_calls[0]["name"] == "get_weather"
    assert ai.tool_calls[0]["args"] == {"city": "Tokyo"}


def test_to_langchain_messages_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown message role"):
        _to_langchain_messages([{"role": "alien", "content": "x"}])


# ---------------------------------------------------------------------------
# _extract_langchain_params
# ---------------------------------------------------------------------------


def test_extract_langchain_params_filters_unknown_and_none():
    extracted = _extract_langchain_params(
        {"temperature": 0.2, "max_tokens": 100, "nonsense": "x", "stop": None},
        _DEFAULT_BIND_PARAMS,
    )
    assert extracted == {"temperature": 0.2, "max_tokens": 100}


def test_extract_langchain_params_respects_custom_allowlist():
    extracted = _extract_langchain_params(
        {"temperature": 0.2, "reasoning_effort": "high"},
        _DEFAULT_BIND_PARAMS | {"reasoning_effort"},
    )
    assert extracted == {"temperature": 0.2, "reasoning_effort": "high"}


# ---------------------------------------------------------------------------
# Built-in parameter translators
# ---------------------------------------------------------------------------


def test_translate_vertex_renames_max_tokens():
    assert translate_vertex({"max_tokens": 100}) == {"max_output_tokens": 100}


def test_translate_azure_renames_max_tokens():
    assert translate_azure({"max_tokens": 100}) == {"max_completion_tokens": 100}


def test_translate_bedrock_drops_unsupported():
    out = translate_bedrock(
        {"max_tokens": 100, "reasoning_effort": "high", "response_format": {"type": "json"}},
    )
    assert out == {"max_tokens": 100}


def test_translators_do_not_mutate_input():
    params = {"max_tokens": 100}
    translate_vertex(params)
    translate_azure(params)
    translate_bedrock(params)
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

    mock_model.bind.assert_called_once()
    bind_kwargs = mock_model.bind.call_args.kwargs
    assert "temperature" in bind_kwargs
    assert "max_tokens" in bind_kwargs
    mock_model.invoke.assert_called_once()


def test_forward_passes_bind_params_through_translator():
    ai = AIMessage(content="x", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    mock_model = _make_mock_chat_model(ai)

    received: list[dict] = []

    def translator(params):
        received.append(dict(params))
        return {"custom_key": 42}

    lm = LangChainLM(
        model="vendor/model",
        langchain_model=mock_model,
        cache=False,
        param_translator=translator,
    )
    lm.forward(messages=[{"role": "user", "content": "hi"}], temperature=0.3)

    assert received and received[0]["temperature"] == 0.3
    mock_model.bind.assert_called_once_with(custom_key=42)


def test_forward_honors_custom_bind_params():
    ai = AIMessage(content="x", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    mock_model = _make_mock_chat_model(ai)

    lm = LangChainLM(
        model="openai/o3-mini",
        langchain_model=mock_model,
        cache=False,
        bind_params=_DEFAULT_BIND_PARAMS | {"reasoning_effort"},
    )
    lm.forward(messages=[{"role": "user", "content": "hi"}], reasoning_effort="high")

    assert mock_model.bind.call_args.kwargs["reasoning_effort"] == "high"


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


def test_supports_function_calling_defaults_to_introspection():
    # MagicMock(spec=BaseChatModel) *does* expose bind_tools because it's on
    # the BaseChatModel spec -- so this introspects True.
    model = MagicMock(spec=BaseChatModel)
    lm = LangChainLM(model="openai/gpt-4o", langchain_model=model)
    assert lm.supports_function_calling is True


def test_supports_function_calling_introspection_negative_when_method_missing():
    # A minimal object without bind_tools triggers the False branch.
    class Minimal:
        pass

    lm = LangChainLM(model="custom/m", langchain_model=Minimal())
    assert lm.supports_function_calling is False


def test_supports_reasoning_is_false_by_default():
    lm = LangChainLM(model="openai/gpt-4o", langchain_model=MagicMock(spec=BaseChatModel))
    assert lm.supports_reasoning is False


def test_capability_overrides_stick():
    lm = LangChainLM(
        model="openai/o3-mini",
        langchain_model=MagicMock(spec=BaseChatModel),
        supports_function_calling=False,
        supports_response_schema=False,
        supports_reasoning=True,
    )
    assert lm.supports_function_calling is False
    assert lm.supports_response_schema is False
    assert lm.supports_reasoning is True


def test_supported_params_reports_bind_params():
    lm = LangChainLM(
        model="openai/o3-mini",
        langchain_model=MagicMock(spec=BaseChatModel),
        bind_params=_DEFAULT_BIND_PARAMS | {"reasoning_effort"},
    )
    assert "reasoning_effort" in lm.supported_params
    assert "temperature" in lm.supported_params


def test_copy_preserves_langchain_model_and_updates_kwargs():
    mock_model = MagicMock(spec=BaseChatModel)
    lm = LangChainLM(model="openai/gpt-4o", langchain_model=mock_model, temperature=0.5)
    lm2 = lm.copy(temperature=0.9)

    assert lm2 is not lm
    assert lm2._langchain_model is mock_model
    assert lm2.kwargs["temperature"] == 0.9
    # original should be unchanged
    assert lm.kwargs["temperature"] == 0.5


def test_copy_preserves_capability_flags_and_translator():
    translator = lambda p: p
    lm = LangChainLM(
        model="openai/o3-mini",
        langchain_model=MagicMock(spec=BaseChatModel),
        supports_reasoning=True,
        bind_params={"temperature", "reasoning_effort"},
        param_translator=translator,
    )
    lm2 = lm.copy(temperature=0.2)
    assert lm2.supports_reasoning is True
    assert lm2._bind_params == {"temperature", "reasoning_effort"}
    assert lm2._param_translator is translator


# ---------------------------------------------------------------------------
# Smoke test: assert the built-in message classes are the real ones.
# ---------------------------------------------------------------------------


def test_role_map_uses_langchain_message_classes():
    from dspy.clients.langchain_lm import ROLE_MAP

    assert ROLE_MAP["system"] is SystemMessage
    assert ROLE_MAP["user"] is HumanMessage
    assert ROLE_MAP["tool"] is ToolMessage
