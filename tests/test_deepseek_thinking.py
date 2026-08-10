"""Offline request-boundary tests for the DeepSeek thinking switch."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from tradingagents.graph import trading_graph as graph_module
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.factory import create_llm_client


def _mock_chat_response(model: str) -> dict:
    return {
        "id": "chatcmpl-offline-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _captured_request_body(
    *,
    provider: str,
    model: str,
    base_url: str | None = None,
    **client_kwargs,
) -> dict:
    """Invoke through an in-memory transport and return the final HTTP JSON body."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_mock_chat_response(model))

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        llm = create_llm_client(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key="placeholder",
            http_client=http_client,
            max_retries=0,
            **client_kwargs,
        ).get_llm()
        llm.invoke([HumanMessage(content="offline payload inspection")])

    assert len(requests) == 1
    return json.loads(requests[0].content)


@pytest.mark.unit
def test_deepseek_v4_flash_thinking_disabled_reaches_request_payload():
    body = _captured_request_body(
        provider="deepseek",
        model="deepseek-v4-flash",
        deepseek_thinking="disabled",
        temperature=0.0,
    )

    assert body["thinking"] == {"type": "disabled"}
    assert "extra_body" not in body


@pytest.mark.unit
def test_deepseek_v4_flash_thinking_enabled_reaches_request_payload():
    body = _captured_request_body(
        provider="deepseek",
        model="deepseek-v4-flash",
        deepseek_thinking="enabled",
        temperature=0.0,
    )

    assert body["thinking"] == {"type": "enabled"}
    assert "extra_body" not in body


@pytest.mark.unit
def test_deepseek_v4_flash_temperature_zero_reaches_request_payload():
    body = _captured_request_body(
        provider="deepseek",
        model="deepseek-v4-flash",
        deepseek_thinking="disabled",
        temperature=0.0,
    )

    assert body["temperature"] == 0.0


@pytest.mark.unit
def test_invalid_deepseek_thinking_fails_loudly():
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"llm_provider": "deepseek", "deepseek_thinking": "automatic"}

    with pytest.raises(ValueError, match="deepseek_thinking must be one of"):
        graph._get_provider_kwargs()


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["openai", "google", "anthropic"])
def test_deepseek_thinking_does_not_leak_to_other_provider_kwargs(provider):
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"llm_provider": provider, "deepseek_thinking": "disabled"}

    assert "deepseek_thinking" not in graph._get_provider_kwargs()


@pytest.mark.unit
def test_deepseek_thinking_does_not_leak_to_openai_request_payload():
    body = _captured_request_body(
        provider="openai",
        model="gpt-4.1",
        base_url="https://offline-openai.invalid/v1",
        deepseek_thinking="disabled",
        temperature=0.0,
    )

    assert "thinking" not in body
    assert "deepseek_thinking" not in body


@pytest.mark.unit
def test_quick_and_deep_clients_share_deepseek_thinking_and_temperature(
    monkeypatch, tmp_path,
):
    captured: list[dict] = []

    class ConstructionComplete(Exception):
        pass

    def capture_client(**kwargs):
        captured.append(kwargs)
        if len(captured) == 2:
            # Both clients are constructed before either get_llm() call. Stop
            # here so this focused test does not compile the full agent graph.
            raise ConstructionComplete
        return object()

    monkeypatch.setattr(graph_module, "create_llm_client", capture_client)
    config = {
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
        "llm_provider": "deepseek",
        "deep_think_llm": "deepseek-v4-pro",
        "quick_think_llm": "deepseek-v4-flash",
        "deepseek_thinking": "disabled",
        "temperature": 0.0,
    }

    with pytest.raises(ConstructionComplete):
        TradingAgentsGraph(config=config)

    assert [call["model"] for call in captured] == [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ]
    for call in captured:
        assert call["provider"] == "deepseek"
        assert call["deepseek_thinking"] == "disabled"
        assert call["temperature"] == 0.0
