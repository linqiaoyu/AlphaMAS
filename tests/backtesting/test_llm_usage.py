from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tradingagents.backtesting.artifacts import LLM_USAGE_COLUMNS
from tradingagents.backtesting.llm_usage import LLMUsageCallback, extract_provider_usage


def _result(message: AIMessage, **llm_output) -> LLMResult:
    return LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output=llm_output,
    )


def test_callback_emits_fixed_schema_provider_usage_for_one_case():
    callback = LLMUsageCallback(
        run_id="run-1", provider="deepseek", thinking_mode="disabled",
    )
    callback.start_case("M0", "AAPL__2024-01-05", "AAPL", "2024-01-05")
    request_id = uuid4()
    callback.on_chat_model_start(
        {"kwargs": {}},
        [[]],
        run_id=request_id,
        metadata={"langgraph_node": "market_analyst", "ls_provider": "openai"},
        invocation_params={"model": "deepseek-v4-flash"},
    )
    message = AIMessage(
        content="ok",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 12,
                "prompt_cache_hit_tokens": 3,
                "prompt_cache_miss_tokens": 9,
                "completion_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 2},
                "total_tokens": 17,
            }
        },
    )
    callback.on_llm_end(
        _result(message, model_name="deepseek-v4-flash"), run_id=request_id,
    )

    rows = callback.finish_case()

    assert len(rows) == 1
    assert tuple(rows[0]) == LLM_USAGE_COLUMNS
    assert rows[0]["experiment_id"] == "M0"
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["usage_source"] == "live_request"
    assert rows[0]["origin_run_id"] == "run-1"
    assert rows[0]["agent_node"] == "market_analyst"
    assert rows[0]["provider"] == "deepseek"
    assert rows[0]["model"] == "deepseek-v4-flash"
    assert rows[0]["thinking_mode"] == "disabled"
    assert rows[0]["prompt_cache_hit_tokens"] == 3
    assert rows[0]["prompt_cache_miss_tokens"] == 9
    assert rows[0]["reasoning_tokens"] == 2
    assert rows[0]["latency_seconds"] >= 0


def test_usage_metadata_is_supported_without_inferring_provider_omissions():
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )

    usage = extract_provider_usage(_result(message))

    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 14
    assert usage["prompt_cache_hit_tokens"] is None
    assert usage["prompt_cache_miss_tokens"] is None
    assert usage["reasoning_tokens"] is None


def test_quick_and_deep_requests_share_case_context_but_keep_their_models():
    callback = LLMUsageCallback(
        run_id="run-1", provider="deepseek", thinking_mode="disabled",
    )
    callback.start_case("M0", "AAPL__2024-01-05", "AAPL", "2024-01-05")
    for model in ("quick-model", "deep-model"):
        request_id = uuid4()
        callback.on_llm_start(
            {"kwargs": {}}, ["prompt"], run_id=request_id,
            invocation_params={"model_name": model},
        )
        callback.on_llm_end(
            _result(
                AIMessage(
                    content="ok",
                    response_metadata={
                        "token_usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        }
                    },
                )
            ),
            run_id=request_id,
        )

    rows = callback.finish_case()

    assert [row["model"] for row in rows] == ["quick-model", "deep-model"]
    assert {row["case_id"] for row in rows} == {"AAPL__2024-01-05"}


def test_failed_real_request_is_recorded_with_null_usage():
    callback = LLMUsageCallback(
        run_id="run-1", provider="deepseek", thinking_mode="disabled",
    )
    callback.start_case("M0", "AAPL__2024-01-05", "AAPL", "2024-01-05")
    request_id = uuid4()
    callback.on_llm_start(
        {"kwargs": {"model": "deepseek-v4-flash"}}, ["prompt"], run_id=request_id,
    )
    callback.on_llm_error(RuntimeError("mock failure"), run_id=request_id)

    row = callback.finish_case()[0]

    assert row["model"] == "deepseek-v4-flash"
    assert row["prompt_tokens"] is None
    assert row["total_tokens"] is None
