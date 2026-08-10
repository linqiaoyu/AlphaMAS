import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.backtesting.artifacts import (
    ANALYSIS_READY_SCHEMAS,
    ARTIFACT_SCHEMA_VERSION,
    CASE_INDEX_COLUMNS,
    DATA_AVAILABILITY_COLUMNS,
    LLM_USAGE_COLUMNS,
    RESULT_TABLES,
    _decision_timeline,
    artifact_schema,
    collect_agent_analysis_records,
    flatten_source_audit,
    graph_config_sha256,
    validate_artifact_bundle,
    write_agent_analysis_tables,
)
from tradingagents.backtesting.cache import cache_key
from tradingagents.backtesting.memory_archive import (
    archive_final_experiment_memory,
    runtime_experiment_memory_path,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_result_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for filename in RESULT_TABLES:
        (path / filename).write_text("placeholder\n", encoding="utf-8")
    (path / "metrics.csv").write_text("metric\n", encoding="utf-8")
    _write_json(path / "metrics.json", {})


def _complete_bundle(root: Path, *, strategy: str = "sma") -> None:
    lineage = {
        "memory_lineage_id": "run-1",
        "memory_lifecycle": "independent_fresh",
        "memory_resumed_from_run_id": None,
    }
    run_identity = {
        "experiment_id": "M0",
        "run_id": "run-1",
        "backtest_protocol_sha256": "b" * 64,
        "implementation_identity": "commit-a",
        "market_input_identity": None,
        **lineage,
    }
    graph_config = {
        "llm_provider": "mock",
        "quick_think_llm": "mock-model",
        "deep_think_llm": "mock-model",
        "deepseek_thinking": "disabled",
        "temperature": 0.0,
        "research_depth": "medium",
        "max_debate_rounds": 3,
        "max_risk_discuss_rounds": 3,
        "historical_memory_lineage_id": lineage["memory_lineage_id"],
    }
    graph_hash = graph_config_sha256(graph_config)
    graph_config["graph_config_sha256"] = graph_hash
    config = {
        "symbols": ["AAPL"],
        "strategy": strategy,
        "experiment_id": "M0",
        "run_id": "run-1",
        "actual_decision_count": 1,
        "graph_config_sha256": graph_hash,
        **run_identity,
    }
    _write_json(root / "manifest.json", {
        "graph_config_sha256": graph_hash,
        **run_identity,
    })
    _write_json(root / "config.resolved.json", config)
    _write_json(root / "graph_config.resolved.json", graph_config)
    _write_json(root / "environment.json", {})
    _write_json(root / "artifact_schema.json", artifact_schema())
    _write_json(root / "run_status.json", {
        "status": "success",
        "experiment_id": run_identity["experiment_id"],
        "run_id": run_identity["run_id"],
        **lineage,
    })
    _write_json(root / "validation/validation_report.json", {"status": "passed"})
    (root / "schedule.csv").write_text(
        "decision_session,execution_session\n2024-01-05,2024-01-08\n",
        encoding="utf-8",
    )
    (root / "failures").mkdir(parents=True)
    (root / "failures/failures.jsonl").write_text("", encoding="utf-8")

    manifest_entries = []
    for symbol in ("AAPL", "SPY"):
        market_path = root / "inputs/market_data" / f"{symbol}.csv"
        market_path.parent.mkdir(parents=True, exist_ok=True)
        market_path.write_text("Date,Close\n2024-01-05,100\n", encoding="utf-8")
        actions_path = root / "inputs/corporate_actions" / f"{symbol}.csv"
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        actions_path.write_text("Date,Dividends,Stock Splits\n", encoding="utf-8")
        manifest_entries.append({
            "symbol": symbol,
            "sha256": hashlib.sha256(market_path.read_bytes()).hexdigest(),
        })
    _write_json(root / "inputs/data_manifest.json", {"symbols": manifest_entries})

    _write_result_directory(root / "strategy/AAPL")
    _write_result_directory(root / "strategy/combined")
    _write_result_directory(root / "benchmarks/AAPL_buy_and_hold")
    _write_result_directory(root / "benchmarks/SPY_buy_and_hold")
    (root / "aggregate").mkdir(parents=True)
    (root / "aggregate/equal_weight_equity.csv").write_text(
        "session,normalized_equity\n", encoding="utf-8",
    )
    _write_json(root / "aggregate/equal_weight_metrics.json", {})

    analysis_root = root / "analysis_ready"
    analysis_root.mkdir(parents=True)
    for filename, columns in ANALYSIS_READY_SCHEMAS.items():
        pd.DataFrame(columns=columns).to_csv(analysis_root / filename, index=False)

    if strategy == "tradingagents":
        case_dir = root / "strategy/cases/AAPL/2024-01-05"
        report_dir = case_dir / "reports"
        report_dir.mkdir(parents=True)
        (report_dir / "complete_report.md").write_text("report", encoding="utf-8")
        case_id = "AAPL:2024-01-05"
        identity = {
            "graph_config_sha256": graph_hash,
            "experiment_id": run_identity["experiment_id"],
            "memory_lineage_id": lineage["memory_lineage_id"],
            "case": case_id,
        }
        identity_key = cache_key(identity)
        model_config = {**graph_config}
        _write_json(case_dir / "decision.json", {
            "action": "BUY",
            "status": "success",
            "metadata": {"model_config": model_config, "cache_status": "miss"},
        })
        _write_json(case_dir / "model_config.json", model_config)
        _write_json(case_dir / "run_context.json", {
            "mode": "historical",
            "experiment_id": run_identity["experiment_id"],
            "memory_lineage_id": lineage["memory_lineage_id"],
        })
        _write_json(case_dir / "cache_identity.json", {
            "cache_key": identity_key, "identity": identity,
        })
        _write_json(case_dir / "source_audit.json", {
            "run_context": {
                "experiment_id": run_identity["experiment_id"],
                "memory_lineage_id": lineage["memory_lineage_id"],
            },
            "sources": [{"source_name": "yfinance", "status": "used"}],
        })
        raw_usage = {
            "experiment_id": "M0",
            "run_id": "run-1",
            "case_id": case_id,
            "symbol": "AAPL",
            "decision_session": "2024-01-05",
            "usage_source": "live_request",
            "origin_run_id": "run-1",
            "provider": "mock",
            "model": "mock-model",
            "thinking_mode": "disabled",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        _write_json(case_dir / "llm_usage.json", [raw_usage])
        _write_json(case_dir / "case_metadata.json", {
            "case_id": case_id,
            "cache_key": identity_key,
            "cache_status": "miss",
        })
        case = {
            "experiment_id": "M0",
            "run_id": "run-1",
            "case_id": case_id,
            "symbol": "AAPL",
            "decision_session": "2024-01-05",
            "action": "BUY",
            "decision_status": "success",
            "provider": "mock",
            "quick_model": "mock-model",
            "deep_model": "mock-model",
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "research_depth": "medium",
            "debate_rounds": 3,
            "risk_rounds": 3,
            "graph_config_sha256": graph_hash,
            "report_path": "strategy/cases/AAPL/2024-01-05/reports",
            "source_audit_path": "strategy/cases/AAPL/2024-01-05/source_audit.json",
            "cache_key": identity_key,
            "cache_status": "miss",
        }
        availability = {
            "experiment_id": "M0",
            "run_id": "run-1",
            "symbol": "AAPL",
            "decision_session": "2024-01-05",
            "source_name": "yfinance",
            "status": "used",
        }
        write_agent_analysis_tables(
            analysis_root,
            case_index=[case],
            data_availability=[availability],
            llm_usage=[raw_usage],
        )
        pd.DataFrame([{
            "symbol": "AAPL",
            "decision_session": "2024-01-05",
            "decision_time_utc": "2024-01-05T21:00:00+00:00",
            "action": "BUY",
            "status": "success",
            "reason": "test decision",
            "target_weight": 1.0,
            "cache_status": "miss",
        }]).reindex(
            columns=ANALYSIS_READY_SCHEMAS["decision_timeline.csv"]
        ).to_csv(analysis_root / "decision_timeline.csv", index=False)

        runtime_memory = root.parent / f".{root.name}-runtime-memory"
        runtime_path = runtime_experiment_memory_path(
            runtime_memory,
            experiment_id=run_identity["experiment_id"],
            graph_config_sha256=graph_hash,
            memory_lineage_id=lineage["memory_lineage_id"],
            symbol="AAPL",
        )
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            "[2024-01-05 | AAPL | Buy | pending]\n\n"
            "DECISION:\nRating: Buy\n\n<!-- ENTRY_END -->\n\n",
            encoding="utf-8",
        )
        descriptor = archive_final_experiment_memory(
            run_dir=root,
            runtime_memory_dir=runtime_memory,
            experiment_id=run_identity["experiment_id"],
            run_id=run_identity["run_id"],
            memory_lineage_id=lineage["memory_lineage_id"],
            memory_lifecycle=lineage["memory_lifecycle"],
            memory_resumed_from_run_id=lineage["memory_resumed_from_run_id"],
            graph_config_sha256=graph_hash,
            symbols=["AAPL"],
        )
        top_manifest_path = root / "manifest.json"
        top_manifest = json.loads(top_manifest_path.read_text(encoding="utf-8"))
        top_manifest["memory_archive"] = descriptor
        _write_json(top_manifest_path, top_manifest)


def test_fixed_agent_tables_are_header_only_for_synthetic_runs(tmp_path):
    paths = write_agent_analysis_tables(tmp_path)

    assert tuple(pd.read_csv(paths["case_index.csv"]).columns) == CASE_INDEX_COLUMNS
    assert tuple(pd.read_csv(paths["data_availability.csv"]).columns) == (
        DATA_AVAILABILITY_COLUMNS
    )
    assert tuple(pd.read_csv(paths["llm_usage.csv"]).columns) == LLM_USAGE_COLUMNS
    assert all(pd.read_csv(path).empty for path in paths.values())


def test_source_audit_flattening_preserves_provider_fields_and_order():
    audit = {
        "sources": [
            {
                "source_name": "fred",
                "capability": "LIVE_ONLY",
                "status": "blocked",
                "requested_end": "2024-01-05T21:00:00+00:00",
                "reason": "no vintage",
            },
            {"source_name": "yfinance", "capability": "PIT", "status": "used"},
        ]
    }
    rows = flatten_source_audit(
        audit,
        experiment_id="M0",
        run_id="run-1",
        symbol="AAPL",
        decision_session="2024-01-05",
    )

    assert [row["source_name"] for row in rows] == ["fred", "yfinance"]
    assert rows[0]["requested_end"] == "2024-01-05T21:00:00+00:00"
    assert rows[0]["latest_event_time"] is None
    assert rows[1]["experiment_id"] == "M0"


def test_artifact_schema_1_3_declares_agent_and_cost_analysis_tables():
    schema = artifact_schema()

    assert ARTIFACT_SCHEMA_VERSION == "1.3"
    assert schema["schema_version"] == "1.3"
    assert schema["analysis_ready_schemas"]["case_index.csv"] == list(CASE_INDEX_COLUMNS)
    assert "slippage_cost" in schema["analysis_ready_schemas"]["decision_timeline.csv"]
    assert (
        "slippage_cost_in_period"
        in schema["analysis_ready_schemas"]["weekly_performance.csv"]
    )
    assert "total_slippage_cost" in schema["definitions"]
    assert "final_memory_archive" in schema["definitions"]
    assert "memory/manifest.json" in schema["memory_artifacts"]
    assert "memory/symbols/<symbol>.md" in schema["memory_artifacts"]
    assert "analysis_ready/data_availability.csv" in schema["csv_files"]
    assert "analysis_ready/llm_usage.csv" in schema["csv_files"]


def test_decision_timeline_preserves_cache_hit_miss_and_bypass_verbatim():
    decisions = pd.DataFrame([
        {
            "decision_session": "2024-01-05",
            "decision_time_utc": "2024-01-05T21:00:00+00:00",
            "action": "BUY",
            "status": "success",
            "reason": "fresh",
            "target_weight": 1.0,
            "metadata": {"cache_status": "miss"},
        },
        {
            "decision_session": "2024-01-12",
            "decision_time_utc": "2024-01-12T21:00:00+00:00",
            "action": "HOLD",
            "status": "cached",
            "reason": "replayed",
            "target_weight": 1.0,
            "metadata": {"cache_status": "hit"},
        },
        {
            "decision_session": "2024-01-19",
            "decision_time_utc": "2024-01-19T21:00:00+00:00",
            "action": "SELL",
            "status": "success",
            "reason": "forced",
            "target_weight": 0.0,
            "metadata": {"cache_status": "bypass"},
        },
        {
            "decision_session": "2024-01-26",
            "decision_time_utc": "2024-01-26T21:00:00+00:00",
            "action": "HOLD",
            "status": "success",
            "reason": "deterministic",
            "target_weight": 0.0,
            "metadata": {},
        },
    ])
    result = SimpleNamespace(
        decisions=decisions,
        orders=pd.DataFrame(),
        fills=pd.DataFrame(),
        daily_equity=pd.DataFrame({
            "session": decisions["decision_session"],
            "current_weight": [0.0, 1.0, 1.0, 0.0],
            "equity": [100_000.0] * 4,
        }),
    )

    timeline = _decision_timeline({"AAPL": result})

    assert timeline["cache_status"].tolist() == ["miss", "hit", "bypass", ""]


def test_collect_agent_analysis_records_aggregates_case_audit_and_usage(tmp_path):
    _complete_bundle(tmp_path, strategy="tradingagents")
    case_root = tmp_path / "strategy/cases/AAPL/2024-01-05"
    _write_json(case_root / "source_audit.json", {
        "sources": [{
            "source_name": "fred",
            "capability": "LIVE_ONLY",
            "status": "blocked",
            "reason": "no vintage",
        }]
    })
    _write_json(case_root / "llm_usage.json", [{
        "agent_node": "market_analyst",
        "model": "deepseek-v4-flash",
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }])
    graph_config = {
        "llm_provider": "deepseek",
        "quick_think_llm": "deepseek-v4-flash",
        "deep_think_llm": "deepseek-v4-flash",
        "deepseek_thinking": "disabled",
        "temperature": 0.0,
        "research_depth": "medium",
        "max_debate_rounds": 3,
        "max_risk_discuss_rounds": 3,
        "graph_config_sha256": "a" * 64,
    }
    result = SimpleNamespace(
        decisions=pd.DataFrame([{
            "symbol": "AAPL",
            "decision_session": "2024-01-05",
            "decision_time_utc": "2024-01-05T21:00:00+00:00",
            "action": "BUY",
            "status": "success",
            "strategy_id": "tradingagents",
            "rebalance_status": "ordered",
            "metadata": {
                "report_path": "strategy/cases/AAPL/2024-01-05/reports",
                "source_audit_path": "strategy/cases/AAPL/2024-01-05/source_audit.json",
                "cache_key": "cache-key",
                "cache_status": "miss",
                "wall_clock_seconds": 1.25,
            },
        }]),
        orders=pd.DataFrame([{
            "created_at": "2024-01-05T21:00:00+00:00",
            "intended_execution_session": "2024-01-08",
        }]),
        daily_equity=pd.DataFrame([{
            "session": "2024-01-05",
            "equity": 100_000.0,
            "current_weight": 0.0,
        }]),
    )

    cases, availability, usage = collect_agent_analysis_records(
        "M0", "run-1", {"AAPL": result},
        tmp_path / "strategy/cases", graph_config,
    )

    assert len(cases) == len(availability) == len(usage) == 1
    assert cases[0]["graph_config_sha256"] == "a" * 64
    assert cases[0]["prompt_tokens"] == 10
    assert cases[0]["total_tokens"] == 14
    assert availability[0]["status"] == "blocked"
    assert usage[0]["run_id"] == "run-1"
    assert usage[0]["thinking_mode"] == "disabled"


def test_validate_artifact_bundle_accepts_complete_synthetic_bundle(tmp_path):
    _complete_bundle(tmp_path)

    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "passed", report
    assert all(report["checks"].values())


def test_validate_artifact_bundle_checks_agent_cases_and_snapshot_hashes(tmp_path):
    _complete_bundle(tmp_path, strategy="tradingagents")
    assert validate_artifact_bundle(tmp_path)["status"] == "passed"

    (tmp_path / "inputs/market_data/AAPL.csv").write_text("tampered", encoding="utf-8")
    (tmp_path / "strategy/cases/AAPL/2024-01-05/decision.json").unlink()
    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "failed"
    assert report["checks"]["market_snapshot_sha256_valid"] is False
    assert report["checks"]["agent_case_artifacts_complete"] is False


@pytest.mark.parametrize("tamper", ("missing", "corrupted"))
def test_artifact_validator_rejects_incomplete_or_tampered_final_memory(
    tmp_path, tamper,
):
    _complete_bundle(tmp_path, strategy="tradingagents")
    memory_path = tmp_path / "memory/symbols/AAPL.md"
    if tamper == "missing":
        memory_path.unlink()
    else:
        memory_path.write_text("tampered final Memory", encoding="utf-8")

    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "failed"
    assert report["checks"]["final_memory_archive_complete_and_valid"] is False
    if tamper == "missing":
        assert "memory/symbols/AAPL.md" in report["missing_files"]
    else:
        assert report["memory_archive_checksum_errors"]


@pytest.mark.parametrize(
    "tamper_target",
    (
        "manifest",
        "config",
        "run_status",
        "graph_config",
        "case_run_context",
        "case_cache_identity",
        "case_source_audit",
    ),
)
def test_validator_rejects_tampered_memory_lineage_at_every_artifact_level(
    tmp_path, tamper_target,
):
    _complete_bundle(tmp_path, strategy="tradingagents")
    case_dir = tmp_path / "strategy/cases/AAPL/2024-01-05"
    targets = {
        "manifest": (tmp_path / "manifest.json", ("memory_lineage_id",)),
        "config": (tmp_path / "config.resolved.json", ("memory_lineage_id",)),
        "run_status": (tmp_path / "run_status.json", ("memory_lineage_id",)),
        "graph_config": (
            tmp_path / "graph_config.resolved.json",
            ("historical_memory_lineage_id",),
        ),
        "case_run_context": (
            case_dir / "run_context.json",
            ("memory_lineage_id",),
        ),
        "case_cache_identity": (
            case_dir / "cache_identity.json",
            ("identity", "memory_lineage_id"),
        ),
        "case_source_audit": (
            case_dir / "source_audit.json",
            ("run_context", "memory_lineage_id"),
        ),
    }
    path, keys = targets[tamper_target]
    document = json.loads(path.read_text())
    target = document
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = "wrong-lineage"
    _write_json(path, document)

    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "failed"
    assert report["checks"]["memory_lineage_provenance_consistent"] is False
    assert report["lineage_errors"]


@pytest.mark.parametrize(
    ("document_name", "field", "value"),
    (
        ("manifest.json", "experiment_id", "other-experiment"),
        ("config.resolved.json", "run_id", "other-run"),
        ("manifest.json", "backtest_protocol_sha256", "c" * 64),
        ("config.resolved.json", "implementation_identity", "other-code"),
        ("manifest.json", "market_input_identity", {"AAPL": "d" * 64}),
    ),
)
def test_validator_rejects_inconsistent_base_run_provenance(
    tmp_path, document_name, field, value,
):
    _complete_bundle(tmp_path)
    path = tmp_path / document_name
    document = json.loads(path.read_text())
    document[field] = value
    _write_json(path, document)

    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "failed"
    assert report["checks"]["memory_lineage_provenance_consistent"] is False
    assert any(field in error for error in report["lineage_errors"])


def test_validator_rejects_cache_status_incompatible_with_memory_lifecycle(tmp_path):
    _complete_bundle(tmp_path, strategy="tradingagents")
    case_dir = tmp_path / "strategy/cases/AAPL/2024-01-05"
    case_path = tmp_path / "analysis_ready/case_index.csv"
    cases = pd.read_csv(case_path)
    cases.loc[0, "cache_status"] = "bypass"
    cases.to_csv(case_path, index=False)
    timeline_path = tmp_path / "analysis_ready/decision_timeline.csv"
    timeline = pd.read_csv(timeline_path)
    timeline.loc[0, "cache_status"] = "bypass"
    timeline.to_csv(timeline_path, index=False)
    for filename in ("case_metadata.json", "decision.json"):
        path = case_dir / filename
        document = json.loads(path.read_text())
        target = document["metadata"] if filename == "decision.json" else document
        target["cache_status"] = "bypass"
        _write_json(path, document)

    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "failed"
    assert report["checks"]["agent_case_artifacts_complete"] is True
    assert report["checks"]["memory_lineage_provenance_consistent"] is False
    assert any("independent_fresh" in error for error in report["lineage_errors"])


@pytest.mark.parametrize(
    "tamper_target",
    ("resolved_graph", "case_index", "model_config", "cache_identity"),
)
def test_validator_rejects_tampered_graph_hash_chain(tmp_path, tamper_target):
    _complete_bundle(tmp_path, strategy="tradingagents")
    case_dir = tmp_path / "strategy/cases/AAPL/2024-01-05"

    if tamper_target == "resolved_graph":
        path = tmp_path / "graph_config.resolved.json"
        value = json.loads(path.read_text())
        value["temperature"] = 0.25
        _write_json(path, value)
    elif tamper_target == "case_index":
        path = tmp_path / "analysis_ready/case_index.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "graph_config_sha256"] = "f" * 64
        frame.to_csv(path, index=False)
    elif tamper_target == "model_config":
        path = case_dir / "model_config.json"
        value = json.loads(path.read_text())
        value["graph_config_sha256"] = "f" * 64
        _write_json(path, value)
    else:
        path = case_dir / "cache_identity.json"
        value = json.loads(path.read_text())
        value["identity"]["graph_config_sha256"] = "f" * 64
        _write_json(path, value)

    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "failed"
    assert (
        report["checks"]["graph_config_sha256_present_and_consistent"] is False
        or report["checks"]["agent_case_artifacts_complete"] is False
    )


def test_validator_rejects_duplicate_case_rows_instead_of_schedule_coverage(tmp_path):
    _complete_bundle(tmp_path, strategy="tradingagents")
    config_path = tmp_path / "config.resolved.json"
    config = json.loads(config_path.read_text())
    config["actual_decision_count"] = 2
    _write_json(config_path, config)
    (tmp_path / "schedule.csv").write_text(
        "decision_session,execution_session\n"
        "2024-01-05,2024-01-08\n"
        "2024-01-12,2024-01-16\n",
        encoding="utf-8",
    )
    case_path = tmp_path / "analysis_ready/case_index.csv"
    cases = pd.read_csv(case_path)
    pd.concat([cases, cases], ignore_index=True).to_csv(case_path, index=False)

    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "failed"
    assert report["checks"]["agent_case_artifacts_complete"] is False
    assert any("duplicate" in error for error in report["agent_artifact_errors"])


@pytest.mark.parametrize(
    "tamper_target",
    ("case_index", "case_metadata", "decision_metadata", "decision_timeline"),
)
def test_validator_rejects_inconsistent_cache_provenance(tmp_path, tamper_target):
    _complete_bundle(tmp_path, strategy="tradingagents")
    case_dir = tmp_path / "strategy/cases/AAPL/2024-01-05"

    if tamper_target == "case_index":
        path = tmp_path / "analysis_ready/case_index.csv"
        document = pd.read_csv(path)
        document.loc[0, "cache_status"] = "bypass"
        document.to_csv(path, index=False)
    elif tamper_target == "case_metadata":
        path = case_dir / "case_metadata.json"
        document = json.loads(path.read_text())
        document["cache_status"] = "bypass"
        _write_json(path, document)
    elif tamper_target == "decision_metadata":
        path = case_dir / "decision.json"
        document = json.loads(path.read_text())
        document["metadata"]["cache_status"] = "bypass"
        _write_json(path, document)
    else:
        path = tmp_path / "analysis_ready/decision_timeline.csv"
        document = pd.read_csv(path)
        document.loc[0, "cache_status"] = "bypass"
        document.to_csv(path, index=False)

    report = validate_artifact_bundle(tmp_path)

    assert report["status"] == "failed"
    assert report["checks"]["agent_case_artifacts_complete"] is False
    assert any("cache_status" in error for error in report["agent_artifact_errors"])


def test_cache_origin_usage_keeps_real_request_provenance(tmp_path):
    _complete_bundle(tmp_path, strategy="tradingagents")
    for filename in ("manifest.json", "config.resolved.json", "run_status.json"):
        path = tmp_path / filename
        document = json.loads(path.read_text())
        document["memory_lifecycle"] = "resume"
        document["memory_resumed_from_run_id"] = "origin-run"
        _write_json(path, document)
    archive_manifest_path = tmp_path / "memory/manifest.json"
    archive_manifest = json.loads(archive_manifest_path.read_text())
    archive_manifest["memory_lifecycle"] = "resume"
    archive_manifest["memory_resumed_from_run_id"] = "origin-run"
    _write_json(archive_manifest_path, archive_manifest)
    top_manifest_path = tmp_path / "manifest.json"
    top_manifest = json.loads(top_manifest_path.read_text())
    top_manifest["memory_archive"]["manifest_sha256"] = hashlib.sha256(
        archive_manifest_path.read_bytes()
    ).hexdigest()
    _write_json(top_manifest_path, top_manifest)
    case_dir = tmp_path / "strategy/cases/AAPL/2024-01-05"
    live_usage_path = case_dir / "llm_usage.json"
    origin_usage = json.loads(live_usage_path.read_text())
    origin_usage[0]["run_id"] = "origin-run"
    origin_usage[0]["origin_run_id"] = "origin-run"
    _write_json(case_dir / "cached_origin_llm_usage.json", origin_usage)
    _write_json(live_usage_path, [])

    case_path = tmp_path / "analysis_ready/case_index.csv"
    cases = pd.read_csv(case_path)
    cases.loc[0, "decision_status"] = "cached"
    cases.loc[0, "cache_status"] = "hit"
    cases.to_csv(case_path, index=False)
    metadata_path = case_dir / "case_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["cache_status"] = "hit"
    _write_json(metadata_path, metadata)
    decision_path = case_dir / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = "cached"
    decision["metadata"]["cache_status"] = "hit"
    _write_json(decision_path, decision)
    timeline_path = tmp_path / "analysis_ready/decision_timeline.csv"
    timeline = pd.read_csv(timeline_path)
    timeline.loc[0, "status"] = "cached"
    timeline.loc[0, "cache_status"] = "hit"
    timeline.to_csv(timeline_path, index=False)

    current_usage = dict(origin_usage[0])
    current_usage.update({
        "run_id": "run-1",
        "usage_source": "cache_origin",
        "origin_run_id": "origin-run",
    })
    write_agent_analysis_tables(
        tmp_path / "analysis_ready",
        case_index=cases,
        data_availability=pd.read_csv(
            tmp_path / "analysis_ready/data_availability.csv"
        ),
        llm_usage=[current_usage],
    )

    report = validate_artifact_bundle(tmp_path)
    usage = pd.read_csv(tmp_path / "analysis_ready/llm_usage.csv").iloc[0]

    assert report["status"] == "passed", report
    assert usage["run_id"] == "run-1"
    assert usage["usage_source"] == "cache_origin"
    assert usage["origin_run_id"] == "origin-run"
