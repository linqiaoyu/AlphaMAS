import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tradingagents.agents.utils.memory_namespace import (
    experiment_memory_namespace_component,
    runtime_experiment_memory_path,
)
from tradingagents.backtesting.memory_archive import (
    archive_final_experiment_memory,
    validate_final_memory_archive,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runtime.run_context import RunContext

GRAPH_SHA = "0b707807d747afba827e6bf76b6f182a7a596ee42ccdb3ee5d861ae67dbf2488"
PILOT_IDENTITY = "bd8dfafdbeb259fc8bac7ee3cdbfeebdc13f8abde8b1b420e494fa4ae8651ba3"
FORMAL_IDENTITY = "f" * 64
EXPERIMENT = "M1_pilot_aapl_2023q4_4w_v1"
LINEAGE = "20260813T221516239387Z_7e765a73"


def _path(tmp_path: Path, *, enabled=False, scope=None, identity=None, **overrides):
    return runtime_experiment_memory_path(
        tmp_path / "runtime" / "memory",
        experiment_id=overrides.get("experiment_id", EXPERIMENT),
        graph_config_sha256=overrides.get("graph_config_sha256", GRAPH_SHA),
        memory_lineage_id=overrides.get("memory_lineage_id", LINEAGE),
        symbol=overrides.get("symbol", "AAPL"),
        finmultitime_evidence_enabled=enabled,
        finmultitime_bundle_scope=scope,
        finmultitime_bundle_identity=identity,
    )


def _graph_stub(
    tmp_path: Path, *, enabled=False, scope=None, identity=None,
    graph_hash=GRAPH_SHA,
) -> TradingAgentsGraph:
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "data_cache_dir": str(tmp_path / "runtime" / "cache"),
        "historical_memory_dir": str(tmp_path / "runtime" / "memory"),
        "historical_memory_log_path": None,
        "memory_mode": "experiment",
        "graph_config_sha256": graph_hash,
        "finmultitime_evidence_enabled": enabled,
        "finmultitime_bundle_scope": scope,
        "finmultitime_expected_input_bundle_identity": identity,
    }
    return graph


def test_shared_resolver_preserves_m0_and_isolates_pilot_formal_namespaces(tmp_path):
    m0 = _path(tmp_path)
    pilot = _path(
        tmp_path, enabled=True, scope="PILOT", identity=PILOT_IDENTITY,
    )
    formal = _path(
        tmp_path, enabled=True, scope="FORMAL", identity=FORMAL_IDENTITY,
    )

    assert m0 == (
        tmp_path / "runtime/memory/historical_memory" / EXPERIMENT
        / GRAPH_SHA / LINEAGE / "AAPL.md"
    )
    assert "/historical_memory/finmultitime-pilot-bd8dfafdbeb259fc/" in str(pilot)
    assert "/historical_memory/finmultitime-formal-ffffffffffffffff/" in str(formal)
    assert len({m0, pilot, formal}) == 3


def test_graph_and_archiver_resolve_exactly_the_same_m1_path(tmp_path):
    graph = _graph_stub(
        tmp_path, enabled=True, scope="PILOT", identity=PILOT_IDENTITY,
    )
    graph_path = Path(graph._historical_memory_config(
        "AAPL",
        RunContext.historical(
            "2023-10-06", experiment_id=EXPERIMENT, memory_lineage_id=LINEAGE,
        ),
    )["memory_log_path"])
    archive_path = _path(
        tmp_path, enabled=True, scope="PILOT", identity=PILOT_IDENTITY,
    )

    assert graph_path == archive_path


@pytest.mark.parametrize(
    ("scope", "identity", "match"),
    (
        ("PILOT", None, "bundle identity"),
        ("OTHER", PILOT_IDENTITY, "bundle scope"),
        ("PILOT", "g" * 64, "bundle identity"),
    ),
)
def test_enabled_namespace_rejects_missing_or_malformed_identity(scope, identity, match):
    with pytest.raises(ValueError, match=match):
        experiment_memory_namespace_component(
            finmultitime_evidence_enabled=True,
            finmultitime_bundle_scope=scope,
            finmultitime_bundle_identity=identity,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"experiment_id": "../escape"},
        {"memory_lineage_id": ""},
        {"graph_config_sha256": "not-a-sha"},
        {"symbol": "../AAPL"},
    ),
)
def test_namespace_rejects_unsafe_or_invalid_path_components(tmp_path, kwargs):
    with pytest.raises(ValueError):
        _path(tmp_path, **kwargs)


def _archive_m1(tmp_path: Path, *, scope: str, identity: str):
    runtime_root = tmp_path / "runtime"
    run_root = tmp_path / "run"
    source = runtime_experiment_memory_path(
        runtime_root,
        experiment_id=EXPERIMENT,
        graph_config_sha256=GRAPH_SHA,
        memory_lineage_id=LINEAGE,
        symbol="AAPL",
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope=scope,
        finmultitime_bundle_identity=identity,
    )
    source.parent.mkdir(parents=True)
    source.write_text("[2023-10-06 | AAPL | HOLD | pending]\n", encoding="utf-8")
    descriptor = archive_final_experiment_memory(
        run_dir=run_root,
        runtime_memory_dir=runtime_root,
        experiment_id=EXPERIMENT,
        run_id=LINEAGE,
        memory_lineage_id=LINEAGE,
        memory_lifecycle="independent_fresh",
        memory_resumed_from_run_id=None,
        graph_config_sha256=GRAPH_SHA,
        symbols=["AAPL"],
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope=scope,
        finmultitime_bundle_identity=identity,
    )
    report = validate_final_memory_archive(
        run_dir=run_root,
        descriptor=descriptor,
        experiment_id=EXPERIMENT,
        run_id=LINEAGE,
        memory_lineage_id=LINEAGE,
        memory_lifecycle="independent_fresh",
        memory_resumed_from_run_id=None,
        graph_config_sha256=GRAPH_SHA,
        symbols=["AAPL"],
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope=scope,
        finmultitime_bundle_identity=identity,
    )
    return source, run_root, descriptor, report


@pytest.mark.parametrize(
    ("scope", "identity", "namespace"),
    (
        ("PILOT", PILOT_IDENTITY, "finmultitime-pilot-bd8dfafdbeb259fc"),
        ("FORMAL", FORMAL_IDENTITY, "finmultitime-formal-ffffffffffffffff"),
    ),
)
def test_m1_archive_and_validator_preserve_namespace_provenance(
    tmp_path, scope, identity, namespace,
):
    source, run_root, _, report = _archive_m1(
        tmp_path, scope=scope, identity=identity,
    )
    manifest = json.loads(
        (run_root / "memory/manifest.json").read_text(encoding="utf-8")
    )
    archived = run_root / "memory/symbols/AAPL.md"

    assert report == {"errors": [], "missing_files": [], "checksum_errors": []}
    assert manifest["runtime_namespace"] == {
        "storage_role": "operational_runtime_only",
        "layout_version": "historical-memory-v1",
        "experiment_id": EXPERIMENT,
        "memory_lineage_id": LINEAGE,
        "graph_config_sha256": GRAPH_SHA,
        "finmultitime_evidence_enabled": True,
        "bundle_scope": scope,
        "bundle_identity": identity,
        "namespace_component": namespace,
    }
    assert archived.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "wrong_kwargs",
    (
        {"finmultitime_bundle_scope": "FORMAL", "finmultitime_bundle_identity": PILOT_IDENTITY},
        {"finmultitime_bundle_scope": "PILOT", "finmultitime_bundle_identity": FORMAL_IDENTITY},
    ),
)
def test_archiver_fails_closed_on_wrong_m1_namespace(tmp_path, wrong_kwargs):
    source = _path(
        tmp_path, enabled=True, scope="PILOT", identity=PILOT_IDENTITY,
    )
    source.parent.mkdir(parents=True)
    source.write_text("fixture", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot archive complete"):
        archive_final_experiment_memory(
            run_dir=tmp_path / "run",
            runtime_memory_dir=tmp_path / "runtime" / "memory",
            experiment_id=EXPERIMENT,
            run_id=LINEAGE,
            memory_lineage_id=LINEAGE,
            memory_lifecycle="independent_fresh",
            memory_resumed_from_run_id=None,
            graph_config_sha256=GRAPH_SHA,
            symbols=["AAPL"],
            finmultitime_evidence_enabled=True,
            **wrong_kwargs,
        )
    assert not (tmp_path / "run/memory").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("bundle_scope", "FORMAL"),
        ("bundle_identity", FORMAL_IDENTITY),
        ("namespace_component", "finmultitime-pilot-wrong"),
    ),
)
def test_validator_rejects_tampered_m1_namespace_provenance(tmp_path, field, value):
    _, run_root, descriptor, _ = _archive_m1(
        tmp_path, scope="PILOT", identity=PILOT_IDENTITY,
    )
    manifest_path = run_root / "memory/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_namespace"][field] = value
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    descriptor["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    report = validate_final_memory_archive(
        run_dir=run_root,
        descriptor=descriptor,
        experiment_id=EXPERIMENT,
        run_id=LINEAGE,
        memory_lineage_id=LINEAGE,
        memory_lifecycle="independent_fresh",
        memory_resumed_from_run_id=None,
        graph_config_sha256=GRAPH_SHA,
        symbols=["AAPL"],
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope="PILOT",
        finmultitime_bundle_identity=PILOT_IDENTITY,
    )

    assert report["errors"]


def test_preserved_failed_pilot_memory_archives_without_touching_fixture(tmp_path):
    fixture = Path(
        "/Users/yulinqiao/Desktop/AlphaMAS-Experiments/experiments/M1/pilot/runs/"
        "20260813T221516239387Z_7e765a73/provenance/memory/historical_memory/"
        "finmultitime-pilot-bd8dfafdbeb259fc/M1_pilot_aapl_2023q4_4w_v1/"
        f"{GRAPH_SHA}/{LINEAGE}/AAPL.md"
    )
    if not fixture.is_file():
        pytest.skip("preserved failed-pilot Memory fixture is not available")
    fixture_bytes = fixture.read_bytes()
    fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()
    runtime_source = runtime_experiment_memory_path(
        tmp_path / "runtime",
        experiment_id=EXPERIMENT,
        graph_config_sha256=GRAPH_SHA,
        memory_lineage_id=LINEAGE,
        symbol="AAPL",
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope="PILOT",
        finmultitime_bundle_identity=PILOT_IDENTITY,
    )
    runtime_source.parent.mkdir(parents=True)
    shutil.copyfile(fixture, runtime_source)

    run_root = tmp_path / "archived-run"
    descriptor = archive_final_experiment_memory(
        run_dir=run_root,
        runtime_memory_dir=tmp_path / "runtime",
        experiment_id=EXPERIMENT,
        run_id=LINEAGE,
        memory_lineage_id=LINEAGE,
        memory_lifecycle="independent_fresh",
        memory_resumed_from_run_id=None,
        graph_config_sha256=GRAPH_SHA,
        symbols=["AAPL"],
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope="PILOT",
        finmultitime_bundle_identity=PILOT_IDENTITY,
    )
    archived = run_root / "memory/symbols/AAPL.md"
    report = validate_final_memory_archive(
        run_dir=run_root,
        descriptor=descriptor,
        experiment_id=EXPERIMENT,
        run_id=LINEAGE,
        memory_lineage_id=LINEAGE,
        memory_lifecycle="independent_fresh",
        memory_resumed_from_run_id=None,
        graph_config_sha256=GRAPH_SHA,
        symbols=["AAPL"],
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope="PILOT",
        finmultitime_bundle_identity=PILOT_IDENTITY,
    )

    assert runtime_source == (
        tmp_path / "runtime/historical_memory/finmultitime-pilot-bd8dfafdbeb259fc"
        / EXPERIMENT / GRAPH_SHA / LINEAGE / "AAPL.md"
    )
    assert runtime_source.stat().st_size == len(fixture_bytes)
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == fixture_sha
    assert archived.read_bytes() == fixture_bytes
    assert report == {"errors": [], "missing_files": [], "checksum_errors": []}
