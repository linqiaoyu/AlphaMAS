import hashlib
import json
from pathlib import Path

import pytest

from tradingagents.backtesting.memory_archive import (
    MEMORY_ARCHIVE_MANIFEST_PATH,
    archive_final_experiment_memory,
    runtime_experiment_memory_path,
    validate_final_memory_archive,
)

EXPERIMENT_ID = "M0"
RUN_ID = "run-1"
GRAPH_HASH = "a" * 64
SYMBOLS = ["AAPL", "AMZN", "JPM"]


def _seed_runtime_memory(runtime_root: Path, symbols=SYMBOLS) -> dict[str, Path]:
    sources = {}
    for index, symbol in enumerate(symbols):
        path = runtime_experiment_memory_path(
            runtime_root,
            experiment_id=EXPERIMENT_ID,
            graph_config_sha256=GRAPH_HASH,
            memory_lineage_id=RUN_ID,
            symbol=symbol,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"[2024-01-0{index + 5} | {symbol} | Buy | pending]\n\n"
            f"DECISION:\n{symbol} final decision\n\n"
            "<!-- ENTRY_END -->\n\n",
            encoding="utf-8",
        )
        sources[symbol] = path
    return sources


def _archive(run_root: Path, runtime_root: Path) -> dict:
    return archive_final_experiment_memory(
        run_dir=run_root,
        runtime_memory_dir=runtime_root,
        experiment_id=EXPERIMENT_ID,
        run_id=RUN_ID,
        memory_lineage_id=RUN_ID,
        memory_lifecycle="independent_fresh",
        memory_resumed_from_run_id=None,
        graph_config_sha256=GRAPH_HASH,
        symbols=SYMBOLS,
    )


def _validate(run_root: Path, descriptor: dict) -> dict[str, list[str]]:
    return validate_final_memory_archive(
        run_dir=run_root,
        descriptor=descriptor,
        experiment_id=EXPERIMENT_ID,
        run_id=RUN_ID,
        memory_lineage_id=RUN_ID,
        memory_lifecycle="independent_fresh",
        memory_resumed_from_run_id=None,
        graph_config_sha256=GRAPH_HASH,
        symbols=SYMBOLS,
    )


def test_final_memory_archive_is_complete_and_independent_of_runtime(tmp_path):
    run_root = tmp_path / "experiment" / "runs" / RUN_ID
    runtime_root = tmp_path / "experiment" / "runtime" / "memory"
    sources = _seed_runtime_memory(runtime_root)
    original = {symbol: path.read_bytes() for symbol, path in sources.items()}

    descriptor = _archive(run_root, runtime_root)
    archive_manifest = json.loads(
        (run_root / MEMORY_ARCHIVE_MANIFEST_PATH).read_text(encoding="utf-8")
    )

    assert descriptor["symbol_count"] == 3
    assert {item["symbol"] for item in archive_manifest["symbols"]} == set(SYMBOLS)
    assert archive_manifest["experiment_id"] == EXPERIMENT_ID
    assert archive_manifest["run_id"] == RUN_ID
    assert archive_manifest["memory_lineage_id"] == RUN_ID
    assert archive_manifest["graph_config_sha256"] == GRAPH_HASH
    assert archive_manifest["runtime_consumed"] is False
    for symbol, source in sources.items():
        archived = run_root / f"memory/symbols/{symbol}.md"
        assert archived.read_bytes() == original[symbol]
        assert not archived.is_symlink()
        assert not source.samefile(archived)

        # Subsequent runtime mutation and deletion cannot alter the archive.
        source.write_text("changed runtime state", encoding="utf-8")
        source.unlink()
        assert archived.read_bytes() == original[symbol]

    assert _validate(run_root, descriptor) == {
        "errors": [], "missing_files": [], "checksum_errors": [],
    }


def test_archive_requires_every_configured_symbol_before_writing(tmp_path):
    run_root = tmp_path / "experiment" / "runs" / RUN_ID
    runtime_root = tmp_path / "experiment" / "runtime" / "memory"
    _seed_runtime_memory(runtime_root, symbols=["AAPL", "JPM"])

    with pytest.raises(ValueError, match="AMZN"):
        _archive(run_root, runtime_root)

    assert not (run_root / "memory").exists()


@pytest.mark.parametrize(
    ("tamper", "expected_bucket"),
    (
        ("symbol_content", "checksum_errors"),
        ("missing_symbol", "missing_files"),
        ("manifest_provenance", "errors"),
        ("runtime_symlink", "missing_files"),
    ),
)
def test_memory_archive_validator_detects_tampering(
    tmp_path, tamper, expected_bucket,
):
    run_root = tmp_path / "experiment" / "runs" / RUN_ID
    runtime_root = tmp_path / "experiment" / "runtime" / "memory"
    sources = _seed_runtime_memory(runtime_root)
    descriptor = _archive(run_root, runtime_root)
    archived_aapl = run_root / "memory/symbols/AAPL.md"

    if tamper == "symbol_content":
        archived_aapl.write_text("tampered", encoding="utf-8")
    elif tamper == "missing_symbol":
        archived_aapl.unlink()
    elif tamper == "runtime_symlink":
        archived_aapl.unlink()
        archived_aapl.symlink_to(sources["AAPL"])
    else:
        manifest_path = run_root / MEMORY_ARCHIVE_MANIFEST_PATH
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["memory_lineage_id"] = "other-lineage"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        # Even if a tamperer updates the outer checksum, cross-document
        # provenance still detects the altered lineage.
        descriptor["manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()

    report = _validate(run_root, descriptor)

    assert report[expected_bucket]
