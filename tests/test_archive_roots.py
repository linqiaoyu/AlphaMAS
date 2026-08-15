from pathlib import Path

import pytest

from tests.archive_roots import (
    ExperimentsArchiveUnavailable,
    resolve_experiments_root,
)


def test_explicit_experiments_root_is_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured"
    configured.mkdir()
    repository = tmp_path / "elsewhere" / "AlphaMAS"
    repository.mkdir(parents=True)
    monkeypatch.setenv("ALPHAMAS_EXPERIMENTS_ROOT", str(configured))

    assert resolve_experiments_root(repository) == configured


def test_invalid_explicit_experiments_root_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "AlphaMAS"
    repository.mkdir()
    (tmp_path / "AlphaMAS-Experiments").mkdir()
    monkeypatch.setenv("ALPHAMAS_EXPERIMENTS_ROOT", str(tmp_path / "missing"))

    with pytest.raises(ValueError, match="explicitly set.*invalid"):
        resolve_experiments_root(repository)


def test_sibling_experiments_root_is_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "AlphaMAS"
    repository.mkdir()
    sibling = tmp_path / "AlphaMAS-Experiments"
    sibling.mkdir()
    monkeypatch.delenv("ALPHAMAS_EXPERIMENTS_ROOT", raising=False)

    assert resolve_experiments_root(repository) == sibling


def test_missing_experiments_root_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "AlphaMAS"
    repository.mkdir()
    monkeypatch.delenv("ALPHAMAS_EXPERIMENTS_ROOT", raising=False)

    with pytest.raises(ExperimentsArchiveUnavailable, match="archive unavailable"):
        resolve_experiments_root(repository)
