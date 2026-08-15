"""Portable resolution of the external AlphaMAS-Experiments repository."""

from __future__ import annotations

import os
from pathlib import Path

EXPERIMENTS_ROOT_ENV = "ALPHAMAS_EXPERIMENTS_ROOT"


class ExperimentsArchiveUnavailable(RuntimeError):
    """Raised when no AlphaMAS-Experiments repository can be resolved."""


def resolve_experiments_root(repository_root: Path) -> Path:
    """Resolve an explicitly configured or sibling experiments repository.

    An explicit path is authoritative and therefore fails closed when invalid.
    """
    if EXPERIMENTS_ROOT_ENV in os.environ:
        configured = Path(
            os.path.expandvars(os.environ[EXPERIMENTS_ROOT_ENV])
        ).expanduser()
        experiments_root = configured.resolve()
        if not experiments_root.is_dir():
            raise ValueError(
                f"{EXPERIMENTS_ROOT_ENV} is explicitly set to an invalid "
                f"AlphaMAS-Experiments root: {experiments_root}"
            )
        return experiments_root

    experiments_root = repository_root.resolve().parent / "AlphaMAS-Experiments"
    if experiments_root.is_dir():
        return experiments_root.resolve()
    raise ExperimentsArchiveUnavailable(
        "AlphaMAS-Experiments archive unavailable. Set "
        f"{EXPERIMENTS_ROOT_ENV} or provide a sibling checkout."
    )
