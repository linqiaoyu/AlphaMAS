# Formal Experiment Environment Freeze

This repository freezes the Baseline M0 formal experiment environment with
`uv.lock`, resolved for CPython 3.12. The lock is intended to preserve package
versions for the Stage 3 formal run without changing the research contract.

## Scope

- Branch: `baseline-m0`
- Python target: CPython 3.12
- Lock file: `uv.lock`
- Dependency source of truth: `pyproject.toml`
- Formal contract config: `configs/backtest_m0_2024h1.json`
- Memory outcome price mode: `adjusted_close`

The freeze does not change ticker universe, dates, model settings, initial
cash, execution assumptions, Memory horizon, data snapshot contract, or
backtesting logic.

## Recreate

Install `uv`, then create a clean locked development environment:

```bash
uv sync --frozen --extra dev --python 3.12
```

For a named environment outside the default `.venv`, use:

```bash
UV_PROJECT_ENVIRONMENT=.venv-m0-locked uv sync --frozen --extra dev --python 3.12
```

Regenerate the lock only when dependencies are intentionally changed:

```bash
uv lock --python 3.12
```

## Verify

Run the validation suite from the locked environment:

```bash
.venv-m0-locked/bin/python -m pytest -q tests/backtesting
.venv-m0-locked/bin/python -m pytest -q tests/test_point_in_time_contract.py
.venv-m0-locked/bin/python -m pytest -q
.venv-m0-locked/bin/ruff check .
git diff --check
```

Run the bounded yfinance Memory outcome probe:

```bash
.venv-m0-locked/bin/python scripts/probe_memory_yfinance_outcome.py \
  --output artifacts/m0_freeze/memory_yfinance_probe_20260811.json
```

The probe is limited to AAPL versus SPY for decision session `2024-03-28` with
a five-XNYS-session holding horizon. It must use these six sessions:
`2024-03-28`, `2024-04-01`, `2024-04-02`, `2024-04-03`, `2024-04-04`, and
`2024-04-05`.
