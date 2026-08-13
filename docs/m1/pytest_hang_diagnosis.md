# Full Pytest Collection Diagnosis

## Scope and frozen environment

This investigation started at `baseline-m1` HEAD `544c71b7d5b33e8cd31c71b2096d734d5e26f7f8`.
The required command was:

```text
uv sync --frozen --extra dev --python 3.12
```

The resulting interpreter was Python 3.12.10 with pytest 9.1.1 and pluggy 1.6.0.
The auto-loaded third-party pytest plugins were `anyio 4.14.2` and
`langsmith 0.10.17`.

## Reproduction and evidence

The pre-existing local `.venv` was Python 3.14.0. Under that interpreter:

```text
perl -e 'alarm 60; exec @ARGV' -- uv run --frozen --python 3.14 pytest --collect-only -q
```

stalled in collection and exited from the diagnostic alarm with status 142.
The same command with `--collect-only -vv -s` remained at `collecting ...`.
Python faulthandler captured the main thread repeatedly blocked in importlib's
source loading while importing the M0 contract test's eager graph surface:

```text
tests/backtesting/test_m0_contract.py
  -> tradingagents.graph.trading_graph
  -> tradingagents.graph.__init__
  -> tradingagents.agents.__init__
  -> tradingagents.agents.analysts.fundamentals_analyst
  -> tradingagents.agents.utils.agent_utils
```

The focused reproducer was:

```text
uv run --frozen --python 3.14 pytest --collect-only -q tests/backtesting/test_m0_contract.py
```

It stalled, while `test_engine.py`, `test_failure_resume.py`, and the other
test modules collected independently. This identifies the blocking path as
the Python 3.14 import/collection interaction, not a test body, fixture,
network call, provider initialization, or a single pytest plugin.

Under the required Python 3.12 environment, the controls completed normally:

```text
pytest --collect-only -q                         764 tests collected in 3.55s
pytest --collect-only -q -p no:langsmith          764 tests collected in 2.88s
pytest --collect-only -q -p no:anyio              764 tests collected in 2.57s
pytest --collect-only -q -p no:subtests           764 tests collected in 2.55s
```

Disabling plugins was therefore diagnostic only; it was not the fix. No live
provider or paid API call was made during diagnosis.

## Fix and provenance

`tests/conftest.py` now fails fast at pytest session start only when Python
3.14 is active. This is the diagnosed collection-hang environment. Python
3.10, 3.11, 3.12, and 3.13 are not intentionally blocked by the guard. The
formal experiment interpreter remains exactly Python 3.12 and continues to use
the frozen `uv.lock`.

This protection is test infrastructure only. Production code and `uv.lock`
remain unchanged, as do Agent behavior, M0 and M1 evidence, the backtester,
execution, valuation, and metrics.

## Pre-formal correctness erratum rerun — 2026-08-13

The existing Python 3.12 environment was independently bounded before the
corrected bundle was accepted. `import pandas` completed, but
`import exchange_calendars` exceeded 60 seconds while reading the installed
`toolz` package. The old environment was moved to the recoverable path
`.venv.preformal-backup-20260813`; a fresh Python 3.12.10 environment was
created with `uv sync --frozen --extra dev --python 3.12`.

The fresh environment imported pandas in about 9 seconds on first import and
exchange-calendars in about 0.5 seconds. The full suite then completed with:

`798 passed, 2 skipped, 19 warnings, 69 subtests passed in 30.74s`

The two skips were the optional `langchain_aws` Bedrock test and the opt-in
live DeepSeek test. No M1 formal run was performed.
