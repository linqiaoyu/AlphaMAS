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

`tests/conftest.py` now fails fast at pytest session start unless the frozen
Python 3.12 interpreter is active. This is a test-only environment guard. It
prevents an unsupported Python 3.14 invocation from entering the known eager
Agent/Graph import stall and gives the exact repair command. It does not alter
production imports, Agent behavior, M0 evidence, M1 evidence, the backtester,
execution, valuation, metrics, or any dependency/lock file.

The guard is regression protection for the diagnosed environment failure. The
validated test environment remains Python 3.12 and uses the frozen `uv.lock`.
