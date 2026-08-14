# M2-02A Local Environment Recovery

## Task identity and verdict

- Project: AlphaMAS MSc Dissertation
- Stage/task: M2 / M2-02A
- Starting branch: `baseline-m2`
- Starting and authoritative architecture SHA: `ea9ca73c15ed94ba3fa37a86e3ee145960d94bc2`
- Frozen M1 source SHA: `ac0d1b006d8019748702fda38399a4316befb9b0`
- Formal M1 run: `20260814T015553499023Z_ac0d1b00`
- Verdict: **PASS — the locked environment was reconstructed with Formal-M1 key-version equivalence and pytest execution was restored.**

No M2 research, data selection, RL implementation, experiment, paid LLM call, or cloud compute occurred.

## Starting-state and host verification

The worktree was clean on `baseline-m2`; local `HEAD` and the existing `origin/baseline-m2` ref both resolved to `ea9ca73c15ed94ba3fa37a86e3ee145960d94bc2`. The first two `git fetch origin` attempts stalled because Git administrative files in the Desktop checkout were being evicted by macOS. An independent clone from GitHub subsequently completed immediately and resolved both `HEAD` and `origin/baseline-m2` to the required SHA, ruling out remote branch drift.

Host and tooling:

```text
OS                    macOS 26.6.1 (Build 25G76), Darwin 25.6.0
Machine               arm64
uv                     0.12.3, /opt/homebrew/bin/uv
Global python3         3.14.0, /opt/homebrew/bin/python3 (not used for the project)
Required CPython       3.12.10, /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
```

## Original symptoms and suspect environment

The failed M2-02 attempt had completed `uv sync --frozen --extra dev --python 3.12`, but NumPy/Pandas imports and pytest collection took minutes or did not complete. The preserved suspect `.venv` had:

```text
Python                 CPython 3.12.10
Machine                arm64
Interpreter            universal Mach-O with arm64 and x86_64 slices
sys.prefix             <repository>/.venv
User site              disabled
Project site-packages  <repository>/.venv/lib/python3.12/site-packages
```

No `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV`, `DYLD_*`, `OMP_*`, `OPENBLAS_*`, `MKL_*`, or `VECLIB_*` variables affected the diagnostic shell. Package metadata already reported the expected versions, and sampled NumPy extension modules were native arm64 Mach-O binaries.

## Layered diagnostics and root cause

The suspect environment passed a standard-library import in 1.994 seconds. Independent 25-second probes then timed out for NumPy, pandas, exchange-calendars, yfinance, and tradingagents. `faulthandler` consistently located the main thread in `importlib._bootstrap_external.get_data`, at different Python modules. Native sampling likewise showed blocking `read(2)` calls rather than BLAS initialization or a native-extension crash.

The decisive evidence was the macOS file state:

```text
Old numpy/_core/multiarray.py   hidden,compressed,dataless
Old pandas/__init__.py          hidden,compressed,dataless
Fresh resident copies           no dataless flag
```

The old and fresh copies had identical sizes and SHA-256 hashes, but reads differed materially:

| File | Suspect `.venv` | Resident copy | Content |
|---|---:|---:|---|
| `numpy/_core/multiarray.py` | 18.892 s | 0.0014 s | identical |
| `pandas/__init__.py` | 8.381 s | 0.0005 s | identical |
| NumPy `_multiarray_umath` extension | 0.0022 s | 0.0027 s | identical, arm64 |

The same `dataless` state affected repository Python sources and Git objects under the Desktop path. A fresh venv created inside the repository also had unaccessed dependencies evicted within minutes, and pytest blocked on those reads. This explains the initial import stalls, pytest silence, and `git fetch` stall.

Best-supported root cause: **macOS cloud/storage optimization evicted hidden virtual-environment, bytecode-cache, source, and Git-object files from the Desktop-hosted repository.** The Python interpreter, wheel architecture, locked versions, and package contents were correct. No package cache clean, package reinstall, version change, or lock regeneration was warranted.

## Recovery method and canonical environment

The first in-repository reconstruction used the unchanged lock and exact interpreter:

```bash
UV_PROJECT_ENVIRONMENT=.venv-m2-locked \
uv sync --frozen --extra dev \
  --python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
```

It proved the lock was healthy but was itself subject to Desktop eviction. That generated directory was moved to Trash and replaced by a symlink to a resident, non-synced environment reconstructed with the same command and lock:

```text
Canonical project path  .venv-m2-locked
Symlink target           /Users/yulinqiao/.local/share/alphamas/venvs/m2-locked
Bytecode cache           /Users/yulinqiao/.local/share/alphamas/pycache/m2-locked
```

The external bytecode-cache location is set by the recovered environment's local `sitecustomize.py`; it changes only cache placement, not runtime source or dependency resolution. `.venv-m2-locked` is excluded locally through `.git/info/exclude`, so no ignore rule or tracked dependency file changed.

For a fully resident smoke check, an independent clone at `/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-m2-02a` was created from GitHub. It resolved to the required starting SHA and used the same `.venv-m2-locked` target. This avoided the Desktop checkout's recurring source and Git-object eviction.

## Formal-M1 equivalence audit

Reference: `results/backtests/M1_finmultitime_prompt_2024H1/runs/20260814T015553499023Z_ac0d1b00/environment.json`. Only environment/provenance fields were inspected.

| Component | Formal M1 | Recovered M2 local | Result |
|---|---:|---:|---|
| Python | 3.12.10 | 3.12.10 | exact |
| Platform | macOS arm64 | macOS arm64 | exact |
| pandas | 2.3.3 | 2.3.3 | exact |
| numpy | 2.5.2 | 2.5.2 | exact |
| exchange-calendars | 4.13.2 | 4.13.2 | exact |
| yfinance | 1.5.2 | 1.5.2 | exact |
| tradingagents | 0.3.1 | 0.3.1 | exact |

Recovered import timings were bounded and healthy: NumPy 0.862 s, pandas 8.421 s, exchange-calendars 0.473 s, yfinance 2.755 s, and top-level tradingagents 0.420-5.343 s across cold probes. All were below the 30-second threshold.

## Test and smoke results

Required tests in the recovered environment after source materialization:

```text
tests/backtesting/test_m1_formal_contract.py  21 passed in 18.95s
tests/test_m1_runtime_evidence.py             23 passed in 24.10s
tests/test_point_in_time_contract.py          19 passed in 2.36s
```

Broader smoke in the exact-SHA resident clone:

```text
tests/backtesting                              205 passed in 7.52s
git diff --check                              PASS
```

`ruff check .` executed normally and reported seven pre-existing findings at the untouched starting SHA: one B007 in `audit_m1_preformal_bundle.py`; F401/E402/I001 findings in `audit_m1_text_integrity.py`; and I001/F401 findings in `preprocess_m1_inputs.py`. M2-02A did not modify those frozen Formal-M1 scripts.

## Dependency-drift audit

`pyproject.toml`, `uv.lock`, and `requirements.txt` were unchanged. The environment was created with `uv sync --frozen`; no `uv lock`, global pip operation, cache clean, package upgrade/downgrade, or dependency-source change was performed. The original suspect `.venv` was preserved.

## Future test convention

Use the explicit recovered interpreter rather than global `pytest`:

```bash
.venv-m2-locked/bin/python -m pytest ...
```

Because the original repository is under a Desktop location subject to recurring eviction, future M2 work should run from resident local storage (the external clone above or a replacement non-synced checkout). If the Desktop checkout is used, first ensure the repository is kept downloaded/materialized; the external venv alone cannot prevent source or `.git` eviction.

This is a local reconstruction of the frozen Formal-M1 environment, not a change to the Formal experiment environment.

## Cost

```text
DeepSeek API calls:     0
DeepSeek API cost:      ¥0
Qwen inference calls:  0
AWS GPU hours:          0
AWS incremental cost:  $0
```

Network access was limited to frozen package installation and Git remote verification. No experiment occurred.
