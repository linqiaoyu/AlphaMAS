# Isolated M1 out-of-window pilot harness

This document records the pre-formal runtime harness for the separate
`finmultitime_m1_pilot_aapl_2023q4_4w_v1` input archive. It is a correctness
probe only; it is not an M1 result, performance experiment, or Agent run.

The pilot schedule is exactly:

- `AAPL / 2023-10-06`
- `AAPL / 2023-10-13`
- `AAPL / 2023-10-20`
- `AAPL / 2023-10-27`

The formal archive remains `experiments/M1/inputs/`. The baseline formal input
tree identity recorded before this task was Git tree
`55333b7502d1184eb3fe90743c5d2337e69033ab` (252 files); the corresponding
tree-list SHA-256 was
`eeae76f3d0960abb4642c951a7b2500c9cfffba881f49aac2ff0b33f055de5f7`.
The formal archive was at experiment-repository commit
`e837b1d564b21f307080510833e7875609f75f70`.

The pilot bundle is always selected by explicit runtime configuration:
`finmultitime_bundle_scope=PILOT`, `pilot_only=true`, and the exact packet
manifest/bundle identities in `configs/m1_pilot_aapl_2023q4.json`. FORMAL mode
continues to default to the strict 78-packet identity. The raw source is used
only by `scripts/finmultitime/build_m1_pilot_inputs.py`; the archived runtime
bundle is self-contained and does not require `/Volumes/Jackson/Dataset/FinMultiTime`.

No Qwen, DeepSeek, paid API, Agent, Formal M1, M2, Trader, execution, metrics,
or Memory-algorithm work is performed by this harness.
