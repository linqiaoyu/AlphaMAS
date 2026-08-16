# M2-12 Global VALIDATION Selection Contract

STATUS: PREREGISTERED — VALIDATION PERFORMANCE NOT YET INSPECTED

M2-12 compares only the nine frozen M2-11 canonical candidates C01–C09. Each
static policy is evaluated, without gradients or adaptation, on the frozen 16-case
VALIDATION population (eight independent symbols, 2023-06-30 then 2023-07-07).
Each symbol starts with $100,000 cash and zero quantity. The first action advances
portfolio state to the second decision close; its fifth-close R3 maturity remains
independent of that child observation.

The primary metric is the equal-weight mean of the 16 chosen-action sequential
local `R3_HOLD_RELATIVE_DRAWDOWN_UTILITY` values. Values within `1e-8` tie, then
the selector applies only: higher worst-symbol cumulative R3, lower Prompt
override rate, smaller learning rate, earlier checkpoint. Human override is
prohibited. FINAL_HOLDOUT, E2E_PILOT, Formal 2024H1, live market data, model APIs,
training, fine-tuning, and online adaptation are prohibited.
