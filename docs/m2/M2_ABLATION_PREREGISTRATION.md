# M2 A1/A2 Ablation Preregistration

This preregistration was created from the frozen M2 architecture before any E2E_PILOT DeepSeek call or performance inspection. Its identity is `e7e2cc8520f9033e57c442b04c882baffd7c51b497bbc9495b71098f64a83b0e`; the bound architecture identity is `35c5a46616f654ce70b0badc01cd59fb0afd433dcdcd99e5e0b8f2419ec4d153`.

## A1 — A1_NO_ONLINE_ADAPTATION

A1 is Full M2 with only delayed per-symbol online fast-parameter adaptation disabled. It starts from the exact C09 global checkpoint and retains the full semantic state, Prompt anchor/prior, Actor/Critic, R3, risk layer, portfolio layer, execution, M1 information, and Formal settings. Matured credits remain audit/scoring records but cannot change any parameter. This estimates the marginal contribution of delayed online adaptation.

## A2 — A2_NO_GLOBAL_PRETRAINING

A2 is Full M2 with only the globally pretrained C09 initialisation replaced by the exact frozen M2-11 initial model, parameter SHA `60a0fec7b69ef2d0576a9c0894be09c377d573585db827c162279fb27483303e` (archived file SHA `4d65dd2c1563b144aee8e79878171feb1ecd990a8bd8e2c584547eaa3a546c9f`). It retains O08 without retuning: LR `1e-3`, two epochs, AdamW, weight decay `1e-4`, gradient clipping `0.5`, the same 165 fast parameters, chronology, risk/portfolio layers, execution, and information environment. This estimates the marginal contribution of global TRAIN-only PA-CTPPO-v2 pretraining.

Neither ablation was chosen using pilot, Holdout, or Formal performance. A1 and A2 branches remain untouched; M2-16 owns branch synchronisation.
