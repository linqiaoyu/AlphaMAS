# M2-09 Semantic State Representation Freeze

## Verdict and scientific purpose

`M2-SEMANTIC-STATE-v1` is research-frozen. It converts the 72 immutable M2-08 semantic
hand-offs into compact Actor inputs while preserving the economic distinction between the
Research Manager plan and Prompt Trader proposal. It represents what the hand-off means,
not how the experiment is organised.

The source is `M2-SEMANTIC-HANDOFF-v1`, identity
`cc1fea692489a5b5791cae1fd96386bfbcf1f0a07f2ea3e22637dd3c43533a43`: 56 TRAIN and
16 VALIDATION states in the corpus-manifest order. FINAL_HOLDOUT and E2E_PILOT states were
not materialised or read.

## Economic payload and metadata boundary

The only static learnable fields are the exact, verbatim
`research_manager.investment_plan`, exact `prompt_trader.rendered_proposal`, and explicit
`prompt_trader.normalized_action`. Future endogenous portfolio state is the only dynamic
learnable input. Every other field is audit metadata. Role, case ID, symbol/ticker,
company/instrument context, decision date/time, evidence identity, Git SHA, model/run
provenance, and archive path are never direct encoder, Actor, or Critic features. Natural
asset, price, and date mentions already present in the two frozen texts remain verbatim.

The two texts are encoded separately. They are never concatenated before encoding and no
custom instruction is added:

```text
z_RM = Encoder(exact investment_plan)
z_PT = Encoder(exact rendered_proposal)
```

## Frozen encoder and environment

The only encoder is `Qwen/Qwen3-Embedding-0.6B` at immutable revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. The 12 locally used snapshot files are
size/SHA-bound by model snapshot identity
`1d7b1bddebe83694815066f5254c5b0c7a1d05febd4e2b9e2120f2ec3fe3c018`.
Weights remain in the persistent Hugging Face cache and are not archived in Git.

The encoder is permanently frozen: no training, fine-tuning, LoRA, adapter, or RL gradient.
Canonical inference used Python 3.12.10, torch 2.7.1+cu126, CUDA 12.6, NVIDIA driver
595.91.07, transformers 4.57.6, sentence-transformers 5.1.2, huggingface-hub 0.36.0,
NumPy 2.2.6, and SciPy 1.15.3 on an NVIDIA A10G. It used float32, `eval()`,
`inference_mode()`, eager attention, deterministic algorithms, TF32 disabled, batch size 1,
left padding, offline model loading, and a 32,768-token ceiling without truncation.

Token maxima were 1,178 TRAIN manager, 411 TRAIN trader, 1,223 VALIDATION manager, and
441 VALIDATION trader tokens. Texts over 32,768: 0. Truncated texts: 0.

## TRAIN-only dimension selection

The full reference dimension was 1024. Only the preregistered MRL prefix dimensions 256,
512, and 1024 were evaluated. Each prefix was L2-normalised. No PCA, projection,
autoencoder, reward, return, action correctness, validation geometry, validation trading
performance, or manual override influenced selection. The smallest candidate passing all
three thresholds in all three TRAIN views was selected: Spearman at least 0.98, cosine MAE
at most 0.02, and mean top-5 overlap at least 0.80.

| Dimension | View | Spearman | Cosine MAE | Top-5 overlap | Pass |
|---:|---|---:|---:|---:|:---:|
| 256 | Manager | 0.9239673591 | 0.0783741436 | 0.9357142857 | No |
| 256 | Trader | 0.9453284428 | 0.0626225956 | 0.8714285714 | No |
| 256 | Joint | 0.9409056172 | 0.0704651473 | 0.9214285714 | No |
| 512 | Manager | 0.9656454668 | 0.0422279137 | 0.9607142857 | No |
| 512 | Trader | 0.9785764104 | 0.0341616098 | 0.9142857143 | No |
| 512 | Joint | 0.9747369689 | 0.0381202611 | 0.9535714286 | No |
| 1024 | Manager | 1.0000000000 | 0.0000000000 | 1.0000000000 | Yes |
| 1024 | Trader | 1.0000000000 | 0.0000000000 | 1.0000000000 | Yes |
| 1024 | Joint | 1.0000000000 | 0.0000000000 | 1.0000000000 | Yes |

The automatically selected and now frozen embedding dimension is 1024.

## Exact representation and future observation

The Prompt Trader action prior is one-hot in permanent `[BUY, HOLD, SELL]` order. The
Manager-to-Trader residual `delta_z = z_PT - z_RM` remains explicit. Semantic agreement is
`dot(z_RM, z_PT)` and lies in `[-1, 1]` because both channels are L2-normalised.

```text
semantic_base = concat(z_RM, z_PT, delta_z, agreement, action_onehot)
semantic_base dimension = 3(1024) + 4 = 3076
```

The 72 float32 semantic-base rows are materialised. The future runtime portfolio vector is
numeric, not embedded text:

```text
[is_cash, is_long, entry_log_return, current_drawdown]
```

`Portfolio.tolerance` determines cash versus long. Entry log return is zero in cash and
`ln(close_price / average_entry_price)` in a valid long state. Current drawdown is the
decision-time non-positive snapshot value. Raw dollars, quantity, market value, symbol, and
timestamp are excluded. Portfolio dimension is 4, so the frozen future Actor observation
dimension is `3076 + 4 = 3080`.

## Reproducibility, lineage, and resources

The two canonical inference runs were byte-identical: maximum absolute difference 0,
minimum corresponding cosine 0.9999998808, and identical selected dimension. All 72 manager,
72 trader, and 72 semantic-base rows are finite and SHA-bound. Representation identity is
`6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe`.

The existing M2 `g5.xlarge`/A10G, EBS, S3, and SSM environment was reused. No AWS resource
was created. GPU session time was 416 seconds (0.115556 hours); the canonical runner used
34.398 seconds. At $1.277/hour, estimated incremental compute cost is $0.147582. The EC2
instance is stopped and no AlphaMAS GPU instance is running.

Starting source SHA: `070b4f30743db2e3ad8b7a5af55b2e808a6b1550`.
Starting Experiments SHA: `d7ba1a3d8a77ede15be82c1b700abd5092066526`.
Representation runner SHA: `cb3a540f87e6daad13665132ac910c484252f636`.
Preregistered plan SHA: `fe629d4d039edefb0405f3b69492416fe301bd61`.
Final Experiments archive SHA: `1a94fdd20c6c3c004cbe5c2340171e86922c49f7`.

DeepSeek calls and cost were 0/¥0. Qwen3-VL calls were 0. Qwen3-Embedding was local only:
144 texts per run, two runs, API cost ¥0. No reward/outcome or raw FinMultiTime data was
accessed. The M2-08/M2-06 corpora, MAXIMUM membership, Prompt Trader, Research Manager,
other MAS agents, R1/R2/R3, and Formal protocol are unchanged.

## M2-10 boundary

M2-10 may combine this frozen semantic representation with frozen R3 reward chronology,
TRAIN temporal sequences, and runtime portfolio state. Actor, Critic, PPO-family method,
optimizer, learning rate, and online adapter remain deferred. M2-10 must not regenerate
DeepSeek semantic states or use FINAL_HOLDOUT.
