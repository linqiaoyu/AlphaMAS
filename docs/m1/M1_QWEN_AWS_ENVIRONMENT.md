# M1 AWS Qwen Environment Freeze

Status: **FROZEN** after two synthetic-only A10G smoke calls.

This environment is the offline image-caption preprocessor for frozen Evidence Contract
`M1-FINMULTITIME-v1.0.1`. It is separate from the AlphaMAS formal Python environment and
does not change `uv.lock` or the formal Agent dependency contract.

## Worker and model identity

- AWS: `g5.xlarge`, `eu-west-2`, `eu-west-2a`, official Deep Learning OSS Nvidia Driver
  AMI GPU PyTorch 2.12 (Ubuntu 24.04) 20260725, `ami-0bd0039f023204bd2`.
- GPU: NVIDIA A10G, 23028 MiB; NVIDIA driver 595.71.05; driver CUDA 13.2.
- OS/kernel: Ubuntu 24.04.4 LTS, `6.17.0-1019-aws`.
- Model: `Qwen/Qwen3-VL-2B-Instruct` at exact revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- Model class: `Qwen3VLForConditionalGeneration`; processor: `Qwen3VLProcessor`.
- Snapshot bytes: 4,266,640,306. The model is unquantized and uses standard SDPA.

The transient EC2 public IP is deliberately excluded from the research identity.

## Runtime and generation

- Python 3.13.14; torch 2.12.1+cu130; PyTorch CUDA runtime 13.0.
- transformers 4.57.6; huggingface_hub 0.36.2; safetensors 0.8.0;
  Pillow 12.3.0; torchvision 0.27.1+cu130; accelerate 1.14.0;
  tokenizers 0.22.2. `qwen-vl-utils` is not used by the official processor path.
- Full 300-line package freeze SHA256:
  `e4b16b0179dab01f2d541b31986de6ff4a658c61b4d60c3222c30fd76778afc2`.
- CUDA, bfloat16, batch size 1, `do_sample=false`, `num_beams=1`,
  `max_new_tokens=256`.
- Python, torch, and CUDA seeds are all zero. cuDNN benchmark is disabled, cuDNN
  deterministic mode is enabled, and torch deterministic algorithms are enabled in
  warn-only mode. This is a reproducibility control, not a claim of universal bitwise
  determinism.

The model repository's sampling defaults are preserved in the snapshot, but the runner
explicitly overrides `do_sample=false`. Transformers therefore reports the snapshot's
`top_p`, `top_k`, and `temperature` as ignored during greedy generation.

## Contract identity

- Contract SHA256: `13563ba0c829addde44d602cf8b9ac0e2879d8091832ee120a5caefa4c843ab3`.
- Prompt SHA256: `284c6e52763796a47f7d30fd2e44cfe68d9211db6831d89ec5ed436920c34df9`.
- Ordered schema SHA256:
  `bf8f04330ffb1bd8468b9bf01eb96bec6b35bb4bad29c8e3f1ad6c47cf0ca8e4`.
- Canonical caption limit: 900 UTF-8 characters.

The runner reads these identities from the Evidence Contract, environment freeze, and
Qwen input manifest and fails closed on mismatch. The model input builder emits exactly
one image and the exact frozen prompt; it accepts no ticker, company, date, case, news,
table, time-series, or outcome context.

## Synthetic smoke evidence

The generated company-free 640×400 PNG has SHA256
`6ffb3fd7da6e769117acc480911a7de1e7c81cf02a75b4ed14a9909a47f02f4b`.

| Run | Inference seconds | Peak GPU bytes | Raw SHA256 | Canonical SHA256 | Length |
| --- | ---: | ---: | --- | --- | ---: |
| 1 | 12.486025 | 4,335,733,760 | `09807960d0f82b69d2f10da316d11f092f0b51adef7bc0dc0ca7f13bfaadf045` | `3549cf3cdce93a0088fc17b7dc88c10399f78114ddfa813e39ce0d75f91eff94` | 274 |
| 2 | 2.582725 | 4,335,733,760 | `09807960d0f82b69d2f10da316d11f092f0b51adef7bc0dc0ca7f13bfaadf045` | `3549cf3cdce93a0088fc17b7dc88c10399f78114ddfa813e39ce0d75f91eff94` | 274 |

Both raw outputs and canonical outputs are identical. Each canonical output contains
exactly the nine ordered fields, is within 900 characters, and contains no price target,
BUY/HOLD/SELL recommendation, forecast, future-return language, or external company
knowledge.

## Formal boundary

- Formal captions generated: **NO**
- AMZN official caption: **NOT_GENERATED**
- JPM official caption: **NOT_GENERATED**
- Final Evidence Packets generated: **NO**

The future formal-caption task must run the exact committed captioning code SHA, the
exact environment/model identities above, and the fail-closed FORMAL mode. This freeze
does not authorize formal inference.
