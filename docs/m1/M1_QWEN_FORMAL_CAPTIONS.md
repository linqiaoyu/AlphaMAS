# M1 Qwen Formal Caption Freeze

## Verdict

The first complete correctness-valid Qwen output for each eligible M1 image is frozen as the official caption. Processing order was AMZN followed by JPM. Neither image was rerun, no candidate outputs were compared, and no caption text was manually edited.

## Frozen identities

- Captioning code: `4ce6af13c6dd82218d1bb9a2600fdbf7f108bc8c`
- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Model revision: `89644892e4d85e24eaac8bacfd4f463576704203`
- AWS environment freeze SHA256: `cd011c2953457c399573319dd8067bede1d3c635061846fc420266010d19d7e2`
- Package freeze SHA256: `e4b16b0179dab01f2d541b31986de6ff4a658c61b4d60c3222c30fd76778afc2`
- Evidence Contract: `M1-FINMULTITIME-v1.0.1`
- Evidence Contract SHA256: `13563ba0c829addde44d602cf8b9ac0e2879d8091832ee120a5caefa4c843ab3`
- Prompt SHA256: `284c6e52763796a47f7d30fd2e44cfe68d9211db6831d89ec5ed436920c34df9`
- Ordered schema SHA256: `bf8f04330ffb1bd8468b9bf01eb96bec6b35bb4bad29c8e3f1ad6c47cf0ca8e4`

## Official captions

| Order | Image | Image SHA256 | Duration | Peak GPU bytes | Raw output SHA256 | Canonical caption SHA256 | Length | Validation |
|---:|---|---|---:|---:|---|---|---:|---|
| 1 | `amzn_2023_H2_candlestick.png` | `215c01f8a03dc55719558644992b14c28d8b9f604d44e3504ef7df0f99997bb8` | 3.502541 s | 4,387,272,192 | `4e04f57d03a7e5aa40d14607c239bd7568cf9459d1d49901efc603db22f6dbad` | `131c2904cd82c850f94b00cc58e2b97b2ef69c7e805fa562ef2c8f2e00a568fe` | 269 | PASS |
| 2 | `jpm_2023_H2_candlestick.png` | `b9e218ab002a8bfd6c0cbf86dd53abe9bcfeda4744a785754cb1b9173cc85611` | 2.722321 s | 4,387,272,192 | `cae5f4770733aad115a913a2008f3c50e18daaa947e4ba5736330075e474b878` | `44e980fa7768c0cf36961d6c42ea769157e4b989d02b60b37618256028b04d70` | 276 | PASS |

Both canonical captions contain exactly the nine ordered frozen fields, are within 900 characters, and pass the prohibited recommendation, target, forecast, prediction, and future-return checks.

## Integration result

- Unique formal images: 2
- Unique official captions: 2
- `GENERATED`: 52
- `NOT_APPLICABLE`: 26
- `PENDING`: 0
- Non-caption research differences across 78 cases: 0
- Final Evidence Packets: `NOT_GENERATED`
- Input bundle frozen: `false`

The complete processed caption files remain Git-ignored. Source control contains only lightweight hashes, provenance, and equivalence evidence.
