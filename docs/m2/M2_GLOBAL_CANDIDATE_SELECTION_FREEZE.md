# M2-12 Global Candidate Selection Freeze

M2-12 mechanically selected C09 from the nine frozen M2-11 canonical global
PA-CTPPO-v2 candidates using only the preregistered 16-decision sequential
VALIDATION protocol.

- Selected parameter SHA: `6baafc03b0b63512b3a66a1ae8f1ce1ce7e774395787626b49905e7b72cd1841`
- Actor SHA: `5af9a28baaf2dc25687e65a7bf8bbefb047fa1a9664b05257bfbf78d2ac5b14d`
- Critic SHA: `e55f942699d47eeab16d92b8ad3ff7277306239a8d1ba7d542f14080c86d5416`
- Learning rate / iteration: `1e-3` / `100`
- Mean sequential local R3: `0.0011119952063548388`
- Worst-symbol cumulative R3: `-0.05822419930530598`
- Prompt override rate: `0.125`
- Selection archive commit: `49eb1d227c45d0efd045ad1a388b608075266494`

The primary metric had no tie; no tie-break was invoked. Human override was not
permitted and did not occur. The audit-only replay reproduced all 144 actions,
chosen-action rewards, candidate primary scores, the tie path, and C09 exactly.

M2-12 used VALIDATION only for the preregistered global checkpoint selection. No
candidate parameters were updated using VALIDATION. This checkpoint is the
permanently frozen global starting point for subsequent M2 development.
