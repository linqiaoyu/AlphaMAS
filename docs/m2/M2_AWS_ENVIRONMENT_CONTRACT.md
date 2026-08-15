# M2 AWS Environment Contract

## Status and purpose

**PRE-FORMAL DEVELOPMENT — NOT FORMAL M2.** This contract freezes the verified
AWS execution environment for later M2 preprocessing and RL work. It does not
freeze an encoder, model architecture, PPO configuration, or training framework.

## Verified lineage and evidence

- Verified runtime AlphaMAS source: `1594ef850999ecb29f150cb4a8eedfe20c6ad7a6`.
- Verified AlphaMAS-Experiments source: `6c9e18d7d0ea1a2b91fd4ac5eefe829160a15cac`.
- AWS evidence archive commit: `3a8e63a9abe89a80787b68aabb9b3523453966fc`.
- Evidence path: `experiments/M2/development/aws_environment_v1/`.

AWS tasks must provide `M2_SOURCE_SHA` and `M2_EXPERIMENTS_SHA` explicitly.
The sync helper validates 40-hex commit identities, refuses dirty repositories,
checks out detached commits, and verifies both resulting HEADs before dependency
sync.

## AWS architecture

- Profile/region: temporary `aws login` profile `alphamas`, `eu-west-2`.
- Compute: one On-Demand `g5.xlarge` in `eu-west-2a`.
- AMI: AWS-owned Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
  20260814, `ami-0ad015dd04cfae0b4`.
- OS/architecture: Ubuntu 22.04.5 LTS, Linux 6.8.0-1061-aws, x86_64.
- GPU: one NVIDIA A10G, 23,028 MiB, driver 595.91.07.
- Access: Systems Manager with a dedicated least-privilege instance profile;
  the security group has zero inbound rules.

## Storage

The encrypted 64 GiB `gp3` data volume is mounted as ext4 at `/mnt/alphamas`
using its filesystem UUID and has `DeleteOnTermination=false`. The encrypted
root volume is 75 GiB `gp3`. Local NVMe instance storage is mounted by the DLAMI
at `/opt/dlami/nvme` and is disposable cache only.

The operational S3 bucket is in `eu-west-2`, uses SSE-S3, has versioning enabled,
and has every Block Public Access control enabled. It is not a research dataset
store or sole archive.

## Runtime and verification

The project environment uses Python 3.12.10 and uv 0.12.5 with `uv sync
--frozen`. AWS Linux results were: reward simulator 44 passed, reward study 11,
M1 runtime 23, M1 Formal 21, PIT 19, M2 86, and backtesting 205. The portable
pilot archive target executed with one pass and no skip.

Hardware smoke used an isolated environment with Torch 2.13.0+cu130 and CUDA
13.0. One-A10G discovery, CUDA availability, matrix multiplication,
synchronization, and finite-result checks passed. This Torch version is not the
frozen final M2 training framework.

## Cost guard and lifecycle

The systemd `alphamas-autostop.timer` is enabled at boot and shuts down after at
most 90 minutes; instance-initiated shutdown behavior is `stop`. Current audited
compute price was USD 1.277/hour. Conservative cumulative M2-05 compute/task
cost is approximately USD 0.444252, below the USD 5 task ceiling. Persistent EBS
idle cost is approximately USD 0.43/day.

Every future AWS task must authenticate, inventory, recheck price/budget, export
exact source SHAs, start the existing instance, verify the timer, sync and verify
detached lineage, persist/archive outputs, stop, wait for `stopped`, and confirm
zero running tagged GPUs.

## Boundaries and verdict

No DeepSeek call, Qwen research inference, RL training, Formal evaluation, or
FinMultiTime upload occurred. The instance was confirmed stopped after all
Linux, CUDA, S3, and EBS checks passed. The environment is ready for M2-06 under
the explicit-lineage and stop discipline above.
