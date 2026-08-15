#!/usr/bin/env bash
set -euo pipefail
readonly root=/mnt/alphamas
readonly source_sha=77058329fdc3e7cc089fcc73680a206f5cd36876
readonly experiments_sha=6c9e18d7d0ea1a2b91fd4ac5eefe829160a15cac

mkdir -p "$root/repos" "$root/venvs"
if [[ ! -d "$root/repos/AlphaMAS/.git" ]]; then
  git clone --branch baseline-m2 https://github.com/linqiaoyu/AlphaMAS.git "$root/repos/AlphaMAS"
fi
git -C "$root/repos/AlphaMAS" fetch origin
git -C "$root/repos/AlphaMAS" checkout --detach "$source_sha"
if [[ ! -d "$root/repos/AlphaMAS-Experiments/.git" ]]; then
  git clone --branch main https://github.com/linqiaoyu/AlphaMAS-Experiments.git "$root/repos/AlphaMAS-Experiments"
fi
git -C "$root/repos/AlphaMAS-Experiments" fetch origin
git -C "$root/repos/AlphaMAS-Experiments" checkout --detach "$experiments_sha"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
UV_PROJECT_ENVIRONMENT="$root/venvs/m2-project" uv sync --project "$root/repos/AlphaMAS" \
  --frozen --extra dev --python 3.12.10
