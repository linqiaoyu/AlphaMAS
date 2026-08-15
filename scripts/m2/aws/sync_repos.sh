#!/usr/bin/env bash
set -euo pipefail

readonly root="${M2_ROOT:-/mnt/alphamas}"
readonly source_remote="${M2_SOURCE_REMOTE:-https://github.com/linqiaoyu/AlphaMAS.git}"
readonly experiments_remote="${M2_EXPERIMENTS_REMOTE:-https://github.com/linqiaoyu/AlphaMAS-Experiments.git}"

validate_sha() {
  local variable_name=$1
  local value=$2
  [[ "$value" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "$variable_name must be exactly 40 hexadecimal characters" >&2
    return 2
  }
}

sync_repository() {
  local repository=$1
  local remote=$2
  local requested_sha
  requested_sha=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')

  if [[ ! -d "$repository/.git" ]]; then
    git clone "$remote" "$repository"
  elif [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
    echo "Refusing to change dirty repository: $repository" >&2
    return 2
  fi

  git -C "$repository" fetch origin
  git -C "$repository" cat-file -e "${requested_sha}^{commit}"
  git -C "$repository" checkout --detach "$requested_sha"

  local actual_sha
  actual_sha=$(git -C "$repository" rev-parse HEAD)
  [[ "$actual_sha" == "$requested_sha" ]] || {
    echo "Repository HEAD does not match requested SHA: $repository" >&2
    return 2
  }
}

main() {
  : "${M2_SOURCE_SHA:?Set M2_SOURCE_SHA to the exact AlphaMAS commit}"
  : "${M2_EXPERIMENTS_SHA:?Set M2_EXPERIMENTS_SHA to the exact AlphaMAS-Experiments commit}"
  validate_sha M2_SOURCE_SHA "$M2_SOURCE_SHA"
  validate_sha M2_EXPERIMENTS_SHA "$M2_EXPERIMENTS_SHA"

  mkdir -p "$root/repos" "$root/venvs"
  sync_repository "$root/repos/AlphaMAS" "$source_remote" "$M2_SOURCE_SHA"
  sync_repository \
    "$root/repos/AlphaMAS-Experiments" "$experiments_remote" "$M2_EXPERIMENTS_SHA"

  local actual_source_sha actual_experiments_sha normalized_source_sha
  local normalized_experiments_sha
  actual_source_sha=$(git -C "$root/repos/AlphaMAS" rev-parse HEAD)
  actual_experiments_sha=$(git -C "$root/repos/AlphaMAS-Experiments" rev-parse HEAD)
  normalized_source_sha=$(printf '%s' "$M2_SOURCE_SHA" | tr '[:upper:]' '[:lower:]')
  normalized_experiments_sha=$(
    printf '%s' "$M2_EXPERIMENTS_SHA" | tr '[:upper:]' '[:lower:]'
  )
  [[ "$actual_source_sha" == "$normalized_source_sha" ]]
  [[ "$actual_experiments_sha" == "$normalized_experiments_sha" ]]

  printf 'AlphaMAS requested: %s\nAlphaMAS actual:    %s\n' \
    "$M2_SOURCE_SHA" "$actual_source_sha"
  printf 'AlphaMAS-Experiments requested: %s\nAlphaMAS-Experiments actual:    %s\n' \
    "$M2_EXPERIMENTS_SHA" "$actual_experiments_sha"
  echo "SOURCE LINEAGE VERIFIED"

  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  UV_PROJECT_ENVIRONMENT="$root/venvs/m2-project" uv sync \
    --project "$root/repos/AlphaMAS" --frozen --extra dev --python 3.12.10
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
