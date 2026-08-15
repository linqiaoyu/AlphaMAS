#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_context
"$(dirname "$0")/inventory.sh"
if [[ "${1:-}" != "--confirm-destroy" ]]; then
  echo "Dry run only. Re-run with --confirm-destroy after dissertation completion." >&2
  exit 2
fi
echo "Destructive teardown is intentionally manual; delete only IDs reviewed above." >&2
exit 2
