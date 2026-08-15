#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_context
instance_id=$(single_instance_id)
[[ "$instance_id" != *$'\t'* && -n "$instance_id" ]] || { echo "Expected exactly one M2 instance" >&2; exit 2; }
aws ec2 start-instances --instance-ids "$instance_id" >/dev/null
aws ec2 wait instance-running --instance-ids "$instance_id"
echo "$instance_id running; boot auto-stop timer will re-arm"
