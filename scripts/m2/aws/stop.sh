#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_context
instance_id=$(single_instance_id)
if [[ -z "$instance_id" ]]; then echo "No tagged M2 instance to stop"; exit 0; fi
[[ "$instance_id" != *$'\t'* ]] || { echo "Refusing to stop multiple instances" >&2; exit 2; }
aws ec2 stop-instances --instance-ids "$instance_id" >/dev/null
aws ec2 wait instance-stopped --instance-ids "$instance_id"
echo "$instance_id stopped"
