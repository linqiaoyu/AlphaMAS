#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_context
instance_id=$(single_instance_id)
[[ -n "$instance_id" && "$instance_id" != *$'\t'* ]] || { echo "Expected exactly one M2 instance" >&2; exit 2; }
aws ec2 describe-instances --instance-ids "$instance_id" \
  --query 'Reservations[0].Instances[0].{Type:InstanceType,State:State.Name,Shutdown:InstanceInitiatedShutdownBehavior,AZ:Placement.AvailabilityZone,Profile:IamInstanceProfile.Arn}'
