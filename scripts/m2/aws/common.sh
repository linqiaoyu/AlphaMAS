#!/usr/bin/env bash
set -euo pipefail

readonly M2_PROFILE="alphamas"
readonly M2_REGION="eu-west-2"
readonly M2_INSTANCE_TYPE="g5.xlarge"
readonly M2_TASK_TAG="M2-05"
readonly M2_MANIFEST="${M2_MANIFEST:-.m2-aws-resources.json}"

require_context() {
  : "${AWS_PROFILE:?Set AWS_PROFILE=alphamas}"
  : "${AWS_REGION:?Set AWS_REGION=eu-west-2}"
  [[ "$AWS_PROFILE" == "$M2_PROFILE" ]] || { echo "AWS_PROFILE must be alphamas" >&2; exit 2; }
  [[ "$AWS_REGION" == "$M2_REGION" ]] || { echo "AWS_REGION must be eu-west-2" >&2; exit 2; }
  export AWS_DEFAULT_REGION="$M2_REGION"
  aws sts get-caller-identity --query Arn --output text >/dev/null
}

single_instance_id() {
  aws ec2 describe-instances --filters \
    "Name=tag:Project,Values=AlphaMAS" "Name=tag:Stage,Values=M2" \
    "Name=tag:ProvisionedByTask,Values=$M2_TASK_TAG" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

require_manifest() {
  [[ -s "$M2_MANIFEST" ]] || { echo "Missing $M2_MANIFEST" >&2; exit 2; }
}
