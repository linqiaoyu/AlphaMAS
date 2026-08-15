#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_context

aws ec2 describe-instances --filters \
  "Name=tag:Project,Values=AlphaMAS" "Name=tag:Stage,Values=M2" \
  "Name=tag:ProvisionedByTask,Values=$M2_TASK_TAG" \
  --query 'Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,AZ:Placement.AvailabilityZone}'
aws ec2 describe-volumes --filters \
  "Name=tag:Project,Values=AlphaMAS" "Name=tag:Stage,Values=M2" \
  "Name=tag:ProvisionedByTask,Values=$M2_TASK_TAG" \
  --query 'Volumes[].{Id:VolumeId,Type:VolumeType,Size:Size,State:State,Encrypted:Encrypted}'
aws ec2 describe-security-groups --filters \
  "Name=tag:Project,Values=AlphaMAS" "Name=tag:Stage,Values=M2" \
  "Name=tag:ProvisionedByTask,Values=$M2_TASK_TAG" \
  --query 'SecurityGroups[].{Id:GroupId,Name:GroupName,Vpc:VpcId}'
aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters Name=instance-type,Values="$M2_INSTANCE_TYPE" \
  --query 'sort_by(InstanceTypeOfferings,&Location)[].Location'
