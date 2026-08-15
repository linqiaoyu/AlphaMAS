#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_context

: "${M2_AMI_ID:?Set M2_AMI_ID to the audited AWS-owned DLAMI}"
: "${M2_SUBNET_ID:?Set M2_SUBNET_ID to the audited public subnet}"
: "${M2_SECURITY_GROUP_ID:?Set M2_SECURITY_GROUP_ID}"
: "${M2_IAM_PROFILE:?Set M2_IAM_PROFILE to the dedicated SSM profile}"

existing=$(single_instance_id)
if [[ -n "$existing" ]]; then
  echo "Reusing tagged M2 instance: $existing"
  exit 0
fi

tags='[{"Key":"Name","Value":"alphamas-m2-gpu"},{"Key":"Project","Value":"AlphaMAS"},{"Key":"Stage","Value":"M2"},{"Key":"Purpose","Value":"AgenticRL"},{"Key":"ResearchScope","Value":"PreFormal"},{"Key":"ManagedBy","Value":"Codex"},{"Key":"ProvisionedByTask","Value":"M2-05"},{"Key":"CostGuard","Value":"Enabled"}]'
tag_specifications="[{\"ResourceType\":\"instance\",\"Tags\":$tags},{\"ResourceType\":\"volume\",\"Tags\":$tags}]"
aws ec2 run-instances --image-id "$M2_AMI_ID" --instance-type "$M2_INSTANCE_TYPE" \
  --count 1 --subnet-id "$M2_SUBNET_ID" --security-group-ids "$M2_SECURITY_GROUP_ID" \
  --iam-instance-profile "Name=$M2_IAM_PROFILE" --instance-initiated-shutdown-behavior stop \
  --user-data "file://$(dirname "$0")/user_data.sh" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":75,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}},{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":64,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":false}}]' \
  --tag-specifications "$tag_specifications" \
  --query 'Instances[0].InstanceId' --output text
