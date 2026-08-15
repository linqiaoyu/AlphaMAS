#!/usr/bin/env bash
set -euo pipefail

cat >/etc/systemd/system/alphamas-autostop.service <<'UNIT'
[Unit]
Description=Stop AlphaMAS GPU instance at the task runtime ceiling
[Service]
Type=oneshot
ExecStart=/usr/sbin/shutdown -h now
UNIT

cat >/etc/systemd/system/alphamas-autostop.timer <<'UNIT'
[Unit]
Description=AlphaMAS 90-minute EC2 auto-stop guard
[Timer]
OnBootSec=90min
AccuracySec=1min
Unit=alphamas-autostop.service
[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now alphamas-autostop.timer

data_device=""
for candidate in $(lsblk -dpno NAME,SIZE,TYPE | awk '$2=="64G" && $3=="disk" {print $1}'); do
  if ! blkid "$candidate" >/dev/null 2>&1; then data_device="$candidate"; break; fi
done
if [[ -n "$data_device" ]]; then
  mkfs.ext4 -F "$data_device"
  uuid=$(blkid -s UUID -o value "$data_device")
  mkdir -p /mnt/alphamas
  printf 'UUID=%s /mnt/alphamas ext4 defaults,nofail 0 2\n' "$uuid" >>/etc/fstab
  mount /mnt/alphamas
fi

mkdir -p /mnt/alphamas/{repos,venvs,data,checkpoints,logs,manifests,scratch-persistent}
chown -R ubuntu:ubuntu /mnt/alphamas

instance_store=$(lsblk -dpno NAME,MODEL,TYPE | awk '$2 ~ /Instance_Storage/ && $3=="disk" {print $1; exit}')
if [[ -n "${instance_store:-}" ]] && ! blkid "$instance_store" >/dev/null 2>&1; then
  mkfs.ext4 -F "$instance_store"
  mkdir -p /mnt/alphamas-cache
  mount "$instance_store" /mnt/alphamas-cache
  chown ubuntu:ubuntu /mnt/alphamas-cache
fi

{
  date -u +%FT%TZ
  uname -a
  nvidia-smi
} >/mnt/alphamas/manifests/bootstrap.txt 2>&1 || true
