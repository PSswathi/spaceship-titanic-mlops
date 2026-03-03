#!/bin/bash
# ─────────────────────────────────────────────
# user_data.sh — EC2 Bootstrap Script
# Runs automatically on first boot
# ─────────────────────────────────────────────

set -e
exec > /var/log/user_data.log 2>&1

echo "========================================="
echo " Spaceship Titanic MLOps — EC2 Bootstrap"
echo "========================================="

# ── 1. System update ──────────────────────────
apt-get update -y
apt-get upgrade -y

# ── 2. Install Docker ─────────────────────────
apt-get install -y docker.io
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# ── 3. Install Git + Python ───────────────────
apt-get install -y git python3 python3-pip python3-venv

# ── 4. Clone repo ─────────────────────────────
cd /home/ubuntu
git clone ${github_repo} spaceship-titanic-mlops
cd spaceship-titanic-mlops
git checkout ${repo_branch}
chown -R ubuntu:ubuntu /home/ubuntu/spaceship-titanic-mlops

# ── 5. Build Docker image ─────────────────────
docker build -t ${app_name}:latest .

# ── 6. Run Docker container ───────────────────
docker run -d \
  -p 8000:8000 \
  --name ${app_name} \
  --restart unless-stopped \
  ${app_name}:latest

echo "========================================="
echo " Bootstrap complete!"
echo " FastAPI : http://$(curl -s ifconfig.me):8000"
echo " Docs    : http://$(curl -s ifconfig.me):8000/docs"
echo " Health  : http://$(curl -s ifconfig.me):8000/health"
echo "========================================="
