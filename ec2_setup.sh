#!/usr/bin/env bash
set -euo pipefail

# Usage: bash ec2_setup.sh <github_repo_url> [branch]
# Example: bash ec2_setup.sh https://github.com/USER/PROJECT-5 main

REPO_URL=${1:-}
BRANCH=${2:-main}

if [[ -z "$REPO_URL" ]];n+then
  echo "Usage: $0 <github_repo_url> [branch]"
  exit 1
fi

sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Install Docker
if ! command -v docker >/dev/null 2>&1; then
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker $USER || true
fi

# Fetch repo and start via compose
if [[ ! -d /opt/app ]]; then
  sudo mkdir -p /opt/app
  sudo chown $USER:$USER /opt/app
fi

cd /opt/app

if [[ -d repo ]]; then
  cd repo
  git fetch --all
  git checkout "$BRANCH"
  git pull
else
  git clone "$REPO_URL" repo
  cd repo
  git checkout "$BRANCH" || true
fi

docker compose pull || true
docker compose build --no-cache
docker compose up -d

echo "Deployment complete. App should be reachable on port 8501."

