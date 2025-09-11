#!/bin/bash
set -euo pipefail

# Configuration
REPO_URL=${1:-}
BRANCH=${2:-main}
APP_DIR="/opt/app"

# Validate input
if [[ -z "$REPO_URL" ]]; then
  echo "Usage: $0 <github_repo_url> [branch]"
  echo "Example: $0 https://github.com/yourusername/PROJECT-5.git main"
  exit 1
fi

echo "🚀 Starting server setup..."

# Install required packages
echo "🛠️  Installing required packages..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release git

# Install Docker if not installed
if ! command -v docker >/dev/null 2>&1; then
  echo "🐳 Installing Docker..."
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  
  # Install Docker Compose standalone
  sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  
  # Add user to docker group
  sudo usermod -aG docker $USER
  
  # Start and enable Docker
  sudo systemctl enable --now docker
  
  echo "✅ Docker and Docker Compose installed successfully"
fi

# Setup application directory
echo "📁 Setting up application directory..."
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Clone or update repository
echo "📥 Setting up repository..."
if [[ -d "$APP_DIR/repo" ]]; then
  cd "$APP_DIR/repo"
  git fetch --all
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
else
  git clone "$REPO_URL" "$APP_DIR/repo"
  cd "$APP_DIR/repo"
  git checkout "$BRANCH" || true
fi

# Build and start containers
echo "🚀 Starting application..."
cd "$APP_DIR/repo"
sudo docker-compose pull
sudo docker-compose build --no-cache
sudo docker-compose up -d

echo ""
echo "✅ Server setup complete!"
echo "📝 Application should be running on port 8501"
echo "🔍 Check logs with: sudo docker-compose logs -f"
