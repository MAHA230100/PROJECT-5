#!/bin/bash
set -euo pipefail

# Configuration
APP_DIR="/opt/app/repo"
BRANCH=${1:-main}

cd "$APP_DIR"

# Update repository
echo "🔄 Updating repository..."
git fetch --all
git checkout "$BRANCH"
git reset --hard origin/"$BRANCH"

# Rebuild and restart containers
echo "🚀 Deploying application..."
sudo docker-compose down
sudo docker-compose pull
sudo docker-compose up -d --build
sudo docker system prune -f

echo ""
echo "✅ Deployment complete!"
echo "🔍 Check logs with: sudo docker-compose logs -f"
