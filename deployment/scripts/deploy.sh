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
docker compose down
docker compose pull
docker compose up -d --build
docker system prune -f

echo ""
echo "✅ Deployment complete!"
echo "🔍 Check logs with: docker compose logs -f"
