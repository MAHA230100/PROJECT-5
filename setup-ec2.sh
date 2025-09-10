#!/bin/bash

# Configuration
EC2_HOST="your-ec2-public-ip"
EC2_USER="ubuntu"
SSH_KEY="/path/to/your-key.pem"
GIT_REPO="your-github-repo-url"
PROJECT_DIR="/home/ubuntu/book-recommendation-app"

# Exit on error
set -e

echo "🚀 Starting EC2 setup..."

# Install required packages
echo "🛠️  Installing required packages..."
ssh -i "$SSH_KEY" $EC2_USER@$EC2_HOST "
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y docker.io docker-compose git
    sudo usermod -aG docker \$USER
    newgrp docker
"

# Clone the repository
echo "📥 Cloning repository..."
ssh -i "$SSH_KEY" $EC2_USER@$EC2_HOST "
    if [ -d '$PROJECT_DIR' ]; then
        echo '⚠️  Project directory already exists. Pulling latest changes...'
        cd $PROJECT_DIR
        git pull
    else
        git clone $GIT_REPO $PROJECT_DIR
        cd $PROJECT_DIR
    fi
"

echo "✅ EC2 setup completed!"
echo "📝 Don't forget to update your environment variables if needed."
