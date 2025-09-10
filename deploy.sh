#!/bin/bash

# Configuration
EC2_HOST="your-ec2-public-ip"
EC2_USER="ubuntu"
SSH_KEY="/path/to/your-key.pem"
PROJECT_DIR="/home/ubuntu/book-recommendation-app"

# Exit on error
set -e

echo "🚀 Starting deployment..."

# SSH into EC2 and run deployment commands
ssh -i "$SSH_KEY" $EC2_USER@$EC2_HOST << 'ENDSSH'
    set -e
    echo "📂 Navigating to project directory..."
    cd $PROJECT_DIR
    
    echo "🔄 Pulling latest changes from GitHub..."
    git pull origin main  # or your branch name
    
    echo "🐳 Rebuilding Docker containers..."
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    
    echo "✅ Deployment completed successfully!"
    echo "🌐 Your application is now live at: http://$EC2_HOST:8501"
ENDSSH

echo "✨ Deployment script completed!"
