# EC2 Deployment Guide for Book Recommendation App

This guide provides step-by-step instructions for deploying the Book Recommendation application on an AWS EC2 instance using Docker and Docker Compose.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [EC2 Instance Setup](#ec2-instance-setup)
3. [Application Deployment](#application-deployment)
4. [Automated Deployment](#automated-deployment)
5. [Maintenance](#maintenance)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

- AWS account with EC2 access
- Running EC2 instance (Ubuntu 20.04/22.04 LTS recommended)
- SSH access to the EC2 instance
- Docker and Docker Compose installed on EC2
- GitHub repository for the project

## EC2 Instance Setup

1. **Connect to your EC2 instance**:
   ```bash
   ssh -i /path/to/your-key.pem ubuntu@your-ec2-public-ip
   ```

2. **Update and install required packages**:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   sudo apt install -y docker.io docker-compose git
   ```

3. **Add your user to the docker group**:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

4. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git /home/ubuntu/book-recommendation-app
   cd /home/ubuntu/book-recommendation-app
   ```

## Application Deployment

### Manual Deployment

1. **Create a deployment script** (`deploy.sh`):
   ```bash
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
       git pull origin main
       
       echo "🐳 Rebuilding Docker containers..."
       docker-compose down
       docker-compose build --no-cache
       docker-compose up -d
       
       echo "✅ Deployment completed successfully!"
       echo "🌐 Your application is now live at: http://$EC2_HOST:8501"
   ENDSSH
   
   echo "✨ Deployment script completed!"
   ```

2. **Make the script executable**:
   ```bash
   chmod +x deploy.sh
   ```

3. **Run the deployment script**:
   ```bash
   ./deploy.sh
   ```

### First-Time Setup

1. **Build and start the containers**:
   ```bash
   cd /home/ubuntu/book-recommendation-app
   docker-compose up --build -d
   ```

2. **Verify the application**:
   - Open your browser and navigate to: `http://your-ec2-public-ip:8501`

## Automated Deployment with GitHub Actions

1. **Create GitHub workflow file** (`.github/workflows/deploy.yml`):
   ```yaml
   name: Deploy to EC2

   on:
     push:
       branches: [ main ]

   jobs:
     deploy:
       runs-on: ubuntu-latest
       
       steps:
       - name: Checkout code
         uses: actions/checkout@v2
       
       - name: Deploy to EC2
         uses: appleboy/ssh-action@master
         with:
           host: ${{ secrets.EC2_HOST }}
           username: ${{ secrets.EC2_USER }}
           key: ${{ secrets.SSH_PRIVATE_KEY }}
           script: |
             cd /home/ubuntu/book-recommendation-app
             git pull origin main
             docker-compose down
             docker-compose build --no-cache
             docker-compose up -d
   ```

2. **Set up GitHub Secrets**:
   - `EC2_HOST`: Your EC2 public IP
   - `EC2_USER`: Usually "ubuntu"
   - `SSH_PRIVATE_KEY`: Your private key for EC2 access

## Security Configuration

1. **Update EC2 Security Group**:
   - Add inbound rule for port 8501 (or your application port)
   - Restrict source IP if possible for better security

2. **Environment Variables**:
   - Store sensitive information in `.env` file (add to `.gitignore`)
   - Or use AWS Parameter Store/Secrets Manager

## Maintenance

### View Logs
```bash
docker-compose logs -f
```

### Restart Application
```bash
cd /home/ubuntu/book-recommendation-app
docker-compose restart
```

### Update Application
1. Make your changes locally
2. Commit and push to main branch
3. The GitHub Action will automatically deploy changes
   - Or run `./deploy.sh` manually

## Troubleshooting

### Common Issues

1. **Permission Denied when running docker commands**
   - Make sure your user is in the docker group
   - Run: `sudo usermod -aG docker $USER` and log out/in

2. **Port already in use**
   - Check running containers: `docker ps`
   - Stop conflicting container: `docker stop <container_id>`

3. **Application not accessible**
   - Check security group settings
   - Verify the application is running: `docker ps`
   - Check logs: `docker-compose logs`

## Backup and Recovery

1. **Backup database** (if applicable):
   ```bash
   docker exec -t your_db_container pg_dumpall -c -U your_user > db_backup_$(date +%Y-%m-%d).sql
   ```

2. **Restore from backup**:
   ```bash
   cat your_backup_file.sql | docker exec -i your_db_container psql -U your_user
   ```

## Scaling (Optional)

For production environments, consider:
1. Using an Application Load Balancer
2. Setting up auto-scaling groups
3. Configuring CloudWatch for monitoring

---

For additional support, please refer to the project's README or contact the development team.
