# Project Deployment Guide

This document outlines the deployment process for the application on an EC2 instance with automated deployments via GitHub Actions.

## Prerequisites

1. An AWS EC2 instance (Ubuntu 20.04 or later recommended)
2. SSH access to the EC2 instance
3. GitHub repository for the project
4. Required ports open in the EC2 security group (22, 80, 443, 8501)

## Initial Server Setup

1. **Connect to your EC2 instance**:
   ```bash
   ssh -i /path/to/your-key.pem ubuntu@your-ec2-ip
   ```

2. **Run the setup script**:
   ```bash
   curl -s https://raw.githubusercontent.com/yourusername/PROJECT-5/main/deployment/scripts/setup_server.sh | bash -s -- https://github.com/yourusername/PROJECT-5.git main
   ```
   Replace the GitHub URL with your repository URL.

## GitHub Repository Setup

1. **Add these secrets to your GitHub repository**:
   - `EC2_HOST`: Your EC2 instance public IP or DNS
   - `SSH_PRIVATE_KEY`: The private key that matches the public key on your EC2 instance

## Automated Deployments

- Pushes to the `main` branch will automatically trigger a deployment
- The GitHub Actions workflow will:
  1. Connect to your EC2 instance via SSH
  2. Pull the latest changes
  3. Rebuild and restart the Docker containers

## Manual Deployment

If you need to manually deploy:

1. **SSH into your EC2 instance**
2. **Run the deploy script**:
   ```bash
   cd /opt/app/repo
   ./deployment/scripts/deploy.sh main
   ```

## Monitoring

- **View application logs**:
  ```bash
  docker compose logs -f
  ```

- **Check container status**:
  ```bash
  docker ps
  ```

## Troubleshooting

- If the application doesn't start, check the logs:
  ```bash
  docker compose logs
  ```

- To completely reset the application:
  ```bash
  cd /opt/app/repo
  docker compose down -v
  rm -rf /opt/app/repo/*
  # Then run the setup script again
  ```
