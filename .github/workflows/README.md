# GitHub Actions CI/CD Setup for EssayBot

This document explains how to set up and use the GitHub Actions workflows for deploying EssayBot to your on-prem server.

## 📁 Repository Structure

```
EssayBot/
├── EssayBot-Server/          # Backend repository
│   ├── .github/workflows/
│   │   ├── deploy.yml        # Backend deployment workflow
│   │   └── README.md         # This file
│   └── ...
└── EssayBot-UI/              # Frontend repository
    ├── .github/workflows/
    │   ├── deploy.yml        # Frontend deployment workflow
    │   └── README.md         # Frontend setup guide
    └── ...
```

## 🔧 Prerequisites

### Server Requirements
- **Ubuntu/Debian** server (recommended)
- **Node.js 18+** installed
- **Python 3.10+** installed
- **PM2** installed globally (`npm install -g pm2`)
- **Git** installed
- **SSH access** configured

### GitHub Repository Setup
- Both repositories should be on GitHub
- SSH keys configured for server access
- GitHub Secrets configured (see below)

## 🔐 GitHub Secrets Configuration

You need to configure the following secrets in both repositories:

### Required Secrets
1. **SSH_HOST**: Your server's IP address or hostname
2. **SSH_USER**: SSH username for your server
3. **SSH_KEY**: Private SSH key for authentication
4. **SSH_PORT**: SSH port (optional, defaults to 22)
5. **PROJECT_PATH**: Path to your EssayBot directory on server (optional, defaults to `/opt/EssayBot`)

### How to Add Secrets
1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with the appropriate name and value

## 🚀 Deployment Workflows

### Backend Deployment (`EssayBot-Server`)
**Triggers:**
- Push to `main` branch
- Manual trigger via GitHub Actions UI

**What it does:**
1. Connects to your server via SSH
2. Pulls latest code from `main` branch
3. Installs Node.js dependencies
4. Sets up Python virtual environment
5. Installs Python dependencies
6. Restarts Express.js and Flask services via PM2
7. Performs health checks

### Frontend Deployment (`EssayBot-UI`)
**Triggers:**
- Push to `main` branch
- Manual trigger via GitHub Actions UI

**What it does:**
1. Connects to your server via SSH
2. Pulls latest code from `main` branch
3. Installs Node.js dependencies
4. Builds Next.js application
5. Restarts frontend service via PM2
6. Performs health checks

## 📋 Server Setup Checklist

Before running the workflows, ensure your server is properly configured:

### 1. Install Required Software
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python 3.10+
sudo apt install python3 python3-pip python3-venv -y

# Install PM2 globally
sudo npm install -g pm2

# Install Git
sudo apt install git -y
```

### 2. Create Project Directory
```bash
# Create project directory
sudo mkdir -p /opt/EssayBot
sudo chown $USER:$USER /opt/EssayBot

# Clone repositories
cd /opt/EssayBot
git clone <your-backend-repo-url> EssayBot-Server
git clone <your-frontend-repo-url> EssayBot-UI
```

### 3. Configure SSH Keys
```bash
# Generate SSH key pair (if not already done)
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# Add public key to server's authorized_keys
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Add private key to GitHub Secrets (SSH_KEY)
cat ~/.ssh/id_rsa
```

## 🧪 Testing the Workflows

### 1. Manual Testing
Before using GitHub Actions, test the deployment manually:

```bash
# On your server
cd /opt/EssayBot

# Test backend deployment
cd EssayBot-Server
git pull origin main
npm ci
cd src/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ../..
pm2 start "npx ts-node src/index.ts" --name essaybot-express --cwd EssayBot-Server

# Test frontend deployment
cd ../EssayBot-UI
git pull origin main
npm ci
npm run build
pm2 start "npm run start" --name essaybot-frontend --cwd EssayBot-UI
```

### 2. Test GitHub Actions
1. Make a small change to your code
2. Push to `main` branch
3. Go to **Actions** tab in GitHub
4. Monitor the workflow execution
5. Check if services are running on your server

## 🔍 Troubleshooting

### Common Issues

#### 1. SSH Connection Failed
- Verify SSH_HOST and SSH_USER are correct
- Ensure SSH_KEY is properly formatted (no extra spaces)
- Test SSH connection manually: `ssh username@host`

#### 2. Permission Denied
- Ensure the user has write permissions to the project directory
- Check if PM2 is installed globally
- Verify Node.js and Python versions

#### 3. Service Failed to Start
- Check PM2 logs: `pm2 logs`
- Verify all dependencies are installed
- Check if ports are already in use

#### 4. Build Failures
- Check if all required environment variables are set
- Verify Node.js version compatibility
- Check for missing dependencies

### Debugging Commands
```bash
# Check PM2 status
pm2 status
pm2 logs

# Check service ports
netstat -tlnp | grep :3000  # Frontend
netstat -tlnp | grep :8000  # Backend
netstat -tlnp | grep :6000  # Flask

# Check application health
curl http://localhost:3000  # Frontend
curl http://localhost:8000/health  # Backend
curl http://localhost:6000/health  # Flask
```

## 🔄 Rollback Procedure

If deployment fails, you can rollback to a previous version:

```bash
# On your server
cd /opt/EssayBot

# Check recent commits
git log --oneline -5

# Rollback to previous commit
git checkout <previous-commit-hash>

# Restart services
pm2 restart essaybot-express
pm2 restart essaybot-frontend
```

## 📊 Monitoring

### PM2 Monitoring
```bash
# View all processes
pm2 list

# Monitor in real-time
pm2 monit

# View logs
pm2 logs --lines 100
```

### Application Monitoring
- Frontend: http://your-server:3000
- Backend API: http://your-server:8000
- Flask AI Service: http://your-server:6000

## 🎯 Best Practices

1. **Always test on a staging environment first**
2. **Keep your SSH keys secure and rotate them regularly**
3. **Monitor deployment logs for any issues**
4. **Set up proper backup procedures**
5. **Use environment-specific configurations**
6. **Implement proper error handling and logging**

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review GitHub Actions logs for detailed error messages
3. Verify all prerequisites are met
4. Test deployment manually before using GitHub Actions

---

**Happy Deploying! 🚀** 