#!/bin/bash

# Simple deployment script for ESSAYBOT-SERVER
set -e

echo "🚀 Deploying ESSAYBOT-SERVER..."

# Update code (assuming you're already in your project directory)
git pull origin main

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Setup environment file
echo "⚙️ Setting up environment configuration..."
if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Please create it with your configuration."
    exit 1
fi

# Setup Python Flask
echo "🐍 Setting up Python Flask..."
cd src/python

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate and install Python dependencies
source venv/bin/activate
pip3 install -r requirements.txt
# Add other dependencies as needed
deactivate

cd ../..

# Setup Nginx configuration (copy to sites-available if needed)
echo "🌐 Setting up Nginx configuration..."
if [ -f "nginx.conf" ]; then
    sudo cp nginx.conf /etc/nginx/sites-available/essaybot
    sudo ln -sf /etc/nginx/sites-available/essaybot /etc/nginx/sites-enabled/essaybot
    sudo nginx -t
fi

# Stop existing processes
echo "🔄 Restarting services..."
pm2 stop all || true
pm2 delete all || true

# Start all applications using ecosystem configuration
echo "🚀 Starting applications with ecosystem config..."
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Restart Nginx
echo "🔄 Restarting Nginx..."
sudo systemctl reload nginx

echo "✅ Backend deployment complete!"
echo "🟢 Express.js running on port 8000"
echo "🐍 Flask running on port 6000"
echo "🌐 Nginx configured and running"
pm2 status