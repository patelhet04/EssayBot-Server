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
pm2 stop  essaybot-express && pm2 delete essaybot-express || true
pm2 stop  essaybot-flask && pm2 delete essaybot-flask || true

# Start applications with simple PM2 commands
echo "🚀 Starting applications..."

# Start Express.js with simple command
echo "Starting Express.js on port 8000..."
PORT=8000 pm2 start "npx ts-node src/index.ts" --name essaybot-express

# Start Flask with original working config style
echo "Starting Flask on port 6000..."
FLASK_PORT=6000 pm2 start src/python/app.py --name essaybot-flask --interpreter src/python/venv/bin/python

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