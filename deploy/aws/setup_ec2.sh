#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  EC2 t2.micro Setup Script (AWS Free Tier)
# ═══════════════════════════════════════════════════════════
#
#  Prerequisites:
#    1. Launch EC2 t2.micro (Amazon Linux 2023 or Ubuntu 22.04)
#    2. Security group: open ports 22 (SSH), 8000 (API), 8501 (Dashboard)
#    3. SSH into the instance
#    4. Run this script: bash setup_ec2.sh
#
#  Free Tier:
#    - t2.micro: 750 hours/month (12-month free tier)
#    - 30 GB EBS storage
#    - This is enough to run the entire system
# ═══════════════════════════════════════════════════════════

set -e

echo "🚀 Setting up Agentic Fraud Detection on EC2..."

# ── Install Docker ──
echo "Installing Docker..."
sudo yum update -y 2>/dev/null || sudo apt-get update -y
sudo yum install -y docker git 2>/dev/null || sudo apt-get install -y docker.io docker-compose git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# ── Install Docker Compose ──
echo "Installing Docker Compose..."
DOCKER_COMPOSE_VERSION="v2.24.0"
sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# ── Clone project ──
echo "Cloning project..."
cd ~
git clone <YOUR_REPO_URL> agentic-fraud-detection || echo "Clone manually or scp the project"
cd agentic-fraud-detection

# ── Create .env file ──
echo "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY"
    echo "   nano .env"
    echo ""
fi

# ── Build and run ──
echo "Building Docker image..."
cd deploy
sudo docker-compose up -d --build

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ Deployment Complete!"
echo ""
echo "  API:       http://$(curl -s ifconfig.me):8000"
echo "  API Docs:  http://$(curl -s ifconfig.me):8000/docs"
echo "  Dashboard: http://$(curl -s ifconfig.me):8501"
echo ""
echo "  Useful commands:"
echo "    docker-compose logs -f    # View logs"
echo "    docker-compose restart    # Restart services"
echo "    docker-compose down       # Stop everything"
echo "═══════════════════════════════════════════════"
