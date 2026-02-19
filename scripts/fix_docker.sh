#!/bin/bash

# Docker Database Fix Script
# Fixes corrupted buildkit database issue

echo "🐳 Fixing Docker Database..."
echo ""

echo "Stopping Docker daemon..."
sudo systemctl stop docker

echo "Removing corrupted buildkit database..."
sudo rm -rf /var/lib/docker/buildx

echo "Starting Docker daemon..."
sudo systemctl start docker

echo "Waiting for Docker to start..."
sleep 5

echo "Checking Docker status..."
if sudo systemctl is-active --quiet docker; then
    echo "✅ Docker is now running!"
    echo ""
    echo "You can now run the demo:"
    echo "  sudo ./demo_vllm_platform.sh"
else
    echo "❌ Docker still not starting. Check logs with:"
    echo "  sudo journalctl -u docker -n 20"
fi
