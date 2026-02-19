#!/bin/bash

# MVP Setup Script - Complete Docker & NGC Configuration
# Run this script manually in your terminal

echo "🚀 DGX Spark vLLM MVP Setup"
echo "============================"
echo ""

echo "Step 1: Adding user to docker group..."
sudo usermod -aG docker $USER

echo ""
echo "Step 2: Fixing Docker database..."
sudo systemctl stop docker
sudo rm -rf /var/lib/docker/buildx
sudo systemctl start docker

echo ""
echo "Step 3: Waiting for Docker to start..."
sleep 5

echo "Step 4: Verifying Docker..."
if sudo systemctl is-active --quiet docker; then
    echo "✅ Docker is running!"
else
    echo "❌ Docker failed to start. Check with: sudo journalctl -u docker -n 20"
    exit 1
fi

echo ""
echo "Step 5: Pulling NVIDIA NGC vLLM container..."
docker pull nvcr.io/nvidia/vllm:25.09-py3

echo ""
echo "Step 6: Testing container pull..."
if docker images | grep -q "nvcr.io/nvidia/vllm"; then
    echo "✅ NGC container ready!"
else
    echo "❌ Container pull failed"
    exit 1
fi

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "Next steps:"
echo "1. LOGOUT and LOG BACK IN (or open new terminal) for docker group changes"
echo "2. Run the demo: ./demo_vllm_platform.sh"
echo ""
echo "After logout/login, you can run Docker commands without sudo!"
