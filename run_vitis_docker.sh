#!/usr/bin/env bash
#
# Vitis AI Docker Container Launcher
#
# Simple script to launch Vitis AI Docker container for FPGA CNN deployment

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default Docker image
VITIS_AI_IMAGE="xilinx/vitis-ai-pytorch-cpu:latest"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Launch Docker container
echo "Starting Vitis AI Docker container..."
docker run --rm -it \
    -v "$PROJECT_DIR:/workspace" \
    -w "/workspace" \
    --shm-size=16G \
    "$VITIS_AI_IMAGE" \
    "$@"
