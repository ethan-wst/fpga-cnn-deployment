#!/usr/bin/env bash
# run_vitis_docker.sh - Helper script to run the Vitis AI Docker container

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH. Please install Docker first."
    exit 1
fi

# Check if the Vitis AI image exists
if ! docker images | grep -q "xilinx/vitis-ai-pytorch-cpu:latest"; then
    echo "Vitis AI Docker image not found locally."
    echo "Pulling the Docker image (this might take a while)..."
    docker pull xilinx/vitis-ai-pytorch-cpu:latest
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to pull the Docker image. Please check your internet connection."
        exit 1
    fi
fi

echo "Starting Vitis AI Docker container..."
echo "Mounting project directory: $PROJECT_DIR"

# Run the Docker container
# - Mount the current directory to /workspace
# - Set working directory to /workspace
# - Remove the container when it exits
# - Allocate a pseudo-TTY
docker run -it --rm \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace \
    xilinx/vitis-ai-pytorch-cpu:latest \
    /bin/bash -c "source setup_env.sh && /bin/bash"

echo "Container exited."
