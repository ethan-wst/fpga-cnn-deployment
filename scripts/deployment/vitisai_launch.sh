#!/bin/bash
#
# Vitis AI Docker Container Launcher
#
# This script launches the Vitis AI Docker container with appropriate volume mounts
# and runs the provided command inside the container. It's designed to simplify
# the execution of Vitis AI tools in a containerized environment.

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# Check if the docker command exists
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in the path."
    exit 1
fi

# Check if the Vitis AI image is available
VITIS_AI_IMAGE="xilinx/vitis-ai-pytorch-gpu:3.5.0.001-1eed93cde"
docker image inspect "$VITIS_AI_IMAGE" &>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Vitis AI Docker image '$VITIS_AI_IMAGE' is not available."
    echo "Please pull the image first with: docker pull $VITIS_AI_IMAGE"
    echo "Or use a different version by editing this script."
    exit 1
fi

# Display usage information if no arguments are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Launch the Vitis AI Docker container and execute the specified command."
    echo "If no command is provided, an interactive shell is started."
    echo
    echo "Example:"
    echo "  $0 python /workspace/scripts/quantization/vitis_ai_quantize.py --model resnet50"
    exit 0
fi

# Set up volume mounts for the Docker container
echo "Launching Vitis AI Docker container with image: $VITIS_AI_IMAGE"
echo "Project mounted at: /workspace"
echo "Command to execute: $@"
echo "------------------------------------------------------------"

# Run with a timeout to prevent hanging indefinitely
timeout_duration=1800 # 30 minutes timeout
timeout $timeout_duration docker run --rm -it \
    -v "$PROJECT_ROOT:/workspace" \
    -w "/workspace" \
    --user "$(id -u):$(id -g)" \
    --shm-size=8G \
    "$VITIS_AI_IMAGE" \
    "$@"

EXIT_CODE=$?

# Check if the command timed out
if [ $EXIT_CODE -eq 124 ]; then
    echo "Error: The operation timed out after ${timeout_duration} seconds."
    exit 124
elif [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Command failed with exit code $EXIT_CODE."
    exit $EXIT_CODE
else
    # If we reach here, the command has completed successfully
    echo "------------------------------------------------------------"
    echo "Vitis AI command completed successfully."
fi
