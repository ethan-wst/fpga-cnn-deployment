#!/bin/bash
#
# Vitis AI Environment Check Script
#
# This script checks if the Vitis AI environment is properly set up
# and the Docker container can be launched successfully.

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
VITISAI_LAUNCHER="$PROJECT_ROOT/scripts/deployment/vitisai_launch.sh"

echo "===== Vitis AI Environment Check ====="

# Check if Docker is installed
echo -n "Checking Docker installation... "
if command -v docker &> /dev/null; then
    echo "OK"
else
    echo "FAILED"
    echo "Error: Docker is not installed or not in PATH."
    exit 1
fi

# Check Docker service
echo -n "Checking Docker service... "
if docker info &> /dev/null; then
    echo "OK"
else
    echo "FAILED"
    echo "Error: Docker service is not running or current user doesn't have permission."
    echo "Try: sudo service docker start"
    echo "Or:  sudo systemctl start docker"
    exit 1
fi

# Check Vitis AI image
VITIS_AI_IMAGE="xilinx/vitis-ai-pytorch-gpu:3.5.0.001-1eed93cde"
echo -n "Checking Vitis AI image ($VITIS_AI_IMAGE)... "
if docker image inspect "$VITIS_AI_IMAGE" &> /dev/null; then
    echo "OK"
else
    echo "NOT FOUND"
    echo "Error: Vitis AI Docker image not found."
    echo "Run: docker pull $VITIS_AI_IMAGE"
    exit 1
fi

# Check if launcher script exists and is executable
echo -n "Checking launcher script... "
if [ -x "$VITISAI_LAUNCHER" ]; then
    echo "OK"
else
    echo "FAILED"
    echo "Error: Launcher script not found or not executable: $VITISAI_LAUNCHER"
    exit 1
fi

# Check if we can run a simple command in the container
echo -n "Testing Docker container... "
OUTPUT=$($VITISAI_LAUNCHER echo "Vitis AI container is working" 2>&1)
if [[ "$OUTPUT" == *"Vitis AI container is working"* ]]; then
    echo "OK"
else
    echo "FAILED"
    echo "Error: Could not run test command in Vitis AI container."
    echo "Output:"
    echo "$OUTPUT"
    exit 1
fi

# Print success message
# Check for ONNX models
ONNX_DIR="/home/ethanwst/fpga-cnn-deployment/models/onnx/standard"
echo -n "Checking for ONNX models... "
MODEL_COUNT=$(ls $ONNX_DIR/*.onnx 2> /dev/null | wc -l)
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "OK ($MODEL_COUNT models found)"
    echo "Available models:"
    for model in $ONNX_DIR/*.onnx; do
        echo "  - $(basename $model)"
    done
else
    echo "NONE FOUND"
    echo "Warning: No ONNX models found in $ONNX_DIR"
    echo "You need to add .onnx model files to this directory before quantization."
fi

# Check calibration data for quantization
CALIBRATION_DIR="/home/ethanwst/fpga-cnn-deployment/data/imagenet/cal_subset"
echo -n "Checking for calibration data... "
if [ -d "$CALIBRATION_DIR" ] && [ "$(ls -A $CALIBRATION_DIR)" ]; then
    echo "OK"
else
    echo "WARNING"
    echo "Calibration data might be missing in $CALIBRATION_DIR"
    echo "Quantization requires calibration images to proceed."
fi

echo
echo "✅ Vitis AI environment check passed!"
echo "✅ Your environment is properly configured."
echo
echo "You can now use the following commands:"
echo "  - ./scripts/quantization/run_quantization.sh --list"
echo "  - ./scripts/quantization/run_quantization.sh --model <model_name>"
echo
echo "For more information, see the README files in scripts/quantization/ and scripts/deployment/"
