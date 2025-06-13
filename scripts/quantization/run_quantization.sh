#!/bin/bash
# 
# Run Vitis AI quantization script inside Vitis AI Docker container
# This script provides a convenient way to run the vitis_ai_quantize.py script
# using the "vitisai" alias you've already set up.

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# Display help information
display_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Quantize models using Vitis AI for Xilinx U50 FPGA"
    echo
    echo "Options:"
    echo "  --model MODEL    Model name to quantize (without extension)"
    echo "  --list           List available models"
    echo "  --all            Quantize all available models"
    echo "  --output DIR     Output directory for quantized models"
    echo "  --verbose, -v    Enable verbose output"
    echo "  --help, -h       Display this help message"
    echo
    echo "Examples:"
    echo "  $0 --list                    List available models"
    echo "  $0 --model mobilenetv2-12    Quantize mobilenetv2-12 model"
    echo "  $0 --all                     Quantize all available models"
}

# Parse command line arguments
ARGS=""
for arg in "$@"; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
        display_help
        exit 0
    else
        ARGS="$ARGS $arg"
    fi
done

# Check if the vitisai_launch.sh script is available
VITISAI_LAUNCHER="$PROJECT_ROOT/scripts/deployment/vitisai_launch.sh"
if [ ! -x "$VITISAI_LAUNCHER" ]; then
    echo "Error: The Vitis AI launcher script is not available or not executable."
    echo "Please make sure $VITISAI_LAUNCHER exists and is executable."
    exit 1
fi

# Run the script inside the Vitis AI container
echo "Starting Vitis AI quantization process..."

# Add a verbose flag to Python command if requested
PYTHON_ARGS="$ARGS"
if [[ "$ARGS" == *"--verbose"* ]] || [[ "$ARGS" == *"-v"* ]]; then
    # Add additional verbosity for Python script
    PYTHON_ARGS="$PYTHON_ARGS --debug"
fi

# Add progress indicator command
CMD="cd /workspace && python3 scripts/quantization/vitis_ai_quantize.py $PYTHON_ARGS"
echo "Running: $CMD"
echo "------------------------------------------------------------"
echo "Status: Initializing Vitis AI environment..."
echo "This may take a minute to start up. Please be patient."
echo "------------------------------------------------------------"

# Start time tracking
START_TIME=$(date +%s)

# Execute the command using the launcher script directly
"$VITISAI_LAUNCHER" bash -c "$CMD"

# End time tracking
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Check the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "------------------------------------------------------------"
    echo "ERROR: Quantization process timed out after 30 minutes."
    echo "This may indicate a problem with the Vitis AI container or model."
    echo "Try running with --verbose flag for more detailed output."
    exit $EXIT_CODE
elif [ $EXIT_CODE -ne 0 ]; then
    echo "------------------------------------------------------------"
    echo "ERROR: Quantization failed with exit code: $EXIT_CODE"
    echo "Process ran for $DURATION seconds before failing."
    exit $EXIT_CODE
else
    echo "------------------------------------------------------------"
    echo "SUCCESS: Quantization completed successfully."
    echo "Total processing time: $DURATION seconds."
fi
