# FPGA Deployment with Vitis AI

This directory contains scripts for deploying quantized models to Xilinx FPGAs using Vitis AI.

## Prerequisites

- Xilinx Vitis AI 3.5 installed
- Docker installed and configured
- Quantized models available in the `models/onnx/quantized/` directory

## Scripts

### `vitisai_launch.sh`

A utility script for launching the Vitis AI Docker container with the appropriate volume mounts and configurations. This script is used by other tools in the repository such as the quantization scripts.

```bash
# To run a command in the Vitis AI container:
./vitisai_launch.sh <command>

# For example:
./vitisai_launch.sh python /workspace/my_script.py

# To get an interactive shell:
./vitisai_launch.sh
```

## Deployment Process

1. Ensure your model has been successfully quantized using the scripts in the `scripts/quantization` directory
2. Configure your deployment parameters for the target FPGA device
3. Use the Vitis AI tools to compile the model for your specific target hardware
4. Deploy the compiled model to your FPGA device

For detailed instructions on deploying to specific FPGA platforms, refer to the Xilinx Vitis AI documentation.
