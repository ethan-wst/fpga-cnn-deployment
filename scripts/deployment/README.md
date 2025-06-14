# FPGA Deployment with Vitis AI

This directory contains scripts for deploying quantized models to Xilinx FPGAs using Vitis AI.

## Prerequisites

- Xilinx Vitis AI 3.5 installed
- Docker installed and configured
- Quantized models available in the `models/onnx/quantized/` directory

## Docker Container

The project uses a centralized Docker launcher script located in the project root directory:

```bash
# To run a command in the Vitis AI container:
../../run_vitis_docker.sh <command>

# For example:
../../run_vitis_docker.sh python scripts/quantization/quantize_model.py --model model.pth

# To get an interactive shell:
../../run_vitis_docker.sh
```

## Deployment Process

1. Ensure your model has been successfully quantized using the scripts in the `scripts/quantization` directory
2. Configure your deployment parameters for the target FPGA device
3. Use the Vitis AI tools to compile the model for your specific target hardware
4. Deploy the compiled model to your FPGA device

For detailed instructions on deploying to specific FPGA platforms, refer to the Xilinx Vitis AI documentation.
