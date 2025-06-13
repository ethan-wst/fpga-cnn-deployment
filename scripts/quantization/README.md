# Model Quantization with Vitis AI

This directory contains scripts for quantizing neural network models using Xilinx Vitis AI, targeting the U50 FPGA platform.

## Prerequisites

- Xilinx Vitis AI 3.5 Docker image installed
- `vitisai` alias set up for launching the Vitis AI container
- Calibration dataset in the expected location (`data/imagenet/cal_subset/`)
- ONNX models in the standard model directory (`models/onnx/standard/`)

## Quantization Scripts

### `run_quantization.sh`

This is a wrapper script that runs the Python quantization script inside the Vitis AI Docker container.

```bash
# List available models
./run_quantization.sh --list

# Quantize a specific model
./run_quantization.sh --model mobilenetv2-12

# Quantize all available models
./run_quantization.sh --all

# Specify a custom output directory
./run_quantization.sh --model resnet50-v1-12 --output /path/to/output/directory
```

### `vitis_ai_quantize.py`

This is the main Python script that performs the quantization. It's designed to be run inside the Vitis AI environment and uses the ImageNet calibration dataset.

```bash
# Run directly (inside Vitis AI container)
python vitis_ai_quantize.py --model mobilenetv2-12
```

## Workflow

1. The script loads the specified ONNX model
2. Prepares calibration data from the ImageNet calibration subset
3. Configures the Vitis AI quantizer for the U50 target
4. Performs quantization using the calibration data
5. Compiles the quantized model for the U50 target
6. Saves the output to the specified directory (default: `models/onnx/quantized/`)

## Output

The quantized models are saved in the `models/onnx/quantized/` directory by default, with the naming convention `[model_name]_quantized.onnx`.

## Extending the Scripts

To support more models or targets:

1. Add your models to the `models/onnx/standard/` directory
2. Update the quantization configuration in `vitis_ai_quantize.py` if needed
3. Run the quantization script with the appropriate options
