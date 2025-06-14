# Vitis AI Model Optimization and Quantization

This directory contains scripts to optimize and quantize deep learning models using Vitis AI tools.
The scripts are designed to work with the Xilinx/Vitis-AI-pytorch-cpu Docker image.

## Prerequisites

1. Docker with Vitis AI image:
```bash
docker pull xilinx/vitis-ai-pytorch-cpu:latest
docker run -it --rm -v /path/to/your/project:/workspace xilinx/vitis-ai-pytorch-cpu:latest
```

2. Calibration dataset:
   - Place a subset of ImageNet or your custom dataset in `data/imagenet/cal_subset/`
   - For best results, use at least 100 diverse images that represent your inference data

## Directory Structure

```
scripts/
  ├── optimization/             # Model optimization scripts
  │   ├── optimize_model.py     # Main optimization script
  │   ├── pytorch_optimizer.py  # PyTorch-specific optimizations
  │   ├── onnx_optimizer.py     # ONNX-specific optimizations
  │   └── tensorflow_optimizer.py  # TensorFlow-specific optimizations
  │
  └── quantization/             # Model quantization scripts
      ├── quantize_model.py     # Main quantization script
      ├── pytorch_quantizer.py  # PyTorch-specific quantization
      ├── onnx_quantizer.py     # ONNX-specific quantization
      └── tensorflow_quantizer.py  # TensorFlow-specific quantization
```

## Usage

### Model Optimization

Optimize a model for better performance on Xilinx FPGA:

```bash
python scripts/optimization/optimize_model.py \
  --model path/to/your/model.pth \
  --format pytorch \
  --pruning \
  --arch_opt \
  --verbose
```

Options:
- `--model`: Path to input model file
- `--format`: Model format (pytorch, onnx, tensorflow)
- `--output_dir`: Output directory (optional)
- `--pruning`: Apply model pruning (optional)
- `--arch_opt`: Apply architecture optimizations (optional)
- `--verbose`: Enable verbose logging (optional)

### Model Quantization

Quantize a model for deployment on Xilinx FPGA:

```bash
python scripts/quantization/quantize_model.py \
  --model path/to/your/model.pth \
  --format pytorch \
  --calib_dataset data/imagenet/cal_subset \
  --precision int8 \
  --verbose
```

Options:
- `--model`: Path to input model file
- `--format`: Model format (pytorch, onnx, tensorflow)
- `--calib_dataset`: Path to calibration dataset
- `--output_dir`: Output directory (optional)
- `--precision`: Quantization precision (int8, mixed)
- `--batch_size`: Batch size for calibration
- `--max_samples`: Maximum number of calibration samples
- `--verbose`: Enable verbose logging (optional)

## Workflow Examples

### Example 1: PyTorch ResNet50 Optimization and Quantization

```bash
# Step 1: Optimize the model
python scripts/optimization/optimize_model.py \
  --model models/original/resnet50.pth \
  --format pytorch \
  --pruning \
  --arch_opt

# Step 2: Quantize the optimized model
python scripts/quantization/quantize_model.py \
  --model models/optimized/resnet50/final_optimized_model.pth \
  --format pytorch \
  --calib_dataset data/imagenet/cal_subset
```

### Example 2: ONNX MobileNetV2 Optimization and Quantization

```bash
# Step 1: Optimize the model
python scripts/optimization/optimize_model.py \
  --model models/onnx/standard/mobilenetv2-12.onnx \
  --format onnx

# Step 2: Quantize the optimized model
python scripts/quantization/quantize_model.py \
  --model models/optimized/mobilenetv2-12/final_optimized_model.onnx \
  --format onnx \
  --calib_dataset data/imagenet/cal_subset
```

## Notes

- For PyTorch models, the scripts generate both PyTorch and ONNX outputs for flexibility.
- When running in the Vitis AI Docker container, make sure to activate the conda environment:
  ```bash
  conda activate vitis-ai-pytorch
  ```
- The optimization and quantization processes are computationally intensive; ensure adequate resources.
- Always verify the model accuracy after optimization and quantization to ensure acceptable results.
