# FPGA-CNN Deployment

## Overview
This repository provides scripts, models, and tools for benchmarking and deploying CNN models (such as MobileNetV2, EfficientNet, and ResNet50) on FPGA hardware. It supports model conversion, quantization, benchmarking, and visualization of inference results using a subset of the ImageNet validation dataset.

## Quick Start Guide

### 1. Set Up Your Environment

```bash
# Create and activate a conda environment
conda create -n fpga-cnn python=3.12
conda activate fpga-cnn

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up your environment (adds ~/.local/bin to PATH)
source setup_env.sh

# Verify your installation
python3 env_check.py
```

*For GPU acceleration (recommended for benchmarking), refer to the `requirements.txt` file*

#### For Vitis AI Optimization and Quantization

For model optimization and quantization with Vitis AI, you need to use the Xilinx Vitis AI Docker container:

```bash
# Use the provided helper script to start the Docker container
./run_vitis_docker.sh

# Inside the container, you can run optimization and quantization scripts
python3 scripts/optimization/optimize_model.py --help
python3 scripts/quantization/quantize_model.py --help

# Example: Optimize and quantize a model with random selection of calibration data
python3 scripts/optimization/optimize_model.py --model models/onnx/resnet18-v1-7.onnx --format onnx --pruning --arch_opt
python3 scripts/quantization/quantize_model.py --model models/optimized/resnet18-v1-7/final_optimized_model.onnx --format onnx --random_selection
```

Alternatively, you can run the Docker container manually:

```bash
# Pull the Vitis AI Docker image
docker pull xilinx/vitis-ai-pytorch-cpu:latest

# Run the Docker container, mounting your project directory
docker run -it --rm -v /path/to/fpga-cnn-deployment:/workspace xilinx/vitis-ai-pytorch-cpu:latest

# Inside the container, set up the environment
cd /workspace
source setup_env.sh
```

### 2. Prepare Data

Place your ImageNet validation subset in `data/imagenet/val_subset/` and calibration subset in `data/imagenet/cal_subset/`. Make sure `data/imagenet/imagenet_class_index.json` is available.

### 3. Run Benchmarks

```bash
# Run all ONNX model benchmarks on CPU
python scripts/benchmarking/onnx_benchmark.py --device cpu

# Run specific PyTorch model benchmark on GPU (if available)
python scripts/benchmarking/pytorch/resnet50_benchmark.py --device cuda

# Visualize benchmark results
python scripts/benchmarking/plot_benchmark.py
```

### 4. Model Conversion & Quantization

Refer to the scripts in `scripts/quantization/` directory for converting and quantizing models.

### 5. FPGA Deployment

Refer to the scripts in `scripts/deployment/` directory for compiling models for FPGA deployment.

## Directory Structure

```
fpga-cnn-deployment/
├── models/                       # Model storage
│   └── onnx/                     # ONNX format models
│       ├── standard/             # Standard ONNX models
│       └── quantized/            # Quantized ONNX models
├── data/                         # All data related files
│   ├── imagenet/                 # ImageNet dataset subsets
│   │   ├── cal_subset/           # Calibration set
│   │   └── val_subset/           # Validation set
│   └── results/                  # All benchmark and inference results
├── scripts/                      # All scripts
│   ├── benchmarking/             # Benchmarking scripts
│   ├── quantization/             # Quantization scripts
│   └── deployment/               # Deployment scripts
├── utils/                        # Utility functions used across scripts
├── env_check.py                  # Environment verification script
└── README.md                     # Project overview and instructions
```

## Common Tasks

### Running a Single Model Benchmark

```bash
# Run ResNet50 ONNX benchmark
python scripts/benchmarking/onnx/resnet50_onnx_benchmark.py --device cpu

# Run MobileNetV2 PyTorch benchmark
python scripts/benchmarking/pytorch/mobilenet_v2_benchmark.py --device cpu
```

### Visualizing Results

```bash
# Generate benchmark comparison plots
python scripts/benchmarking/plot_benchmark.py
```

### Checking Environment

```bash
# Verify your environment setup
python env_check.py
```

## Troubleshooting

- **Missing CUDA support?** Check requirements.txt for GPU installation instructions
- **Model not found?** Place models in their respective directories under `models/`
- **Import errors?** Verify your environment with `python env_check.py`

## Notes

- Large datasets and model binaries are not tracked in Git (see `.gitignore`)
- For best performance, use a CUDA-compatible GPU with appropriate drivers
- The Vitis-AI directory is excluded from version control

## License
Refer to individual model licenses as needed. ImageNet usage is subject to its terms and conditions.