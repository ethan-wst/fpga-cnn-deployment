# Model Optimization with Vitis AI

This directory contains scripts and utilities for optimizing deep learning models using Xilinx's Vitis AI framework. These optimizations prepare models for efficient deployment on Xilinx FPGA hardware.

## How Vitis AI Optimization Works

Vitis AI provides a comprehensive suite of tools to optimize deep learning models for deployment on Xilinx FPGAs. The optimization process includes several techniques:

### 1. Model Pruning

Pruning reduces model size by removing less important connections (weights) in a neural network:

- **Weight Magnitude Pruning**: Removes weights based on their absolute magnitude, assuming smaller weights contribute less to the output
- **Structured Pruning**: Removes entire filters/channels to maintain regular computational patterns that are more hardware-friendly
- **Benefits**: Smaller model size, reduced computational requirements, and faster inference

### 2. Architecture Optimization

Optimizes the network architecture while preserving the model's functionality:

- **Operator Fusion**: Combines multiple operators into a single optimized operator (e.g., Conv+BatchNorm+ReLU)
- **Layer Collapsing**: Merges consecutive layers when possible
- **Constant Folding**: Pre-computes constant expressions
- **Redundancy Elimination**: Removes redundant operations
- **Benefits**: Reduced latency, fewer computational resources, and more efficient hardware usage

### 3. ONNX Export

The optimization process also exports the optimized model to ONNX format for broader compatibility:

- **Standard Format**: ONNX provides interoperability between different deep learning frameworks
- **Better Support**: Many deployment tools work better with ONNX models
- **Optimized Graph**: The exported ONNX model contains all the optimizations applied

### 4. Target-Specific Optimizations

When a specific FPGA target is specified, additional optimizations are performed:

- **Hardware-Aware Optimizations**: Adjustments based on specific hardware resources
- **Datatype Optimizations**: Precision adjustments to match hardware capabilities
- **Memory Layout Optimizations**: Data arrangement for better memory access patterns

## Running Model Optimization

### Prerequisites

- Vitis AI Docker environment (see `scripts/deployment` folder)
- Pre-trained models in PyTorch, ONNX, or TensorFlow format

### Basic Usage

```bash
# Inside Vitis AI Docker container
python scripts/optimization/optimize_model.py \
    --model <path_to_model> \
    --format <pytorch|onnx|tensorflow> \
    --output_dir <output_directory> \
    [--pruning] [--arch_opt] [--verbose]
```

### Example Commands

1. **Basic optimization** (architecture optimization only):

```bash
python scripts/optimization/optimize_model.py \
    --model ./models/pytorch/resnet18.pth \
    --format pytorch \
    --output_dir ./models/optimized/
```

2. **Full optimization** (pruning + architecture optimization):

```bash
python scripts/optimization/optimize_model.py \
    --model ./models/pytorch/resnet18.pth \
    --format pytorch \
    --output_dir ./models/optimized/ \
    --pruning \
    --arch_opt
```

3. **Running in Vitis AI Docker container**:

```bash
./run_vitis_docker.sh python scripts/optimization/optimize_model.py \
    --model ./models/pytorch/resnet18.pth \
    --format pytorch \
    --output_dir ./models/optimized/
```

### Optimization Parameters

- `--model`: Path to the input model file
- `--format`: Model format (pytorch, onnx, or tensorflow)
- `--output_dir`: Directory to save optimized models (default: models/optimized)
- `--pruning`: Enable model pruning for weight reduction
- `--arch_opt`: Enable architecture optimizations
- `--verbose`: Enable detailed logging

## Output Files

The optimization process produces several output files:

- `final_optimized_model.pth`: The fully optimized PyTorch model (if input is PyTorch)
- `optimized_model.onnx`: ONNX version of the optimized model
- Other intermediate files depending on the optimization steps applied

## Next Steps

After optimization, models are ready for:

- Quantization (see `scripts/quantization` folder)
- Compilation for specific FPGA targets
- Deployment on Xilinx FPGA hardware

## References

- [Vitis AI Developer Guide](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai)
- [Vitis AI GitHub Repository](https://github.com/Xilinx/Vitis-AI)
- [ONNX Model Zoo](https://github.com/onnx/models)
