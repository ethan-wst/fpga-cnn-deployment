# Model Quantization with Vitis AI

This directory contains scripts for quantizing deep learning models using Xilinx's Vitis AI framework. Quantization is a critical step for efficient deployment of neural networks on FPGA hardware, reducing memory footprint and improving inference performance.

## Understanding Quantization

### What is Quantization?

Quantization is the process of converting the weights and activations of a neural network from high-precision floating-point (FP32) to lower-precision formats (like INT8). This significantly reduces:

- **Memory usage**: INT8 requires 4x less memory than FP32
- **Computational complexity**: Integer operations are faster and more power-efficient
- **Data transfer bandwidth**: Less data needs to be moved between memory and compute units

### Types of Quantization in Vitis AI

Vitis AI 3.5 supports several quantization approaches:

1. **Post-Training Quantization (PTQ)**:
   - Applied after a model is trained
   - Requires a small calibration dataset (no labels needed)
   - Minimal accuracy loss for most models
   - **This is the primary method implemented in these scripts**

2. **Quantization-Aware Training (QAT)**:
   - Simulates quantization effects during training
   - Usually achieves higher accuracy than PTQ
   - Requires retraining the model

3. **Precision Options**:
   - **INT8**: Standard 8-bit integer quantization (most common)
   - **Mixed Precision**: Keeps sensitive layers in higher precision
   - **Binary/Ternary**: Extreme quantization for specific use cases

### How Vitis AI Quantization Works

The quantization process consists of several key steps:

1. **Calibration**: Uses a small dataset to determine the dynamic range of activations
   - Different calibration methods: min-max, entropy, percentile
   - Typically needs 100-1000 images for accurate calibration

2. **Scale Factor Determination**: Calculates optimal scale factors for each layer
   - Maps the floating-point range to integer range
   - Minimizes quantization error

3. **Quantization Parameter Insertion**: Adds quantization nodes to the model
   - These nodes define how values are converted between formats

4. **Bias Correction**: Compensates for systematic errors introduced by quantization

5. **Fine-tuning** (optional): Makes small adjustments to weights to recover accuracy

## Prerequisites

- Xilinx Vitis AI 3.5 Docker image installed
- Docker environment properly configured
- Calibration dataset in the expected location (`data/imagenet/cal_subset/`)
- Pre-trained models (PyTorch, ONNX, or TensorFlow)

## Using the Quantization Scripts

### Basic Usage

```bash
# Inside Vitis AI Docker container
python scripts/quantization/quantize_model.py \
    --model <path_to_model> \
    --format <pytorch|onnx|tensorflow> \
    --calib_dataset <path_to_calibration_data> \
    [--output_dir <output_directory>] \
    [--precision <int8|mixed>] \
    [--verbose]
```

### Running with Docker Script

```bash
# Using the Vitis AI Docker launch script
./run_vitis_docker.sh python scripts/quantization/quantize_model.py \
    --model ./models/optimized/optimized_model.onnx \
    --format onnx \
    --calib_dataset ./data/imagenet/cal_subset
```

### Example Commands

1. **Quantize PyTorch Model**:

```bash
python scripts/quantization/quantize_model.py \
    --model ./models/pytorch/resnet18.pth \
    --format pytorch \
    --calib_dataset ./data/imagenet/cal_subset
```

2. **Quantize ONNX Model**:

```bash
python scripts/quantization/quantize_model.py \
    --model ./models/optimized/optimized_model.onnx \
    --format onnx \
    --calib_dataset ./data/imagenet/cal_subset
```

3. **Specify Custom Output Directory**:

```bash
python scripts/quantization/quantize_model.py \
    --model ./models/pytorch/mobilenet_v2.pth \
    --format pytorch \
    --calib_dataset ./data/imagenet/cal_subset \
    --output_dir ./models/quantized/custom_dir
```



### Quantization Parameters

- `--model`: Path to the input model file
- `--format`: Model format (pytorch, onnx, or tensorflow)
- `--calib_dataset`: Path to calibration dataset
- `--output_dir`: Directory to save quantized models (default: models/quantized)
- `--precision`: Quantization precision (int8 or mixed)
- `--batch_size`: Batch size for calibration (default: 32)
- `--max_samples`: Maximum number of samples to use for calibration (default: 1000)
- `--verbose`: Enable detailed logging

## Workflow Details

1. **Model Loading**: The appropriate loader is selected based on the model format
2. **Calibration Data Preparation**: Images are preprocessed according to model requirements
3. **Quantizer Configuration**: Parameters are set for target hardware
4. **Calibration**: The model is analyzed with the calibration dataset
5. **Quantization**: The model weights and activations are quantized
6. **Validation** (optional): The quantized model is validated against test data
7. **Export**: The quantized model is saved in the appropriate format

## Output Files

The quantization process produces several output files:

- `<model_name>_int8.onnx`: INT8 quantized ONNX model
- `<model_name>_int8.pth`: INT8 quantized PyTorch model (if input is PyTorch)
- Quantization statistics and reports (in JSON format)

## Evaluation

After quantization, you can evaluate the model's accuracy to ensure it hasn't degraded significantly:

```bash
python scripts/benchmarking/evaluate_model.py \
    --model ./models/quantized/<model_name>_int8.onnx \
    --format onnx \
    --val_dataset ./data/imagenet/val_subset
```

## References

- [Vitis AI Quantizer Documentation](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Quantizer)
- [Vitis AI Quantization Methodology](https://www.xilinx.com/content/dam/xilinx/support/documents/white_papers/wp540-vitis-ai-quantization.pdf)
- [ONNX Quantization Specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#quantizelinear)
