# FPGA-CNN Deployment

## Overview
This repository provides scripts, models, and tools for benchmarking and deploying CNN models (such as MobileNetV2, EfficientNet, and ResNet50) on FPGA hardware. It supports model conversion, quantization, benchmarking, and visualization of inference results using a subset of the ImageNet validation dataset.

## Directory Structure

```
models/           # Original or ONNX-exported models
quantized/        # INT8 models (quantized)
compiled/         # FPGA-compiled .xmodel files
hls/              # HLS ops (custom or upcoming)
scripts/          # Benchmarking and utility scripts
  benchmark_scripts/
    onnx_benchmark.py
    plot_benchmark.py
    pytorch_benchmark.py
    onnx/
      efficientnet_lite4_onnx_benchmark.py
      mobilenet_v2_onnx_benchmark.py
      resnet50_onnx_benchmark.py
    pytorch/
      mobilenet_v2_benchmark.py
      mobilenet_v3_large_benchmark.py
      resnet18_benchmark.py
      resnet50_benchmark.py
data/             # Inference results, calibration sets, and datasets
  benchmark_results/
    onnx_benchmark_results.csv
    pytorch_benchmark_results.csv
  imagenet_val_set/
    imagenet_class_index.json
notebooks/        # Optional interactive work (Jupyter, etc.)
docs/             # Diagrams, slides, and documentation
README.md         # Project overview and instructions
.gitignore        # Ignore model binaries, logs, etc.
```

## Getting Started

1. **Set up your environment:**
   - Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed.
   - Create and activate a conda environment:
     ```sh
     conda create -n fpga-cnn python=3.10
     conda activate fpga-cnn
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

2. **Prepare ImageNet subset and class index:**
   - Download the ImageNet validation subset (update when kaggle is public)
   - Place the validation images in `data/imagenet/imagenet_val_set/`.


## Notes
- Large datasets and model binaries are not tracked in git (see `.gitignore`).
- All scripts are designed to be run from the `scripts/benchmark_scripts/` directory unless otherwise noted.
- For best results, use Python 3.8+ and the package versions in `requirements.txt`.

## License
See `data/imagenet_val_set/ILSVRC2012_devkit_t12/COPYING` for the ImageNet devkit license and refer to individual model licenses as needed.