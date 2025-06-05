# onnx_benchmark.py
# Run all ONNX benchmarking scripts in the onnx folder with both standard and quantized variants

import subprocess
import sys
import os
import argparse
import glob

# Parse arguments
parser = argparse.ArgumentParser(description='Run all ONNX benchmarks')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                   help='Device to run benchmarks on')
parser.add_argument('--skip_quant', action='store_true',
                   help='Skip quantized models (only run standard models)')
args = parser.parse_args()

# Base directories
SCRIPT_DIR = os.path.dirname(__file__)
ONNX_DIR = os.path.join(SCRIPT_DIR, 'onnx')

# Find all *_onnx_benchmark.py scripts in the onnx directory
onnx_scripts = glob.glob(os.path.join(ONNX_DIR, '*_onnx_benchmark.py'))

# Map script base names to model names
script_model_map = {
    'mobilenet_v2_onnx_benchmark.py': ['mobilenet_v2', 'mobilenet_v2_quant'],
    'resnet50_onnx_benchmark.py': ['resnet50', 'resnet50_quant'],
    'efficientnet_lite4_onnx_benchmark.py': ['efficientnet_lite4', 'efficientnet_lite4_quant'],
}

# Build benchmark jobs: (script_path, model_name)
jobs = []
for script_path in onnx_scripts:
    script_name = os.path.basename(script_path)
    if script_name in script_model_map:
        models = script_model_map[script_name]
        if args.skip_quant:
            models = [m for m in models if not m.endswith('_quant')]
        for model in models:
            jobs.append((script_path, model))
    else:
        # Fallback: use base name logic
        if script_name.endswith('_onnx_benchmark.py'):
            base = script_name[:-len('_onnx_benchmark.py')]
            jobs.append((script_path, base))
            if not args.skip_quant:
                jobs.append((script_path, base + '_quant'))

# Count total benchmarks
total_benchmarks = len(jobs)
completed_benchmarks = 0

print(f"Starting ONNX benchmarking on {args.device}")
print(f"Total benchmarks to run: {total_benchmarks}")
print("=" * 80)

# Run all benchmarks
for script_path, model in jobs:
    completed_benchmarks += 1
    script_name = os.path.basename(script_path)
    print(f"\n[{completed_benchmarks}/{total_benchmarks}] Running {script_name} for model {model} on {args.device}")
    print("-" * 80)
    
    cmd = [sys.executable, script_path, '--model', model, '--device', args.device]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Successfully completed {model}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name} for model {model}")
        print(f"  Return code: {e.returncode}")
    except Exception as e:
        print(f"✗ Unexpected error running {script_name} for model {model}")
        print(f"  Error: {str(e)}")

print("\n" + "=" * 80)
print(f"ONNX benchmarks completed: {completed_benchmarks}/{total_benchmarks}")
print("Results saved to: /home/ethanwst/FPGA-Accelerator/cnn_benchmarking/benchmark_results/onnx_benchmark_results.csv")