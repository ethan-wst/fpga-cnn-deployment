# pytorch_benchmark.py
# Run all PyTorch benchmarking scripts in the pytorch directory for both standard and quantized variants on both CPU and CUDA

import subprocess
import sys
import os
import glob
import argparse

SCRIPT_DIR = os.path.dirname(__file__)
PYTORCH_DIR = os.path.join(SCRIPT_DIR, 'pytorch')

# Find all *_benchmark.py scripts in the pytorch directory
pytorch_scripts = glob.glob(os.path.join(PYTORCH_DIR, '*_benchmark.py'))

# Map script base names to model names
script_model_map = {
    'mobilenet_v2_benchmark.py': ['mobilenet_v2', 'mobilenet_v2_quant'],
    'mobilenet_v3_large_benchmark.py': ['mobilenet_v3_large', 'mobilenet_v3_large_quant'],
    'resnet18_benchmark.py': ['resnet18', 'resnet18_quant'],
    'resnet50_benchmark.py': ['resnet50', 'resnet50_quant'],
}

# Parse arguments
parser = argparse.ArgumentParser(description='Run all PyTorch benchmarks')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'all'],
                   help='Device to run benchmarks on (cpu, cuda, or all)')
parser.add_argument('--skip_quant', action='store_true',
                   help='Skip quantized models (only run standard models)')
args = parser.parse_args()

# Devices to benchmark on
if args.device == 'all':
    devices = ['cpu', 'cuda']
else:
    devices = [args.device]

# Build benchmark jobs: (script_path, model_name, device)
jobs = []
for script_path in pytorch_scripts:
    script_name = os.path.basename(script_path)
    if script_name in script_model_map:
        models = script_model_map[script_name]
        if args.skip_quant:
            models = [m for m in models if not m.endswith('_quant')]
        for model in models:
            for device in devices:
                jobs.append((script_path, model, device))
    else:
        # Fallback: use base name logic
        if script_name.endswith('_benchmark.py'):
            base = script_name[:-len('_benchmark.py')]
            jobs.append((script_path, base, 'cpu'))
            jobs.append((script_path, base, 'cuda'))
            if not args.skip_quant:
                jobs.append((script_path, base + '_quant', 'cpu'))
                jobs.append((script_path, base + '_quant', 'cuda'))

# Count total benchmarks
total_benchmarks = len(jobs)
completed_benchmarks = 0

print(f"Starting PyTorch benchmarking on CPU and CUDA")
print(f"Total benchmarks to run: {total_benchmarks}")
print("=" * 80)

# Run all benchmarks
for script_path, model, device in jobs:
    completed_benchmarks += 1
    script_name = os.path.basename(script_path)
    print(f"\n[{completed_benchmarks}/{total_benchmarks}] Running {script_name} for model {model} on {device}")
    print("-" * 80)
    cmd = [sys.executable, script_path, '--model', model, '--device', device]
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Successfully completed {model} on {device}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name} for model {model} on {device}")
        print(f"  Return code: {e.returncode}")
    except Exception as e:
        print(f"✗ Unexpected error running {script_name} for model {model} on {device}")
        print(f"  Error: {str(e)}")

print("\n" + "=" * 80)
print(f"PyTorch benchmarks completed: {completed_benchmarks}/{total_benchmarks}")
print("Results saved to: /home/ethanwst/FPGA-Accelerator/cnn_benchmarking/benchmark_results/pytorch_benchmark_results.csv")
