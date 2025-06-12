#!/usr/bin/env python3
"""
Environment check utility for FPGA-CNN deployment.
Run this script to verify that all required dependencies are installed and configured correctly.
"""

import importlib
import sys
import platform
import subprocess
from importlib.util import find_spec
from packaging.version import Version as parse_version

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets minimum version requirement."""
    try:
        if find_spec(package_name) is not None:
            pkg = importlib.import_module(package_name)
            if hasattr(pkg, '__version__'):
                version = pkg.__version__
                print(f"✅ {package_name} {version} is installed")
                if min_version and parse_version(version) < parse_version(min_version):
                    print(f"⚠️  Warning: {package_name} version {version} is older than recommended {min_version}")
                return True
            else:
                print(f"✅ {package_name} is installed (version not available)")
                return True
        else:
            print(f"❌ {package_name} is not installed")
            return False
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.version.cuda}")
            print(f"✅ GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ CUDA is not available for PyTorch")
    except ImportError:
        print("❌ PyTorch is not installed")

def check_onnx_provider(provider):
    """Check if an ONNX provider is available."""
    try:
        import onnxruntime as ort
        if provider in ort.get_available_providers():
            print(f"✅ {provider} is available")
            return True
        else:
            print(f"❌ {provider} is not available")
            if provider == "CUDAExecutionProvider":
                print("   Note: You need to install onnxruntime-gpu for CUDA support")
                print("   Check requirements.txt for the correct CUDA-compatible version")
            return False
    except ImportError:
        print("❌ ONNX Runtime is not installed")
        return False

def check_cuda_toolkit():
    """Check CUDA toolkit availability."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ CUDA Toolkit is installed:")
            for line in result.stdout.strip().split('\n'):
                if "release" in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ CUDA Toolkit is not available (nvcc not found)")
            return False
    except FileNotFoundError:
        print("❌ CUDA Toolkit is not available (nvcc not found)")
        return False

def main():
    """Run all environment checks."""
    print("=" * 60)
    print("FPGA-CNN Deployment Environment Check")
    print("=" * 60)
    
    print("\nSystem Information:")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    
    print("\nChecking core dependencies:")
    check_package("numpy", "1.24.0")
    check_package("pandas", "2.0.0")
    check_package("matplotlib", "3.7.0")
    check_package("PIL", "10.0.0")  # Pillow
    
    # OpenCV requires special handling
    try:
        import cv2
        cv_version = cv2.__version__
        print(f"✅ OpenCV {cv_version} is installed")
        if parse_version(cv_version) < parse_version("4.8.0"):
            print(f"⚠️  Warning: OpenCV version {cv_version} is older than recommended 4.8.0")
    except ImportError:
        print("❌ OpenCV (cv2) is not installed")
    
    print("\nChecking deep learning frameworks:")
    check_package("torch", "2.0.0")
    check_package("torchvision", "0.15.0")
    check_package("onnxruntime", "1.15.0")
    
    print("\nChecking CUDA support:")
    check_cuda_toolkit()
    check_pytorch_cuda()
    
    print("\nChecking ONNX Runtime providers:")
    check_onnx_provider("CPUExecutionProvider")
    check_onnx_provider("CUDAExecutionProvider")
    
    # Print summary of requirements
    print("\nRequirements Summary:")
    print("1. For CPU-only use: Standard requirements are sufficient")
    print("2. For GPU acceleration: Ensure CUDA providers are available")
    print("   - If missing, check requirements.txt for GPU-specific installations")
    
    print("\n" + "=" * 60)
    
if __name__ == "__main__":
    main()
