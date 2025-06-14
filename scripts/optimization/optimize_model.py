#!/usr/bin/env python3
"""
Model optimization utility for Vitis AI.

This script enables various model optimization techniques using the Vitis AI framework
to prepare models for efficient deployment on Xilinx FPGA hardware.

Basic usage:
    python optimize_model.py --model <model_path> --format <pytorch|onnx|tensorflow> [options]
"""

import os
# Suppress CUDA initialization messages for CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import sys
import logging
from enum import Enum

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project utilities
from utils.file_utils import ensure_directory, MODELS_DIR, ORIGINAL_MODELS_DIR
import utils.file_utils as file_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vitis_optimize")

class ModelFormat(Enum):
    PYTORCH = 'pytorch'
    ONNX = 'onnx'
    TENSORFLOW = 'tensorflow'

def check_vitis_ai_environment():
    """Check if Vitis AI environment is properly set up"""
    try:
        # Try importing key Vitis AI packages
        import pytorch_nndct
        logger.info("✓ Vitis AI PyTorch integration detected")
        return True
    except ImportError:
        logger.error("✗ Vitis AI environment not properly set up")
        logger.error("  Make sure you're running in the xilinx/vitis-ai-pytorch-cpu Docker container")
        logger.error("  and the Vitis AI environment is activated")
        return False

# Import model-specific optimizers
from scripts.optimization.pytorch_optimizer import optimize_pytorch_model
from scripts.optimization.onnx_optimizer import optimize_onnx_model
from scripts.optimization.tensorflow_optimizer import optimize_tensorflow_model

def get_output_dir(model_path, output_base_dir):
    """Generate appropriate output directory based on model path and format"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join(output_base_dir, 'optimized', model_name)
    ensure_directory(output_dir)
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Optimize models for Vitis AI deployment")
    parser.add_argument('--model', required=True, help='Path to the model file')
    parser.add_argument('--format', required=True, choices=['pytorch', 'onnx', 'tensorflow'], 
                        help='Format of the input model')
    parser.add_argument('--output_dir', default=None, 
                        help='Output directory for the optimized model')
    parser.add_argument('--pruning', action='store_true', 
                        help='Apply model pruning for weight reduction')
    parser.add_argument('--arch_opt', action='store_true', 
                        help='Apply architecture optimization')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if Vitis AI environment is properly set up
    if not check_vitis_ai_environment():
        logger.error("Vitis AI environment not properly set up. Please run in the Vitis AI docker container.")
        sys.exit(1)
    
    # Create output directory if not specified
    if args.output_dir is None:
        output_base_dir = MODELS_DIR
        args.output_dir = get_output_dir(args.model, output_base_dir)
    
    ensure_directory(args.output_dir)
    
    # Create options dictionary
    options = {
        'pruning': args.pruning,
        'arch_opt': args.arch_opt,
    }
    
    # Dispatch based on model format
    model_format = ModelFormat(args.format)
    
    if model_format == ModelFormat.PYTORCH:
        success = optimize_pytorch_model(args.model, args.output_dir, options)
    elif model_format == ModelFormat.ONNX:
        success = optimize_onnx_model(args.model, args.output_dir, options)
    elif model_format == ModelFormat.TENSORFLOW:
        success = optimize_tensorflow_model(args.model, args.output_dir, options)
    
    if success:
        logger.info(f"Model optimization completed successfully")
        return 0
    else:
        logger.error(f"Model optimization failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
