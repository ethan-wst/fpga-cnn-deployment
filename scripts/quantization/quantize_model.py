#!/usr/bin/env python3
"""
Model quantization utility for Vitis AI.

This script provides a unified interface for quantizing PyTorch, ONNX, and TensorFlow
models using the Vitis AI quantization tools for optimal deployment on Xilinx FPGA hardware.

Basic usage:
    python quantize_model.py --model <model_path> --format <pytorch|onnx|tensorflow> --calib_dataset <path> [options]
"""

import os
# Suppress CUDA initialization messages for CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import sys
import logging
import glob
from enum import Enum
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project utilities
from utils.file_utils import ensure_directory, IMAGENET_CAL_DIR, QUANTIZED_MODELS_DIR
import utils.file_utils as file_utils
from utils.image_utils import preprocess_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vitis_quantize")

class ModelFormat(Enum):
    PYTORCH = 'pytorch'

class QuantPrecision(Enum):
    INT8 = 'int8'
    MIXED = 'mixed'

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

def load_calibration_dataset(calib_dir, batch_size=32, max_samples=1000, random_selection=True):
    """Load calibration dataset for quantization
    
    Args:
        calib_dir: Directory containing calibration images
        batch_size: Batch size for calibration
        max_samples: Maximum number of samples to use (0 = no limit)
        random_selection: If True, randomly select max_samples images
        
    Returns:
        List of image file paths to use for calibration
    """
    if not os.path.exists(calib_dir):
        logger.error(f"Calibration directory {calib_dir} does not exist")
        return None
    
    # Check for val_subset directory
    val_subset_dir = os.path.join(calib_dir, 'val_subset')
    if os.path.exists(val_subset_dir):
        logger.info(f"Using validation subset directory: {val_subset_dir}")
        calib_dir = val_subset_dir
    
    logger.info(f"Loading calibration data from {calib_dir}")
    image_files = glob.glob(os.path.join(calib_dir, '**', '*.JPEG'), recursive=True)
    image_files += glob.glob(os.path.join(calib_dir, '**', '*.jpg'), recursive=True)
    image_files += glob.glob(os.path.join(calib_dir, '**', '*.jpeg'), recursive=True)
    image_files += glob.glob(os.path.join(calib_dir, '**', '*.png'), recursive=True)
    
    if not image_files:
        logger.error(f"No image files found in {calib_dir}")
        return None
    
    # Limit the number of samples if needed
    if max_samples > 0:
        if random_selection:
            import random
            logger.info(f"Randomly selecting {max_samples} images from {len(image_files)} available images")
            image_files = random.sample(image_files, min(max_samples, len(image_files)))
        else:
            logger.info(f"Taking first {max_samples} images from {len(image_files)} available images")
            image_files = image_files[:max_samples]
    
    logger.info(f"Selected {len(image_files)} calibration images")
    return image_files

# Import model-specific quantizer
# Using absolute imports instead of relative imports to fix execution as main script
from scripts.quantization.pytorch_quantizer import quantize_pytorch_model

def get_output_dir(model_path, output_base_dir):
    """Generate appropriate output directory based on model path and format"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join(output_base_dir, model_name)
    ensure_directory(output_dir)
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Quantize models for Vitis AI deployment")
    parser.add_argument('--model', required=True, help='Path to the model file')
    parser.add_argument('--format', required=True, choices=['pytorch'], 
                        help='Format of the input model')
    parser.add_argument('--calib_dataset', default=IMAGENET_CAL_DIR, 
                        help=f'Path to calibration dataset (default: {IMAGENET_CAL_DIR})')
    parser.add_argument('--output_dir', default=None, 
                        help='Output directory for the quantized model')
    parser.add_argument('--precision', default='int8', choices=['int8', 'mixed'],
                        help='Quantization precision (int8 or mixed precision)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for calibration')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of calibration samples to use (0 = no limit)')
    parser.add_argument('--random_selection', action='store_true', default=True,
                        help='Randomly select calibration samples (default: True)')
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
    
    # Load calibration dataset
    calib_dataset = load_calibration_dataset(args.calib_dataset, 
                                        batch_size=args.batch_size,
                                        max_samples=args.max_samples,
                                        random_selection=args.random_selection)
    
    if calib_dataset is None:
        logger.error("Failed to load calibration dataset")
        sys.exit(1)
    
    # Create output directory if not specified
    if args.output_dir is None:
        output_base_dir = QUANTIZED_MODELS_DIR
        args.output_dir = get_output_dir(args.model, output_base_dir)
    
    ensure_directory(args.output_dir)
    
    # Create options dictionary
    options = {
        'precision': QuantPrecision(args.precision),
        'batch_size': args.batch_size,
    }
    
    # Dispatch based on model format
    model_format = ModelFormat(args.format)
    
    # Only PyTorch models are supported
    success = quantize_pytorch_model(args.model, args.output_dir, calib_dataset, options)
    
    if success:
        logger.info(f"Model quantization completed successfully")
        return 0
    else:
        logger.error(f"Model quantization failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
