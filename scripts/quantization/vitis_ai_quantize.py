#!/usr/bin/env python3
"""
Vitis AI Quantization Script for ONNX models

This script provides functions to quantize ONNX models using Vitis AI tools,
targeting Xilinx U50 FPGA deployment.

Usage:
  python vitis_ai_quantize.py --model [MODEL_NAME] --format [onnx/pytorch] --output [OUTPUT_PATH]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import project utilities
from utils.file_utils import (
    ensure_directory, 
    ONNX_STANDARD_MODELS_DIR,
    ONNX_QUANTIZED_MODELS_DIR, 
    IMAGENET_CAL_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vitis_ai_quantize')

def check_vitis_ai_env():
    """
    Check if script is running inside the Vitis AI Docker environment.
    
    Returns:
        bool: True if running in Vitis AI environment, False otherwise
    """
    logger.info("Checking Vitis AI environment...")
    # Check for Vitis AI environment variables or files
    if not os.path.exists('/opt/vitis_ai'):
        logger.warning("This script should be run inside the Vitis AI Docker container")
        logger.info("Run with: 'vitisai vitis_ai_quantize.py [args]'")
        return False
    
    logger.info("Vitis AI environment detected")
    # Print Vitis AI version info if available
    try:
        import vai
        logger.info(f"Vitis AI version: {vai.__version__}")
    except (ImportError, AttributeError):
        logger.info("Vitis AI package found but version info not available")
    
    return True

def get_available_models():
    """Get list of available models in the standard ONNX directory"""
    logger.info(f"Scanning for models in directory: {ONNX_STANDARD_MODELS_DIR}")
    
    if not os.path.exists(ONNX_STANDARD_MODELS_DIR):
        logger.error(f"Directory not found: {ONNX_STANDARD_MODELS_DIR}")
        return []
        
    models = []
    try:
        for file in os.listdir(ONNX_STANDARD_MODELS_DIR):
            if file.endswith('.onnx'):
                model_name = file[:-5]  # Remove .onnx extension
                models.append(model_name)
                logger.debug(f"Found model: {model_name}")
    except Exception as e:
        logger.error(f"Error scanning models directory: {e}")
    
    logger.info(f"Found {len(models)} ONNX models")
    return models

def prepare_calibration_dataset(calibration_dir=IMAGENET_CAL_DIR, batch_size=8):
    """
    Prepare calibration dataset from the ImageNet calibration subset.
    
    Args:
        calibration_dir (str): Directory containing calibration images
        batch_size (int): Batch size for calibration
        
    Returns:
        list: List of preprocessed image data for calibration
    """
    from utils.image_utils import get_imagenet_transforms
    import torch
    from torchvision import datasets
    from torch.utils.data import DataLoader
    
    logger.info(f"Preparing calibration dataset from {calibration_dir}")
    
    # Check if calibration directory exists
    if not os.path.exists(calibration_dir):
        logger.error(f"Calibration directory {calibration_dir} not found")
        return None
    
    # Load calibration data
    try:
        transform = get_imagenet_transforms()
        dataset = datasets.ImageFolder(calibration_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare calibration data (first few batches)
        calibration_data = []
        num_batches = min(50, len(dataloader))  # Limit number of calibration images
        
        logger.info(f"Loading {num_batches} batches for calibration...")
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            calibration_data.append(images.numpy())
            
        logger.info(f"Prepared {len(calibration_data)} calibration batches")
        return calibration_data
        
    except Exception as e:
        logger.error(f"Error preparing calibration dataset: {e}")
        return None

def quantize_onnx_model(model_name, output_dir=None):
    """
    Quantize an ONNX model using Vitis AI.
    
    Args:
        model_name (str): Name of the model (e.g., 'mobilenetv2-12')
        output_dir (str, optional): Directory to save the quantized model
        
    Returns:
        str: Path to the quantized model
    """
    if not check_vitis_ai_env():
        logger.error("This script must be run in Vitis AI Docker environment")
        return None
        
    # Default output directory is the project's quantized models directory
    if output_dir is None:
        output_dir = ONNX_QUANTIZED_MODELS_DIR
        
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Input model path
    input_model = os.path.join(ONNX_STANDARD_MODELS_DIR, f"{model_name}.onnx")
    if not os.path.exists(input_model):
        logger.error(f"Model {input_model} not found")
        return None
        
    # Output model path
    output_model = os.path.join(output_dir, f"{model_name}_quantized.onnx")
    
    # Import Vitis AI libraries (only available in Vitis AI environment)
    try:
        # Import Vitis AI quantization libraries
        import vai.dpuv1.rt as vai_rt
        from vai.dpuv1.tools.onnx.xmodel_transformer import xmodel_transformer
        from vai.dpuv1.tools.onnx.vai_q_onnx import vai_q_onnx
    except ImportError:
        logger.error("Failed to import Vitis AI libraries. Are you running in the Vitis AI container?")
        return None
        
    logger.info(f"Starting quantization for {model_name}")
    logger.info(f"Input model: {input_model}")
    logger.info(f"Output model: {output_model}")
    
    try:
        # Prepare calibration data using ImageNet calibration subset
        calibration_data = prepare_calibration_dataset()
        if calibration_data is None:
            logger.error("Failed to prepare calibration dataset")
            return None
            
        # Create working directory for quantization artifacts
        work_dir = os.path.join(output_dir, f"{model_name}_work")
        ensure_directory(work_dir)
        
        # Step 1: Configure the quantizer
        quant_config = {
            'input_model': input_model,
            'output_dir': work_dir,
            'target': 'U50',  # Target Xilinx U50 FPGA
            'calibration_data': calibration_data,
            'input_fn': None,  # Use default input function
            'input_shapes': None,  # Use shapes from model
        }
        
        # Step 2: Perform quantization
        logger.info("Starting Vitis AI quantization...")
        vai_q = vai_q_onnx(quant_config)
        quantized_model = vai_q.quantize()
        
        # Step 3: Compile the quantized model for the U50 target
        logger.info("Compiling quantized model for U50 target...")
        compiler_config = {
            'arch': 'U50',
            'input_model': quantized_model,
            'output_dir': output_dir,
            'output_model': os.path.basename(output_model),
        }
        
        xmodel_transformer(compiler_config)
        
        # Check if quantization was successful
        if os.path.exists(output_model):
            logger.info(f"Quantization completed successfully. Quantized model saved to: {output_model}")
            return output_model
        else:
            logger.error("Quantization failed. Output model not generated.")
            return None
            
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        return None

def main():
    import time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Quantize models using Vitis AI for Xilinx U50 FPGA')
    parser.add_argument('--model', type=str, help='Model name (without extension)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--all', action='store_true', help='Quantize all available models')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output directory for quantized models')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging (more verbose than --verbose)')
    
    logger.info("Parsing command line arguments...")
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logger.info("Verbose logging enabled")
    
    # Print environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check Vitis AI environment before proceeding
    if not check_vitis_ai_env():
        logger.error("Vitis AI environment check failed")
        sys.exit(1)
    
    if args.list:
        logger.info("Listing available models...")
        print("\n" + "="*60)
        print(" Available Models for Vitis AI Quantization")
        print("="*60)
        
        models = get_available_models()
        if models:
            for i, model in enumerate(models, 1):
                print(f"  {i:2d}. {model}")
            print("\n")
        else:
            print("\n  No ONNX models found in the standard directory")
            print(f"  Expected directory: {ONNX_STANDARD_MODELS_DIR}")
            print("\n  Please add .onnx model files to this directory first.\n")
        
        elapsed = time.time() - start_time
        print(f"Command completed in {elapsed:.2f} seconds\n")
        return
    
    if args.all:
        models = get_available_models()
        if not models:
            logger.error("No models found to quantize")
            return
            
        logger.info(f"Quantizing all {len(models)} available models")
        for model in models:
            logger.info(f"Processing model: {model}")
            quantize_onnx_model(model, args.output)
        
    elif args.model:
        quantize_onnx_model(args.model, args.output)
    else:
        parser.error("Please specify a model name with --model, use --all to quantize all models, or use --list to see available models")

if __name__ == "__main__":
    main()
