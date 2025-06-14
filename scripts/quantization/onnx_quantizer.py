#!/usr/bin/env python3
"""
ONNX model quantization with Vitis AI.

This module provides implementations for quantizing ONNX models using the
Vitis AI quantization tools.
"""

import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Configure logging
logger = logging.getLogger("vitis_onnx_quant")

def repair_onnx_model(model):
    """
    Repair common issues in ONNX models that cause problems with quantization
    
    Args:
        model: The ONNX model to repair
        
    Returns:
        The repaired ONNX model
    """
    import onnx
    import copy
    
    try:
        logger.info("Repairing ONNX model for compatibility with quantization tools...")
        repaired_model = copy.deepcopy(model)
        
        # Get all graph inputs
        input_names = [input.name for input in repaired_model.graph.input]
        
        # Get all initializers
        init_names = [init.name for init in repaired_model.graph.initializer]
        
        # Find initializers that are not in graph inputs
        missing_inputs = []
        for name in init_names:
            if name not in input_names:
                missing_inputs.append(name)
        
        logger.info(f"Found {len(missing_inputs)} initializers that are not in graph inputs")
        
        # Remove problematic initializers from the list that must be provided as inputs
        value_info_to_remove = []
        for input_value in repaired_model.graph.input:
            if input_value.name in init_names and input_value.name not in missing_inputs:
                value_info_to_remove.append(input_value)
        
        # Remove the inputs that have initializers
        for value_info in value_info_to_remove:
            repaired_model.graph.input.remove(value_info)
            logger.debug(f"Removed {value_info.name} from graph inputs")
        
        # Check the model
        onnx.checker.check_model(repaired_model)
        logger.info("ONNX model successfully repaired")
        
        return repaired_model
    except Exception as e:
        logger.error(f"Failed to repair ONNX model: {str(e)}")
        # Return original model on error
        return model

def load_onnx_model(model_path):
    """Load an ONNX model from path"""
    try:
        import onnx
        
        # Check if the path is to an .onnx file
        if not model_path.endswith('.onnx'):
            logger.error(f"Unsupported ONNX model format: {model_path}")
            return None
        
        # Load the model
        model = onnx.load(model_path)
        
        try:
            # Basic validation - may fail with optimized models
            onnx.checker.check_model(model)
            logger.info(f"Successfully loaded ONNX model from {model_path}")
            
            # Try to repair the model for better compatibility with quantization
            model = repair_onnx_model(model)
            return model
        except Exception as inner_e:
            logger.warning(f"Model validation failed: {str(inner_e)}. Attempting repair...")
            model = repair_onnx_model(model)
            
            # Try validation again after repair
            try:
                onnx.checker.check_model(model)
                logger.info("Model validation successful after repair")
                return model
            except Exception as final_e:
                logger.error(f"Model still invalid after repair: {str(final_e)}")
                return None
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {str(e)}")
        return None

def prepare_calibration_data(calib_dataset, batch_size=32, transform=None):
    """Prepare calibration data for ONNX model quantization"""
    from PIL import Image
    from utils.image_utils import get_imagenet_transforms
    import numpy as np
    
    # Use default transform if none provided
    if transform is None:
        transform = get_imagenet_transforms()
    
    # Process images in batches
    batches = []
    batch = []
    
    for img_path in tqdm(calib_dataset, desc="Preparing calibration data"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch.append(img_tensor.numpy())
            
            if len(batch) == batch_size:
                batches.append(np.stack(batch))
                batch = []
        except Exception as e:
            logger.warning(f"Error processing image {img_path}: {str(e)}")
    
    # Add any remaining images
    if batch:
        batches.append(np.stack(batch))
    
    logger.info(f"Prepared {len(batches)} calibration batches")
    return batches

def quantize_model(model, calib_data, output_dir, options):
    """
    Quantize an ONNX model using Vitis AI.
    
    Args:
        model: ONNX model to quantize
        calib_data: List of numpy arrays with calibration data
        output_dir: Directory to save the quantized model
        options: Quantization options
    
    Returns:
        bool: Success status
    """
    try:
        # Import vai_q_onnx
        try:
            from vai_q_onnx.vai_q_onnx import vai_q_onnx
        except ImportError:
            logger.warning("vai_q_onnx not found. Falling back to ONNX Runtime quantization.")
            return quantize_model_onnxruntime(model, calib_data, output_dir, options)
        
        # Save model to temporary file if it's an in-memory model
        if isinstance(model, bytes) or not os.path.exists(model):
            import onnx
            temp_model_path = os.path.join(output_dir, "temp_model.onnx")
            onnx.save(model, temp_model_path)
            model_path = temp_model_path
        else:
            model_path = model
        
        # Create a temporary directory for calibration data
        calib_dir = os.path.join(output_dir, "calib_data")
        os.makedirs(calib_dir, exist_ok=True)
        
        # Save calibration data
        logger.info("Saving calibration data...")
        for i, batch in enumerate(calib_data):
            np.save(os.path.join(calib_dir, f"calib_batch_{i}.npy"), batch)
        
        # Create vai_q_onnx config
        bit_width = 8 if options['precision'].value == 'int8' else None
        
        # Set up quantizer
        output_model_path = os.path.join(output_dir, "quantized_model.onnx")
        
        # Run quantization
        logger.info("Starting quantization...")
        vai_q_onnx(
            model_path=model_path,
            calib_data=calib_dir,
            output_model=output_model_path,
            quant_method="method1",
            bitwidth=bit_width
        )
        
        logger.info(f"Quantized ONNX model saved to {output_model_path}")
        
        # Clean up temporary files
        if os.path.exists(calib_dir):
            import shutil
            shutil.rmtree(calib_dir)
        
        return True
    except Exception as e:
        logger.error(f"ONNX quantization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def quantize_onnx_model(model_path, output_dir, calib_dataset, options):
    """Main entry point for ONNX model quantization"""
    # 1. Load the ONNX model
    model = load_onnx_model(model_path)
    if model is None:
        logger.error(f"Failed to load ONNX model from {model_path}")
        return False
    
    # 2. Prepare the calibration data
    try:
        calib_data = prepare_calibration_data(calib_dataset, batch_size=options['batch_size'])
        if not calib_data:
            logger.error("Failed to prepare calibration data")
            return False
    except Exception as e:
        logger.error(f"Error preparing calibration data: {str(e)}")
        return False
    
    # 3. Quantize the model
    success = quantize_model(model, calib_data, output_dir, options)

    return success
