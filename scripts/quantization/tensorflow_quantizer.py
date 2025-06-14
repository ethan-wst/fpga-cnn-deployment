#!/usr/bin/env python3
"""
TensorFlow model quantization with Vitis AI.

This module provides implementations for quantizing TensorFlow models using the
Vitis AI quantization tools.
"""

import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Configure logging
logger = logging.getLogger("vitis_tensorflow_quant")

def load_tensorflow_model(model_path):
    """Load a TensorFlow model from path"""
    try:
        import tensorflow as tf
        
        # Check if the path is to a valid TF model format
        valid_extensions = ['.pb', '.h5', '.keras', '']  # Empty for SavedModel directory
        valid_format = any(model_path.endswith(ext) for ext in valid_extensions) or os.path.isdir(model_path)
        
        if not valid_format:
            logger.error(f"Unsupported TensorFlow model format: {model_path}")
            return None
        
        # Load the model based on format
        if model_path.endswith('.h5') or model_path.endswith('.keras'):
            model = tf.keras.models.load_model(model_path)
        elif model_path.endswith('.pb'):
            # This is a frozen graph, need special handling
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            
            # We need to know the input and output nodes
            # For now, we'll just create a placeholder since this requires model-specific info
            logger.warning("Frozen .pb models require input/output node names. "
                         "Please use a SavedModel or Keras model if possible.")
            model = graph_def  # Just return the graph_def for now
        else:
            # Assume SavedModel directory
            model = tf.saved_model.load(model_path)
        
        logger.info(f"Successfully loaded TensorFlow model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load TensorFlow model: {str(e)}")
        return None

def prepare_calibration_data(calib_dataset, batch_size=32, transform=None):
    """Prepare calibration data for TensorFlow model quantization"""
    from PIL import Image
    from utils.image_utils import get_imagenet_transforms
    import numpy as np
    import tensorflow as tf
    
    # Process images in batches
    batches = []
    batch = []
    
    for img_path in tqdm(calib_dataset, desc="Preparing calibration data"):
        try:
            # Use TensorFlow's image processing for consistency
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32) / 255.0
            
            # Apply normalization similar to ImageNet
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            batch.append(img.numpy())
            
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
    Quantize a TensorFlow model using Vitis AI.
    
    Args:
        model: TensorFlow model to quantize
        calib_data: List of numpy arrays with calibration data
        output_dir: Directory to save the quantized model
        options: Quantization options
    
    Returns:
        bool: Success status
    """
    try:
        # Try importing Vitis AI TF modules
        try:
            from tensorflow_nndct.apis import vai_q_tensorflow
        except ImportError:
            logger.error("tensorflow_nndct not found. Make sure Vitis AI TensorFlow tools are installed.")
            return False
        
        import tensorflow as tf
        
        # For Keras models, we can use the quantization-aware training API
        if isinstance(model, tf.keras.Model):
            # Convert to quantization-aware model
            quantizer = vai_q_tensorflow.VaiQTensorflow(
                model=model,
                quantize_strategy='post_training_static'
            )
            
            # Run calibration
            logger.info("Starting calibration...")
            for i, batch in enumerate(tqdm(calib_data, desc="Calibration")):
                quantizer.run_calibration(batch)
            
            # Get quantized model
            quantized_model = quantizer.get_quantized_model()
            
            # Save the model
            quantized_model_path = os.path.join(output_dir, "quantized_model")
            tf.saved_model.save(quantized_model, quantized_model_path)
            
            # Export to DPU model if needed
            if options.get('export_dpu', True):
                dpu_model_path = os.path.join(output_dir, "dpu_model")
                vai_q_tensorflow.export_vai_model(
                    quantizer=quantizer,
                    output_dir=dpu_model_path
                )
                logger.info(f"Exported DPU model to {dpu_model_path}")
            
            logger.info(f"Quantized TensorFlow model saved to {quantized_model_path}")
            return True
        else:
            # For non-Keras models, we need different handling
            logger.error("Currently only Keras models are supported for TensorFlow quantization")
            return False
            
    except Exception as e:
        logger.error(f"TensorFlow quantization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def quantize_tensorflow_model(model_path, output_dir, calib_dataset, options):
    """Main entry point for TensorFlow model quantization"""
    # 1. Load the TensorFlow model
    model = load_tensorflow_model(model_path)
    if model is None:
        return False
    
    # 2. Prepare the calibration data
    calib_data = prepare_calibration_data(calib_dataset, batch_size=options['batch_size'])
    
    # 3. Quantize the model
    success = quantize_model(model, calib_data, output_dir, options)
    
    return success
