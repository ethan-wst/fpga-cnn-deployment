#!/usr/bin/env python3
"""
ONNX model optimization with Vitis AI.

This module provides implementations for optimizing ONNX models using the
Vitis AI optimization tools.
"""

import os
import sys
import logging
import numpy as np
from tqdm import tqdm

# Configure logging
logger = logging.getLogger("vitis_onnx_optimize")

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
        
        # Basic validation
        onnx.checker.check_model(model)
        
        logger.info(f"Successfully loaded ONNX model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {str(e)}")
        return None

def optimize_model_graph(model, output_dir, options):
    """Apply graph optimizations to the ONNX model"""
    import onnx
    try:
        import onnxruntime as ort
        logger.info("Using ONNX Runtime for graph optimization")

        # Create temporary file for input model
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_in:
            temp_in_path = temp_in.name
            onnx.save(model, temp_in_path)

        # Configure session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = os.path.join(output_dir, "graph_optimized_model.onnx")

        # Create session (this will optimize and save the model)
        session = ort.InferenceSession(temp_in_path, sess_options)
        logger.info(f"Graph optimized model saved to {sess_options.optimized_model_filepath}")

        # Load the optimized model
        optimized_model = onnx.load(sess_options.optimized_model_filepath)

        # Clean up temporary file
        os.unlink(temp_in_path)
        has_optimizer = True
    except (ImportError, Exception) as e:
        logger.warning(f"ONNX Runtime optimization failed: {str(e)}")
        logger.warning("Using original model without graph optimizations")
        optimized_model = model
        has_optimizer = False

    # Save optimized model (if not already saved by ONNX Runtime)
    if not has_optimizer:
        output_model_path = os.path.join(output_dir, "graph_optimized_model.onnx")
        if not os.path.exists(output_model_path):
            onnx.save(optimized_model, output_model_path)
            logger.info(f"Graph optimized model saved to {output_model_path}")

    return optimized_model

def optimize_model_shape_inference(model, output_dir, options):
    """Apply shape inference to the ONNX model"""
    try:
        import onnx
        from onnx import shape_inference
        
        # Apply shape inference
        logger.info("Applying ONNX shape inference...")
        inferred_model = shape_inference.infer_shapes(model)
        
        # Save shape-inferred model
        output_model_path = os.path.join(output_dir, "shape_inferred_model.onnx")
        onnx.save(inferred_model, output_model_path)
        logger.info(f"Shape-inferred model saved to {output_model_path}")
        
        return inferred_model
    except Exception as e:
        logger.error(f"ONNX shape inference failed: {str(e)}")
        return model  # Return original model on failure


def optimize_onnx_model(model_path, output_dir, options):
    """Main entry point for ONNX model optimization"""
    # 1. Load the ONNX model
    model = load_onnx_model(model_path)
    if model is None:
        return False
    
    # 2. Apply optimizations based on options
    model = optimize_model_graph(model, output_dir, options)
    model = optimize_model_shape_inference(model, output_dir, options)
    
    # 4. Export final optimized model
    try:
        import onnx
        
        # Final optimized model path
        final_model_path = os.path.join(output_dir, "final_optimized_model.onnx")
        onnx.save(model, final_model_path)
        
        logger.info(f"Final optimized ONNX model saved to {final_model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export final optimized model: {str(e)}")
        return False
