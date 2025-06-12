"""
File utility functions for handling paths, model files, and data.
"""

import os
import glob

# Project base directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Key directories
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')

# Model directories
ORIGINAL_MODELS_DIR = os.path.join(MODELS_DIR, 'original')
ONNX_MODELS_DIR = os.path.join(MODELS_DIR, 'onnx')
ONNX_STANDARD_MODELS_DIR = os.path.join(ONNX_MODELS_DIR, 'standard')
ONNX_QUANTIZED_MODELS_DIR = os.path.join(ONNX_MODELS_DIR, 'quantized')
QUANTIZED_MODELS_DIR = os.path.join(MODELS_DIR, 'quantized')
COMPILED_MODELS_DIR = os.path.join(MODELS_DIR, 'compiled')

# Data directories
IMAGENET_DIR = os.path.join(DATA_DIR, 'imagenet')
IMAGENET_CAL_DIR = os.path.join(IMAGENET_DIR, 'cal_subset')
IMAGENET_VAL_DIR = os.path.join(IMAGENET_DIR, 'val_subset')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
BENCHMARK_RESULTS_DIR = os.path.join(RESULTS_DIR, 'benchmark')
QUANT_RESULTS_DIR = os.path.join(RESULTS_DIR, 'quantization')

def ensure_directory(directory_path):
    """
    Ensure a directory exists, creating it if necessary.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def get_onnx_model_path(model_name, quantized=False):
    """
    Get the path to an ONNX model.
    
    Args:
        model_name (str): Name of the model (e.g., 'mobilenetv2-12')
        quantized (bool): Whether to return the quantized version
        
    Returns:
        str: Path to the model file
    """
    if quantized:
        # Check if model name already includes int8 suffix
        if 'int8' not in model_name:
            model_name = f"{model_name}-int8"
        return os.path.join(ONNX_QUANTIZED_MODELS_DIR, f"{model_name}.onnx")
    else:
        return os.path.join(ONNX_STANDARD_MODELS_DIR, f"{model_name}.onnx")

def list_available_models(model_type='onnx', quantized=False):
    """
    List all available models of a specific type.
    
    Args:
        model_type (str): Type of models ('onnx', 'pytorch', 'compiled')
        quantized (bool): Whether to list quantized models
        
    Returns:
        list: List of available model filenames
    """
    if model_type == 'onnx':
        if quantized:
            directory = ONNX_QUANTIZED_MODELS_DIR
        else:
            directory = ONNX_STANDARD_MODELS_DIR
    elif model_type == 'pytorch':
        directory = ORIGINAL_MODELS_DIR
    elif model_type == 'compiled':
        directory = COMPILED_MODELS_DIR
    else:
        return []
        
    # Get all files with appropriate extension
    if model_type == 'onnx':
        pattern = os.path.join(directory, "*.onnx")
    elif model_type == 'compiled':
        pattern = os.path.join(directory, "*.xmodel")
    else:  # pytorch
        pattern = os.path.join(directory, "*.pth")
        
    return [os.path.basename(f) for f in glob.glob(pattern)]
