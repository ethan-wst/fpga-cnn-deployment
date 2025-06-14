#!/usr/bin/env python3
"""
PyTorch model quantization with Vitis AI.

This module provides implementations for quantizing PyTorch models using the
Vitis AI quantization tools.
"""

import os
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Configure logging
logger = logging.getLogger("vitis_pytorch_quant")

def load_pytorch_model(model_path, device='cpu'):
    """Load a PyTorch model from path"""
    try:
        # Check if the path is to a .pth or .pt file
        if not model_path.endswith(('.pth', '.pt')):
            logger.error(f"Unsupported PyTorch model format: {model_path}")
            return None
        
        model = torch.load(model_path, map_location=device)
        
        # Handle both state_dict and direct model saves
        if isinstance(model, dict):
            # Assuming it's a state dict, we need the model class
            logger.warning("The file contains a state dict rather than a model. "
                        "Please provide a full PyTorch model or specify the model class.")
            return None
        
        model.eval()
        logger.info(f"Successfully loaded PyTorch model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {str(e)}")
        return None

def prepare_calibration_loader(calib_dataset, batch_size=32, transform=None):
    """Create a data loader for calibration images"""
    from torch.utils.data import DataLoader, Dataset
    from utils.image_utils import get_imagenet_transforms
    from PIL import Image
    
    class CalibrationDataset(Dataset):
        def __init__(self, image_paths, transform=None):
            self.image_paths = image_paths
            self.transform = transform or get_imagenet_transforms()
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
    
    # Create dataset and loader
    dataset = CalibrationDataset(calib_dataset, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return loader

def quantize_model(model, calib_loader, output_dir, options):
    """
    Quantize a PyTorch model using Vitis AI.
    
    Args:
        model: PyTorch model to quantize
        calib_loader: DataLoader with calibration data
        output_dir: Directory to save the quantized model
        options: Quantization options
    
    Returns:
        bool: Success status
    """
    try:
        from pytorch_nndct.apis import torch_quantizer, dump_xmodel
        
        # Set quantization parameters based on options
        quant_mode = "calib" # Start with calibration mode
        bitwidth = 8 if options['precision'].value == 'int8' else None
        
        # Create sample input for the model
        sample_batch = next(iter(calib_loader))
        input_shape = sample_batch.shape
        
        # Configure quantizer
        quantizer = torch_quantizer(
            quant_mode=quant_mode,
            module=model,
            input_args=(sample_batch,),
            bitwidth=bitwidth,
            device=torch.device('cpu')  # Vitis AI requires CPU for quantization
        )
        
        # Get quantized model
        quant_model = quantizer.quant_model
        
        # Calibrate using the calibration dataset
        logger.info("Starting calibration...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(calib_loader, desc="Calibration")):
                output = quant_model(batch)
        
        # Switch to test mode and evaluate quantization
        quantizer.quant_mode = "test"
        logger.info("Testing quantized model...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(calib_loader, desc="Testing", total=min(5, len(calib_loader)))):
                if batch_idx >= 5:  # Test with a few batches only
                    break
                output = quant_model(batch)
        
        # Export the quantized model
        deploy_check = True
        quantizer.export_quant_config()
        
        # Save quantized model to .xmodel format for deployment
        if deploy_check:
            output_xmodel_path = os.path.join(output_dir, "deploy.xmodel")
            dump_xmodel(quantizer=quantizer, output_dir=output_dir, deploy_check=deploy_check)
            logger.info(f"Exported deploy model to {output_xmodel_path}")
        
        # Also save the PyTorch quantized model for reference
        output_pytorch_path = os.path.join(output_dir, "quantized_model.pth")
        torch.save(quant_model.state_dict(), output_pytorch_path)
        logger.info(f"Saved quantized PyTorch model to {output_pytorch_path}")
        
        return True
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def quantize_pytorch_model(model_path, output_dir, calib_dataset, options):
    """Main entry point for PyTorch model quantization"""
    # 1. Load the PyTorch model
    model = load_pytorch_model(model_path)
    if model is None:
        return False
    
    # 2. Prepare the calibration data loader
    calib_loader = prepare_calibration_loader(calib_dataset, batch_size=options['batch_size'])
    
    # 3. Quantize the model
    success = quantize_model(model, calib_loader, output_dir, options)
    
    return success
