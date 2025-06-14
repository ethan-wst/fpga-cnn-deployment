#!/usr/bin/env python3
"""
Script to download pretrained PyTorch models and save them to the models/pytorch folder
Models: ResNet-18, ResNet-34, ResNet-50, MobileNetV2, MobileNetV3 Small, MobileNetV3 Large
"""

import os
import sys
import torch
import torchvision.models as models
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.file_utils import ensure_directory as ensure_dir

def download_model(model_name, model_fn, save_dir):
    """
    Download a pretrained model and save it to the specified directory
    
    Args:
        model_name (str): Name of the model for saving
        model_fn (callable): Function to call to get the model
        save_dir (Path): Directory to save the model to
    """
    print(f"Downloading {model_name}...")
    try:
        # Use the modern approach with weights parameter when available
        try:
            # Import the appropriate weights
            if 'resnet18' in model_name:
                from torchvision.models import ResNet18_Weights
                model = model_fn(weights=ResNet18_Weights.DEFAULT)
            elif 'resnet34' in model_name:
                from torchvision.models import ResNet34_Weights
                model = model_fn(weights=ResNet34_Weights.DEFAULT)
            elif 'resnet50' in model_name:
                from torchvision.models import ResNet50_Weights
                model = model_fn(weights=ResNet50_Weights.DEFAULT)
            elif 'mobilenet_v2' in model_name:
                from torchvision.models import MobileNet_V2_Weights
                model = model_fn(weights=MobileNet_V2_Weights.DEFAULT)
            elif 'mobilenet_v3_large' in model_name:
                from torchvision.models import MobileNet_V3_Large_Weights
                model = model_fn(weights=MobileNet_V3_Large_Weights.DEFAULT)
            elif 'mobilenet_v3_small' in model_name:
                from torchvision.models import MobileNet_V3_Small_Weights
                model = model_fn(weights=MobileNet_V3_Small_Weights.DEFAULT)
            else:
                # Fallback for models that don't match the patterns above
                model = model_fn(pretrained=True)
        except (ImportError, AttributeError):
            # Fallback to the legacy approach if the modern one fails
            model = model_fn(pretrained=True)
        
        # Save the full model instead of just the state dict
        save_path = save_dir / f"{model_name}.pth"
        torch.save(model, save_path)
        print(f"✅ Saved full {model_name} model to {save_path}")
        
        return True
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def main():
    # Setup directory to save models
    project_root = Path(__file__).parent.parent.parent
    save_dir = project_root / "models" / "pytorch"
    ensure_dir(save_dir)
    
    # Define models to download
    models_to_download = [
        ("resnet18", models.resnet18),
        ("resnet34", models.resnet34),
        ("resnet50", models.resnet50),
        ("mobilenet_v2", models.mobilenet_v2),
        ("mobilenet_v3_small", models.mobilenet_v3_small),
        ("mobilenet_v3_large", models.mobilenet_v3_large),
    ]
    
    # Download all models
    successful = 0
    failed = 0
    
    print(f"Downloading models to {save_dir}...")
    for model_name, model_fn in models_to_download:
        if download_model(model_name, model_fn, save_dir):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*50)
    print(f"Download Summary: {successful} successful, {failed} failed")
    if successful == len(models_to_download):
        print("All models downloaded successfully!")
    print("="*50)

if __name__ == "__main__":
    main()