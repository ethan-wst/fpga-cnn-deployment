"""
Image and model preprocessing utilities.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def get_imagenet_transforms():
    """
    Get standard transforms for ImageNet preprocessing.
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image_path, transform=None):
    """
    Preprocess an image for model inference.
    
    Args:
        image_path (str): Path to the image
        transform (callable, optional): Transform to apply
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if transform is None:
        transform = get_imagenet_transforms()
        
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def get_imagenet_classes(json_path):
    """
    Load ImageNet class labels from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file with class mappings
        
    Returns:
        dict: ImageNet class index to label mapping
    """
    import json
    with open(json_path) as f:
        class_idx = json.load(f)
        
    idx_to_class = {int(k): v[1] for k, v in class_idx.items()}
    return idx_to_class
