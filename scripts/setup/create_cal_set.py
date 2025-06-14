#!/usr/bin/env python3
"""
Create a calibration dataset by sampling one image from each class in the ImageNet validation set.

This script creates a calibration dataset by taking one random image from each class
folder in the validation subset and copying it to a calibration folder. This is useful
for quantization processes that require a small representative dataset.
"""

import os
import sys
import random
import shutil
from pathlib import Path
import argparse
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.file_utils import ensure_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("create_cal_set")

def create_calibration_set(val_dir, cal_dir, num_images_per_class=1):
    """
    Create a calibration dataset by sampling images from each class in the validation set.
    
    Args:
        val_dir: Path to the ImageNet validation subset directory
        cal_dir: Path to the output calibration directory
        num_images_per_class: Number of images to sample from each class (default: 1)
    """
    # Ensure calibration directory exists
    os.makedirs(cal_dir, exist_ok=True)
    
    # Get list of class directories
    class_dirs = [d for d in Path(val_dir).iterdir() if d.is_dir()]
    logger.info(f"Found {len(class_dirs)} class directories in validation set")
    
    # Process each class directory
    total_copied = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all image files in the class directory
        image_files = list(class_dir.glob("*.JPEG"))
        if not image_files:
            # Try other common extensions if .JPEG isn't found
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        
        if not image_files:
            logger.warning(f"No images found in {class_dir}")
            continue
            
        # Randomly sample images
        sampled_images = random.sample(image_files, min(num_images_per_class, len(image_files)))
        
        # Copy each sampled image to the calibration directory
        for img_file in sampled_images:
            # Create class subdirectory in cal_dir to maintain structure
            class_cal_dir = os.path.join(cal_dir, class_name)
            os.makedirs(class_cal_dir, exist_ok=True)
            
            # Destination path
            dest_path = os.path.join(class_cal_dir, img_file.name)
            
            # Copy the file
            shutil.copy2(img_file, dest_path)
            total_copied += 1
            
    logger.info(f"Successfully copied {total_copied} images to calibration set from {len(class_dirs)} classes")
    return total_copied

def main():
    parser = argparse.ArgumentParser(description="Create calibration dataset from ImageNet validation subset")
    parser.add_argument('--val_dir', default=None, 
                        help='Path to ImageNet validation subset directory')
    parser.add_argument('--cal_dir', default=None,
                        help='Path to output calibration directory')
    parser.add_argument('--num_images', type=int, default=1,
                        help='Number of images to sample from each class (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Determine directory paths
    project_root = Path(__file__).parent.parent.parent
    
    # If val_dir not specified, use the default location
    if args.val_dir is None:
        args.val_dir = project_root / "data" / "imagenet" / "val_subset"
    
    # If cal_dir not specified, use the default location
    if args.cal_dir is None:
        args.cal_dir = project_root / "data" / "imagenet" / "cal_subset"
    
    # Ensure directories are absolute paths
    val_dir = Path(args.val_dir).absolute()
    cal_dir = Path(args.cal_dir).absolute()
    
    # Check if validation directory exists
    if not val_dir.exists() or not val_dir.is_dir():
        logger.error(f"Validation directory not found: {val_dir}")
        return 1
    
    # Create calibration directory if it doesn't exist
    os.makedirs(cal_dir, exist_ok=True)
    
    logger.info(f"Creating calibration set from {val_dir}")
    logger.info(f"Output directory: {cal_dir}")
    logger.info(f"Sampling {args.num_images} image(s) per class")
    
    # Create the calibration set
    total_copied = create_calibration_set(val_dir, cal_dir, args.num_images)
    
    if total_copied > 0:
        logger.info(f"Calibration set created successfully with {total_copied} images")
        return 0
    else:
        logger.error("Failed to create calibration set")
        return 1

if __name__ == "__main__":
    sys.exit(main())
