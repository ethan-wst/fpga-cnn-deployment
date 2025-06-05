# mobilenetv2_onnx_benchmark.py
# Benchmark MobileNetV2 ONNX models from local directory on ImageNet_SubSet

import numpy as np
import onnxruntime as ort
import argparse
import time
import csv
import os
import json
from PIL import Image
from torchvision import transforms
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Benchmark MobileNetV2 ONNX models')
parser.add_argument('--model', type=str, default='mobilenet_v2', 
                    choices=['mobilenet_v2', 'mobilenet_v2_quant'], 
                    help='Model name: mobilenet_v2, mobilenet_v2_quant')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                   help='Device to run the benchmark on')
parser.add_argument('--imagenet_dir', type=str, 
                   default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../imagenet/imagenet_subset')), 
                   help='Path to ImageNet_SubSet directory')
parser.add_argument('--models_dir', type=str, 
                   default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')), 
                   help='Directory with ONNX models')
args = parser.parse_args()

# --- Map model names to file paths ---
MODEL_FILES = {
    'mobilenet_v2': os.path.join(args.models_dir, 'mobilenetv2-12.onnx'),
    'mobilenet_v2_quant': os.path.join(args.models_dir, 'mobilenetv2-12-int8.onnx')
}

# --- Device Setup for ONNX Runtime ---
providers = ['CPUExecutionProvider']
if args.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("Using CUDA for inference")
else:
    print("Using CPU for inference")

# --- Load model from local directory ---
model_path = MODEL_FILES[args.model]
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
    
# Get model size
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

# --- Create ONNX Inference Session ---
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

# Get input details
inputs = session.get_inputs()
input_name = inputs[0].name
input_shape = inputs[0].shape
if len(input_shape) == 4:  # NCHW format
    _, _, height, width = input_shape
    if height is None or width is None:  # Handle dynamic shapes
        height, width = 224, 224
else:
    height, width = 224, 224  # Default to standard size if shape is dynamic

# --- Load WNID to model class index mapping (for accuracy) ---
with open(os.path.join(os.path.dirname(__file__), '../../imagenet/imagenet_class_index.json'), 'r') as f:
    class_idx = json.load(f)
wnid_to_model_idx = {v[0]: int(k) for k, v in class_idx.items()}

# --- Gather all image paths and their WNIDs ---
image_paths = []
gt_wnids = []
for wnid in os.listdir(args.imagenet_dir):
    wnid_dir = os.path.join(args.imagenet_dir, wnid)
    if not os.path.isdir(wnid_dir):
        continue
    for fname in os.listdir(wnid_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(wnid_dir, fname))
            gt_wnids.append(wnid)

# --- Collect Sample Images ---
sampled = list(zip(image_paths, gt_wnids))
print(f"Running inference on all {len(sampled)} available images")

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.CenterCrop((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Benchmark and Accuracy ---
times = []
correct = 0
top5_correct = 0
total = 0

# Get output name
output_name = session.get_outputs()[0].name

for img_path, wnid in sampled:
    try:
        img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).numpy()  # Convert to numpy for ONNX
        
        # Run inference
        start = time.time()
        outputs = session.run([output_name], {input_name: input_tensor})
        end = time.time()
        
        times.append((end - start))
        
        # Process outputs
        output = outputs[0]
        pred = np.argmax(output, axis=1)[0]
        gt_idx = wnid_to_model_idx.get(wnid, -1)
        
        if pred == gt_idx:
            correct += 1
        
        # Top-5 accuracy
        top5_indices = np.argsort(output[0])[-5:][::-1]
        if gt_idx in top5_indices:
            top5_correct += 1
            
        total += 1
        
        if total % 500 == 0:
            print(f"Processed {total} images...")
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

avg_time = sum(times) / len(times) * 1000 if times else 0  # ms/image
accuracy = correct / total * 100 if total > 0 else 0
top5_accuracy = top5_correct / total * 100 if total > 0 else 0

# --- Output Results ---
print("\n----- Results -----")
print(f"Model: {args.model} (Local ONNX)")
print(f"Input Size: {width}x{height}")
print(f"Average Inference Time: {avg_time:.2f} ms/image over {total} images")
print(f"Top-1 Accuracy: {accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
print(f"Model Size: {model_size_mb:.2f} MB")

# --- Data Export ---
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                      '../../benchmark_results/onnx_benchmark_results.csv'))
write_header = not os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Device", "Model", "Format", "Input Size", "Avg Inference Time (ms)", 
                       "Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)"])
    writer.writerow([args.device, args.model, "ONNX Local", f"{width}x{height}", 
                    f"{avg_time:.2f}", f"{accuracy:.2f}", f"{top5_accuracy:.2f}", f"{model_size_mb:.2f}"])