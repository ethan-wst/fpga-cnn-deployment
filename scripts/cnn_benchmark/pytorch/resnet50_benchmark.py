# resnet50_benchmark.py
# Benchmark ResNet50 (unquantized and quantized) using PyTorch

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models.quantization import ResNet50_QuantizedWeights
import argparse
import time
import csv
import os
from PIL import Image
from torchvision import transforms
import json

parser = argparse.ArgumentParser(description='Benchmark ResNet50 (unquantized and quantized)')
parser.add_argument('--model', type=str, default='resnet50', 
                    choices=['resnet50', 'resnet50_quant'], 
                    help='Model name')
parser.add_argument('--device', type=str, default='cpu', 
                    choices=['cpu', 'cuda'], 
                    help='Device to run the benchmark on')
parser.add_argument('--imagenet_dir', type=str, 
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../imagenet/imagenet_subset')), 
                    help='Path to ImageNet_SubSet directory')
args = parser.parse_args()

if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on: {device}")

if args.model == 'resnet50_quant':
    weights = ResNet50_QuantizedWeights.DEFAULT
    model = models.quantization.resnet50(quantize=True, weights=weights)
elif args.model == 'resnet50':
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
else:
    raise ValueError(f"Unknown model: {args.model}")

input_size = weights.transforms().crop_size[0] if hasattr(weights, 'transforms') and hasattr(weights.transforms(), 'crop_size') else 224
model.eval()
model.to(device)

with open(os.path.join(os.path.dirname(__file__), '../../imagenet/imagenet_class_index.json'), 'r') as f:
    class_idx = json.load(f)
wnid_to_model_idx = {v[0]: int(k) for k, v in class_idx.items()}

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

sampled = list(zip(image_paths, gt_wnids))
print(f"Running inference on all {len(sampled)} available images")

preprocess = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

times = []
correct = 0
top5_correct = 0
total = 0

with torch.no_grad():
    for img_path, wnid in sampled:
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            start = time.time()
            output = model(input_tensor)
            end = time.time()
            times.append((end - start))
            pred = output.argmax(dim=1).item()
            gt_idx = wnid_to_model_idx.get(wnid, -1)
            if pred == gt_idx:
                correct += 1
            top5 = output.topk(5, dim=1).indices.squeeze(0).tolist()
            if gt_idx in top5:
                top5_correct += 1
            total += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

avg_time = sum(times) / len(times) * 1000 if times else 0
accuracy = correct / total * 100 if total > 0 else 0
top5_accuracy = top5_correct / total * 100 if total > 0 else 0

is_quantized = 'quant' in args.model
param_size = 0
buffer_size = 0
if is_quantized:
    for name, param in model.state_dict().items():
        if isinstance(param, torch.Tensor):
            size_bytes = param.numel() * (1 if param.dtype in [torch.qint8, torch.quint8] else 4)
            if 'weight' in name or 'bias' in name:
                param_size += size_bytes
            else:
                buffer_size += size_bytes
else:
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
total_size_mb = (param_size + buffer_size) / 1024**2

print(f"Model: {args.model}")
print(f"Input Size: {input_size}x{input_size}")
print(f"Average Inference Time: {avg_time:.2f} ms/image over {total} images")
print(f"Top-1 Accuracy: {accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
print(f"Total size in MB: {total_size_mb:.2f}")

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../benchmark_results/pytorch_benchmark_results.csv'))
write_header = not os.path.exists(csv_path)
with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Device", "Model", "Input Size", "Avg Inference Time (ms)", "Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)"])
    writer.writerow([device, args.model, f"{input_size}x{input_size}", f"{avg_time:.2f}", f"{accuracy:.2f}", f"{top5_accuracy:.2f}", f"{total_size_mb:.2f}"])