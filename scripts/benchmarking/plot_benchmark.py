import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
onnx_csv_path = os.path.join(os.path.dirname(__file__), 'onnx_benchmark_results.csv')
pytorch_csv_path = os.path.join(os.path.dirname(__file__), 'pytorch_benchmark_results.csv')

csv_paths = [
    (onnx_csv_path, 'onnx'),
    (pytorch_csv_path, 'pytorch')
]

def process_and_plot(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    with open(csv_path, 'r') as f:
        first_line = f.readline()
        if 'Model' not in first_line:
            columns = ["Device", "Model", "Format", "Input Size", "Avg Inference Time (ms)", 
                       "Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)"]
            df = pd.read_csv(csv_path, names=columns)
        else:
            df = pd.read_csv(csv_path)
    if df.columns[0].startswith('Device'):
        df.columns = [c.strip() for c in df.columns]
    if 'Device' in df.columns and 'Model' in df.columns:
        df = df.drop_duplicates(subset=["Device", "Model"], keep='last')
    else:
        df = df.drop_duplicates()
    for col in ["Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Total Size (MB)", "Avg Inference Time (ms)"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    cpu_df = df[df['Device'] == 'cpu']
    cuda_df = df[df['Device'] == 'cuda']
    
    def plot_bubble_map(device_df, device_name):
        plt.figure(figsize=(12, 8))
        x = device_df["Avg Inference Time (ms)"]
        y = device_df["Top-1 Accuracy (%)"]
        sizes = device_df["Total Size (MB)"] * 100  # scale for visibility
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(device_df['Model'].unique())))
        color_map = {name: colors[i] for i, name in enumerate(device_df['Model'].unique())}
        color_list = [color_map[m] for m in device_df['Model']]
        
        plt.scatter(x, y, s=sizes, c=color_list, alpha=0.7, edgecolors='w', linewidths=1)

        for i, row in device_df.iterrows():
            plt.text(row["Avg Inference Time (ms)"], row["Top-1 Accuracy (%)"], row["Model"], 
                     fontsize=9, ha='center', va='center')
        
        plt.xlabel("Avg Inference Time (ms)")
        plt.ylabel("Top-1 Accuracy (%)")
        plt.title(f"{label.upper()} CNN Models on {device_name.upper()}: Inference Time vs. Top-1 Accuracy (Bubble Size = Model Size MB)")

        # Set consistent y-axis range from 65% to 100%
        plt.ylim(25, 100)
        plt.xlim(0, 50)
        
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10)  # Maximize plot area
        plt.savefig(f"benchmark_bubble_map_{label}_{device_name}.png")
        plt.close()
    if not cpu_df.empty:
        plot_bubble_map(cpu_df, "cpu")
def plot_model_size_bar(onnx_csv_path, pytorch_csv_path):
    if not os.path.exists(onnx_csv_path):
        print(f"ONNX CSV file not found: {onnx_csv_path}")
        return
    df_onnx = pd.read_csv(onnx_csv_path)
    df_onnx_cpu = df_onnx[df_onnx['Device'] == 'cpu']

    # Models to ensure are included
    required_models = [
        "mobilenet_v2", "mobilenet_v2_quant",
        "mobilenet_v3_large", "mobilenet_v3_large_quant",
        "resnet18", "resnet18_quant",
        "resnet50", "resnet50_quant",
        "efficientnet_lite4", "efficientnet_lite4_quant"
    ]

    # Check which required models are missing from ONNX
    missing_models = [m for m in required_models if m not in df_onnx_cpu['Model'].values]

    # If any required models are missing, try to get them from PyTorch results
    if missing_models and os.path.exists(pytorch_csv_path):
        df_pytorch = pd.read_csv(pytorch_csv_path)
        df_pytorch_cpu = df_pytorch[df_pytorch['Device'] == 'cpu']
        # Only keep the missing models
        df_missing = df_pytorch_cpu[df_pytorch_cpu['Model'].isin(missing_models)].copy()
        # Add a 'Format' column for consistency if missing
        if 'Format' not in df_missing.columns:
            df_missing['Format'] = 'PyTorch'
        # Add to ONNX DataFrame
        df_onnx_cpu = pd.concat([df_onnx_cpu, df_missing], ignore_index=True)

    # Drop duplicate models, prefer ONNX (keep first occurrence)
    df_onnx_cpu = df_onnx_cpu.drop_duplicates(subset=['Model'], keep='first')

    # Group models by base name (remove _quant suffix for grouping)
    def base_model_name(model):
        return model.replace('_quant', '')

    df_onnx_cpu['BaseModel'] = df_onnx_cpu['Model'].apply(base_model_name)
    df_onnx_cpu['Quantized'] = df_onnx_cpu['Model'].apply(lambda x: 'Quantized' if 'quant' in x else 'Unquantized')

    # Only plot required models (and in the order specified)
    plot_bases = []
    for m in required_models:
        base = m.replace('_quant', '')
        if base not in plot_bases:
            plot_bases.append(base)

    # Prepare data for grouped bar plot
    bar_width = 0.35
    x = np.arange(len(plot_bases))
    unquant_sizes = []
    quant_sizes = []
    for base in plot_bases:
        unquant_row = df_onnx_cpu[(df_onnx_cpu['BaseModel'] == base) & (df_onnx_cpu['Quantized'] == 'Unquantized')]
        quant_row = df_onnx_cpu[(df_onnx_cpu['BaseModel'] == base) & (df_onnx_cpu['Quantized'] == 'Quantized')]
        unquant_sizes.append(unquant_row['Total Size (MB)'].values[0] if not unquant_row.empty else 0)
        quant_sizes.append(quant_row['Total Size (MB)'].values[0] if not quant_row.empty else 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - bar_width/2, unquant_sizes, bar_width, label='Unquantized', color='royalblue')
    bars2 = ax.bar(x + bar_width/2, quant_sizes, bar_width, label='Quantized', color='orange')

    ax.set_ylabel('Model Size (MB)')
    ax.set_xlabel('Model')
    ax.set_title('Model Size (MB) for ONNX/PyTorch Models (CPU)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_bases, rotation=30, ha='right')
    ax.legend()

    # Annotate bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("onnx_model_sizes_bar.png")
    plt.close()

for csv_path, label in csv_paths:
    process_and_plot(csv_path, label)

# Add this at the end to plot the ONNX model size bar plot, ensuring mobilenet_v3 and resnet18 are included
plot_model_size_bar(onnx_csv_path, pytorch_csv_path)