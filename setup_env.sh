#!/usr/bin/env bash
# setup_env.sh - Environment setup script for FPGA-CNN-Deployment

# Add ~/.local/bin to PATH if it's not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "Adding $HOME/.local/bin to PATH..."
    export PATH="$HOME/.local/bin:$PATH"
    
    # Also add to shell configuration file
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "PATH=\"\$HOME/.local/bin:\$PATH\"" "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            echo "Added ~/.local/bin to your .bashrc file"
        fi
    elif [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "PATH=\"\$HOME/.local/bin:\$PATH\"" "$HOME/.zshrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
            echo "Added ~/.local/bin to your .zshrc file"
        fi
    else
        echo "Could not find .bashrc or .zshrc. Please manually add the following line to your shell config file:"
        echo 'export PATH="$HOME/.local/bin:$PATH"'
    fi
fi

# Check if we're running in a Vitis AI Docker container
if [ -f "/.dockerenv" ] && command -v conda &> /dev/null; then
    # Try to activate the Vitis AI conda environment if it exists
    if conda env list | grep -q "vitis-ai"; then
        echo "Activating Vitis AI conda environment..."
        conda activate vitis-ai
    elif conda env list | grep -q "vitis-ai-pytorch"; then
        echo "Activating Vitis AI PyTorch conda environment..."
        conda activate vitis-ai-pytorch
    else
        echo "Vitis AI conda environments not found. Please activate the appropriate environment manually."
    fi
fi

# Check if key packages can be imported
echo "Checking key Python packages..."
python3 -c "
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
except ImportError:
    print('✗ PyTorch not found')

try:
    import onnx
    print(f'✓ ONNX {onnx.__version__}')
except ImportError:
    print('✗ ONNX not found')

try:
    import tensorflow as tf
    print(f'✓ TensorFlow {tf.__version__}')
except ImportError:
    print('✗ TensorFlow not found')

try:
    import pytorch_nndct
    print(f'✓ Vitis AI PyTorch integration')
except ImportError:
    print('✗ Vitis AI PyTorch integration not found')
"

# Print success message
echo ""
echo "Environment setup complete. You can now run the FPGA-CNN-Deployment scripts."
echo "To ensure these changes take effect in your current shell, please run:"
echo "source setup_env.sh"
