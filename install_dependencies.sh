#!/bin/bash
set -e

echo "======================================================================"
echo "Decipher Batch Correction - Dependency Installation"
echo "======================================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"

# Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
python3 -m venv decipher_env
echo "✓ Virtual environment created: decipher_env/"

# Activate virtual environment
source decipher_env/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"

# Install PyTorch
echo ""
echo "[3/5] Installing PyTorch..."
echo "Select installation type:"
echo "  1) CPU only (smaller, works everywhere)"
echo "  2) GPU with CUDA 11.8 (requires NVIDIA GPU)"
echo "  3) GPU with CUDA 12.1 (requires NVIDIA GPU)"
read -p "Enter choice [1-3] (default: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo "Installing PyTorch (CPU)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    2)
        echo "Installing PyTorch (CUDA 11.8)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    3)
        echo "Installing PyTorch (CUDA 12.1)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    *)
        echo "Invalid choice, installing CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
esac
echo "✓ PyTorch installed"

# Install Pyro and other dependencies
echo ""
echo "[4/5] Installing Pyro and other dependencies..."
pip install pyro-ppl numpy pandas matplotlib scikit-learn
echo "✓ Pyro and core dependencies installed"

# Optional: Install scanpy
echo ""
read -p "Install scanpy for AnnData support? [Y/n]: " install_scanpy
install_scanpy=${install_scanpy:-Y}

if [[ $install_scanpy =~ ^[Yy]$ ]]; then
    pip install scanpy
    echo "✓ scanpy installed"
fi

# Verify installation
echo ""
echo "[5/5] Verifying installation..."
python3 << 'EOF'
import sys
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.is_available()} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"  CUDA available: False (CPU only)")
except ImportError:
    print("✗ PyTorch not found")
    sys.exit(1)

try:
    import pyro
    print(f"✓ Pyro {pyro.__version__}")
except ImportError:
    print("✗ Pyro not found")
    sys.exit(1)

try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except ImportError:
    print("✗ NumPy not found")
    sys.exit(1)

try:
    import scanpy
    print(f"✓ scanpy {scanpy.__version__}")
except ImportError:
    print("  scanpy not installed (optional)")
EOF

echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source decipher_env/bin/activate"
echo "  2. Run tests: python3 test_integration.py"
echo "  3. Check quick start: cat QUICK_START.md"
echo ""
echo "To deactivate environment later: deactivate"
echo ""
