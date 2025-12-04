# Installation Guide - Ubuntu

## Required Dependencies

### Core Dependencies (Required)
- **Python 3.8+**
- **PyTorch** (≥1.12.0)
- **Pyro** (pyro-ppl ≥1.8.0)
- **NumPy** (≥1.21.0)

### Optional but Recommended
- **scanpy** - For working with AnnData objects
- **pandas** - Data manipulation
- **matplotlib** - Visualization
- **scikit-learn** - Metrics and preprocessing

---

## Installation Steps for Ubuntu

### Option 1: Using pip (Recommended)

```bash
# Update system packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create a virtual environment (recommended)
python3 -m venv decipher_env
source decipher_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR Install PyTorch (GPU version with CUDA 11.8)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Pyro
pip install pyro-ppl

# Install other dependencies
pip install numpy pandas matplotlib scikit-learn

# Install scanpy (for AnnData support)
pip install scanpy

# Optional: Install Jupyter for notebooks
pip install jupyter ipykernel
```

### Option 2: Using conda (Alternative)

```bash
# Install miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, then restart terminal

# Create conda environment
conda create -n decipher_env python=3.10
conda activate decipher_env

# Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# OR for GPU: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pyro
pip install pyro-ppl

# Install other dependencies
conda install numpy pandas matplotlib scikit-learn
conda install -c conda-forge scanpy

# Optional: Jupyter
conda install jupyter ipykernel
```

---

## Verify Installation

After installation, verify everything works:

```bash
cd /home/alan/Documents/decipher_batch_correction
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import pyro; print(f'Pyro: {pyro.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import scanpy; print(f'Scanpy: {scanpy.__version__}')"
```

Expected output:
```
PyTorch: 2.x.x
Pyro: 1.x.x
NumPy: 1.x.x
Scanpy: 1.x.x
```

---

## Run Integration Test

Once dependencies are installed:

```bash
cd /home/alan/Documents/decipher_batch_correction
python3 test_integration.py
```

You should see:
```
============================================================
BATCH-CORRECTED DECIPHER INTEGRATION TEST
============================================================

[1/6] Testing imports...
✓ All imports successful

[2/6] Creating mock AnnData...
✓ Created mock data: 200 cells, 100 genes, 3 batches

[3/6] Initializing configuration...
✓ Config initialized:
  - dim_z: 8
  - dim_genes: 100
  - n_batches: 3

[4/6] Creating model...
✓ Model created with XXX parameters

[5/6] Testing forward pass...
✓ Encoder (guide) works
✓ Decoder (z->x with batch correction) works
✓ Attention mechanism works
✓ Imputation works

[6/6] Testing data loaders...
✓ Single data loader works
✓ Train/val split works

============================================================
ALL TESTS PASSED! ✓
============================================================
```

---

## GPU Support (Optional)

### Check NVIDIA GPU availability
```bash
nvidia-smi
```

### Install CUDA toolkit (if needed)
```bash
# For CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Verify GPU PyTorch
```python
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Solution**: Install PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: `ModuleNotFoundError: No module named 'pyro'`
**Solution**: Install Pyro
```bash
pip install pyro-ppl
```

### Issue: Permission denied errors
**Solution**: Use virtual environment or add `--user` flag
```bash
pip install --user torch pyro-ppl numpy
```

### Issue: Out of disk space
**Solution**: Check available space
```bash
df -h
# PyTorch is ~800MB, all dependencies ~2GB total
```

### Issue: Old Python version
**Solution**: Install Python 3.8+
```bash
sudo apt-get install python3.10 python3.10-venv python3.10-dev
# Use python3.10 instead of python3
```

---

## Quick Install Script

Save this as `install_dependencies.sh`:

```bash
#!/bin/bash
set -e

echo "Installing Decipher Batch Correction Dependencies..."

# Create virtual environment
python3 -m venv decipher_env
source decipher_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing Pyro..."
pip install pyro-ppl

echo "Installing other dependencies..."
pip install numpy pandas matplotlib scikit-learn scanpy

echo "Verifying installation..."
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python3 -c "import pyro; print(f'✓ Pyro: {pyro.__version__}')"
python3 -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python3 -c "import scanpy; print(f'✓ Scanpy: {scanpy.__version__}')"

echo ""
echo "Installation complete!"
echo "To activate environment: source decipher_env/bin/activate"
echo "To run tests: python3 test_integration.py"
```

Run with:
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

---

## Minimal Installation (Core Only)

If you want just the essentials:

```bash
python3 -m venv decipher_env
source decipher_env/bin/activate
pip install torch pyro-ppl numpy
```

This is enough to run the batch-corrected model without scanpy/AnnData support.

---

## Package Versions Tested

```
python==3.10.12
torch==2.1.0
pyro-ppl==1.8.6
numpy==1.24.3
scanpy==1.9.6
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
```

---

## Next Steps After Installation

1. Run integration test: `python3 test_integration.py`
2. Check [QUICK_START.md](QUICK_START.md) for usage
3. Review [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) for details
