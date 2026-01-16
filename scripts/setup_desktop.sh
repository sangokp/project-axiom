#!/usr/bin/env bash

# Project Axiom: Desktop (RTX 5090) Setup Script
#
# Run this on the Windows Desktop via SSH to set up the training environment.
# Prerequisites: Python 3.11, CUDA 12.x, Git
#
# Usage:
#   ssh Dan@100.125.59.26
#   git clone https://github.com/sangokp/project-axiom.git
#   cd project-axiom
#   git checkout young-link-axiom
#   ./scripts/setup_desktop.sh

set -e

echo "=== Project Axiom: Desktop Setup ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Python: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ "3.10" ]] && [[ ! "$PYTHON_VERSION" =~ "3.11" ]]; then
    echo "WARNING: Python 3.10 or 3.11 recommended for TensorFlow compatibility"
    echo "Current: $PYTHON_VERSION"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU Info ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo ""
else
    echo "WARNING: nvidia-smi not found. CUDA may not be installed."
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate (Windows vs Unix)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support first (for RTX 5090)
echo ""
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install TensorFlow with GPU support
echo ""
echo "Installing TensorFlow..."
pip install tensorflow[and-cuda]

# Install remaining requirements
echo ""
echo "Installing project requirements..."
pip install -r requirements.txt

# Install the project in development mode
echo ""
echo "Installing project in development mode..."
pip install -e .

# Verify installations
echo ""
echo "=== Verification ==="
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import melee; print(f'libmelee: OK')"
python -c "import peppi; print(f'peppi-py: OK')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy replay data to: data/replays/raw/"
echo "  2. Parse replays: python slippi_db/parse_local.py --input_dir data/replays/raw --output_dir data/replays/parsed --allowed_characters younglink"
echo "  3. Augment data: python scripts/augment_yl_data.py --input data/replays/parsed --output data/replays/augmented --mirror --player_swap"
echo "  4. Set environment variables:"
echo "     export DATA_ROOT=./data/replays"
echo "     export MODEL_ROOT=./models"
echo "     export DOLPHIN_PATH=/path/to/slippi-dolphin"
echo "     export ISO_PATH=/path/to/SSBM.iso"
echo "  5. Start training: ./scripts/yl_imitation.sh"
