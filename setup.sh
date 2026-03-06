#!/usr/bin/env bash
# CompRAG setup script
# Creates a Python venv, installs dependencies, checks for CUDA toolkit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "=== CompRAG Setup ==="

# --- Python venv ---
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
else
    echo "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing requirements..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# --- CUDA check ---
echo ""
echo "=== CUDA Toolkit Check ==="
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "CUDA toolkit found: ${CUDA_VERSION}"
else
    echo "WARNING: nvcc not found. CUDA toolkit is required for:"
    echo "  - llama.cpp GPU inference (LLAMA_CUDA=1)"
    echo "  - Local model serving via llama-server"
    echo "  Install CUDA toolkit 12.x from https://developer.nvidia.com/cuda-downloads"
fi

if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. No NVIDIA GPU detected."
fi

# --- llama.cpp note ---
echo ""
echo "=== llama.cpp ==="
if command -v llama-server &>/dev/null; then
    echo "llama-server found on PATH."
else
    echo "NOTE: llama-server not found on PATH."
    echo "  To build llama.cpp with CUDA support:"
    echo "    git clone https://github.com/ggerganov/llama.cpp"
    echo "    cd llama.cpp && make LLAMA_CUDA=1 -j\$(nproc)"
    echo "  Then add the build directory to your PATH."
fi

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
echo "Run tests with: pytest"
echo "Run CompRAG with: python -m comprag --help"
