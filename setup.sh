#!/usr/bin/env bash
# setup.sh — One-shot environment setup for CUMRAG eval harness
# Installs system deps, creates venv, installs Python deps, builds llama.cpp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LLAMA_DIR="${HOME}/llama.cpp"

echo "=== CUMRAG Setup ==="

# --- System dependencies ---
echo "[1/5] Installing system dependencies..."
if command -v apt &>/dev/null; then
    sudo apt update && sudo apt install -y \
        python3 python3-venv python3-pip \
        git cmake build-essential
elif command -v dnf &>/dev/null; then
    sudo dnf install -y python3 python3-devel git cmake gcc gcc-c++ make
elif command -v pacman &>/dev/null; then
    sudo pacman -Sy --noconfirm python python-pip git cmake base-devel
else
    echo "WARNING: Unknown package manager. Install python3, python3-venv, git, cmake, build-essential manually."
fi

# --- Python virtual environment ---
echo "[2/5] Creating Python virtual environment at ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# --- Python dependencies ---
echo "[3/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# --- llama.cpp ---
echo "[4/5] Building llama.cpp..."
if [ ! -d "${LLAMA_DIR}" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_DIR}"
fi
cd "${LLAMA_DIR}"
git pull --ff-only 2>/dev/null || true

# Detect CUDA
CUDA_FLAG="-DGGML_CUDA=OFF"
if command -v nvcc &>/dev/null; then
    echo "  CUDA detected, building with GPU support."
    CUDA_FLAG="-DGGML_CUDA=ON"
else
    echo "  No CUDA detected, building CPU-only."
fi

cmake -B build ${CUDA_FLAG}
cmake --build build --config Release -j"$(nproc)"

cd "${SCRIPT_DIR}"

# --- Directory structure ---
echo "[5/5] Ensuring directory structure..."
mkdir -p config scripts cumrag datasets/rgb datasets/nq datasets/halueval \
    models index results/raw results/aggregated results/figures docs

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
echo "llama.cpp server at: ${LLAMA_DIR}/build/bin/llama-server"
