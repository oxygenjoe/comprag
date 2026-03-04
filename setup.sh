#!/usr/bin/env bash
# setup.sh — One-shot environment setup for CompRAG eval harness
# Installs system deps, creates venv, installs Python deps, downloads spacy
# model, builds llama.cpp, downloads/normalizes datasets, builds vector index.
#
# Flags:
#   --skip-datasets   Skip dataset download and normalization
#   --skip-index      Skip vector index build
#   --skip-llama      Skip llama.cpp clone and build
#   --help, -h        Show usage
#
# Each step is idempotent: re-running is safe, completed steps are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LLAMA_DIR="${HOME}/llama.cpp"

# --- Parse flags ---
SKIP_DATASETS=0
SKIP_INDEX=0
SKIP_LLAMA=0
for arg in "$@"; do
    case "$arg" in
        --skip-datasets) SKIP_DATASETS=1 ;;
        --skip-index)    SKIP_INDEX=1 ;;
        --skip-llama)    SKIP_LLAMA=1 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "One-shot environment setup for CompRAG eval harness."
            echo "Each step is idempotent — safe to re-run."
            echo ""
            echo "Options:"
            echo "  --skip-datasets   Skip dataset download and normalization"
            echo "  --skip-index      Skip vector index build"
            echo "  --skip-llama      Skip llama.cpp clone and build"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: $0 [--skip-datasets] [--skip-index] [--skip-llama] [--help]"
            exit 1
            ;;
    esac
done

TOTAL_STEPS=8
CURRENT_STEP=0
step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo ""
    echo "[${CURRENT_STEP}/${TOTAL_STEPS}] $1"
}

echo "=== CompRAG Setup ==="
echo "  skip-datasets: ${SKIP_DATASETS}"
echo "  skip-index:    ${SKIP_INDEX}"
echo "  skip-llama:    ${SKIP_LLAMA}"

# --- Step 1: System dependencies ---
step "Installing system dependencies..."
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

# --- Step 2: Python virtual environment ---
step "Creating Python virtual environment at ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created new venv."
else
    echo "  Venv already exists, skipping creation."
fi
source "${VENV_DIR}/bin/activate"

# --- Step 3: Python dependencies (staged install) ---
step "Installing Python dependencies..."
pip install --upgrade pip

# Staged install: ragchecker and refchecker have strict version pins that
# conflict with langchain-anthropic. Install core deps first, then install
# ragchecker/refchecker with --no-deps to avoid resolver conflicts.
# Build a filtered requirements file on-the-fly, excluding the conflicting packages.
CORE_REQS=$(grep -v -E '^(ragchecker|refchecker)' "${SCRIPT_DIR}/requirements.txt")
echo "${CORE_REQS}" | pip install -r /dev/stdin

# Now install ragchecker and refchecker with --no-deps to avoid pulling
# incompatible pinned transitive dependencies.
pip install --no-deps ragchecker refchecker

# --- Step 4: spaCy model ---
step "Downloading spaCy language model (en_core_web_sm)..."
if python -c "import spacy; spacy.load('en_core_web_sm')" &>/dev/null; then
    echo "  en_core_web_sm already installed, skipping."
else
    python -m spacy download en_core_web_sm
fi

# --- Step 5: llama.cpp ---
step "Building llama.cpp..."
if [ "${SKIP_LLAMA}" -eq 1 ]; then
    echo "  --skip-llama flag set, skipping."
else
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
fi

# --- Step 6: Directory structure ---
step "Ensuring directory structure..."
mkdir -p config scripts cumrag datasets/rgb datasets/nq datasets/halueval \
    models index results/raw results/scored results/aggregated results/figures docs

# --- Step 7: Dataset download and normalization ---
step "Downloading and normalizing datasets..."
if [ "${SKIP_DATASETS}" -eq 1 ]; then
    echo "  --skip-datasets flag set, skipping."
else
    # Download datasets (idempotent — download_datasets.py skips existing)
    echo "  Downloading datasets..."
    python "${SCRIPT_DIR}/scripts/download_datasets.py"

    # Normalize datasets (idempotent — checks for normalized/ dirs)
    echo "  Normalizing datasets..."
    python "${SCRIPT_DIR}/scripts/normalize_datasets.py" --all
fi

# --- Step 8: Build vector index ---
step "Building vector index..."
if [ "${SKIP_INDEX}" -eq 1 ]; then
    echo "  --skip-index flag set, skipping."
else
    # build_index.py --all is idempotent — skips collections that exist
    python "${SCRIPT_DIR}/scripts/build_index.py" --all
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
echo "llama.cpp server at: ${LLAMA_DIR}/build/bin/llama-server"
