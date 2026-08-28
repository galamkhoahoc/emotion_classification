#!/bin/bash
# =============================================================================
# ViEmoText - All-in-One Training Script
# =============================================================================
# Chạy toàn bộ pipeline: cài đặt requirements → train → evaluate
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh                          # Chạy với config mặc định (phobert)
#   ./run_all.sh --model_type bamibert    # Chạy với BamiBERT
#   ./run_all.sh --num_epochs 5 --batch_size 32
# =============================================================================

set -e  # Dừng ngay nếu có lỗi

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# =============================================================================
# Helper functions
# =============================================================================
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║             🎯 ViEmoText - Emotion Classification          ║"
    echo "║                   All-in-One Training Pipeline              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}  ✔ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}  ✘ $1${NC}"
}

# =============================================================================
# Main pipeline
# =============================================================================
print_banner

START_TIME=$(date +%s)

# ─── Step 1: Check Python ────────────────────────────────────────────────────
print_step "Step 1/4: Checking Python installation"

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found! Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
print_success "Found: $PYTHON_VERSION"

# ─── Step 2: Create virtual environment ──────────────────────────────────────
print_step "Step 2/4: Setting up environment"

# Detect Google Colab / container environments where venv is unnecessary
IS_COLAB=false
if [ -n "$COLAB_RELEASE_TAG" ] || [ -d "/content" ] && [ -f "/usr/local/bin/pip" ]; then
    IS_COLAB=true
fi

if [ "$IS_COLAB" = true ]; then
    print_warning "Google Colab detected — skipping virtual environment."
    print_success "Using Colab's built-in environment."
else
    if [ ! -d "$VENV_DIR" ]; then
        echo "  Creating virtual environment at $VENV_DIR ..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_success "Virtual environment created."
    else
        print_warning "Virtual environment already exists, reusing it."
    fi

    # Activate virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        source "$VENV_DIR/Scripts/activate"
    else
        print_error "Cannot find virtual environment activation script!"
        exit 1
    fi
    print_success "Virtual environment activated."
fi

# Upgrade pip
echo "  Upgrading pip..."
pip install --upgrade pip --quiet
print_success "pip upgraded."

# ─── Step 3: Install requirements ────────────────────────────────────────────
print_step "Step 3/4: Installing requirements"

if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

pip install -r "$SCRIPT_DIR/requirements.txt"
print_success "All dependencies installed."

# Check GPU availability
echo ""
$PYTHON_CMD -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  ✔ GPU detected: {gpu_name} ({gpu_mem:.1f} GB)')
else:
    print('  ⚠ No GPU detected. Training will use CPU (slower).')
"

# ─── Step 4: Start training ──────────────────────────────────────────────────
print_step "Step 4/4: Starting training"

echo "  Arguments passed: $@"
echo ""

$PYTHON_CMD "$SCRIPT_DIR/scripts/train.py" "$@"

# ─── Done ────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}${GREEN}          ✅  Pipeline completed successfully!              ${NC}${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Total time: ${YELLOW}${MINUTES}m ${SECONDS}s${NC}                                     ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                            ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  📁 Model saved to:     ${YELLOW}outputs/${NC}                          ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  📁 Checkpoints:        ${YELLOW}checkpoints/${NC}                      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  📁 Logs:               ${YELLOW}logs/${NC}                             ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
