#!/usr/bin/env bash
# Create a virtual environment, install requirements, and activate it.
#
# Run with `source` so the activation sticks in your current shell:
#   source setup.sh
# (Running `bash setup.sh` installs everything, but activation ends when the script exits.)

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# Use `return` when sourced, `exit` when executed.
_fail() { echo "setup.sh: $1" >&2; return 1 2>/dev/null || exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || _fail "cannot cd to $SCRIPT_DIR" || return 1 2>/dev/null

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR" || _fail "venv creation failed (on Ubuntu: sudo apt install python3-venv)" || return 1 2>/dev/null
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate" || _fail "could not activate $VENV_DIR" || return 1 2>/dev/null

python -m pip install --upgrade pip || _fail "pip upgrade failed" || return 1 2>/dev/null

# Torch first from the CUDA 12.8 index (needed for RTX 50-series / sm_120).
echo "Installing torch/torchvision from $TORCH_INDEX_URL ..."
pip install torch torchvision --index-url "$TORCH_INDEX_URL" || _fail "torch install failed" || return 1 2>/dev/null

echo "Installing requirements.txt ..."
pip install -r requirements.txt || _fail "requirements install failed" || return 1 2>/dev/null

echo
echo "Done. Virtual environment active: $VIRTUAL_ENV"
echo "Deactivate with: deactivate"
