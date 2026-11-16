#!/usr/bin/env bash
# =============================================================================
# Set up the ASU Sol environment for "Hidden in Plain Sight".
#
# Creates a mamba env (default: hips-vllm, Python 3.11) with everything the
# Slurm jobs need: vLLM (serving) + torch, wandb, requests, pillow, numpy.
# This is the env the scripts/*.sbatch jobs activate.
#
# Run on a Sol LOGIN node (has internet; compute nodes may not):
#   bash scripts/setup_sol.sh
#
# Options:
#   --verify-only   don't install; just check the current Python has the deps
#
# Env overrides:
#   ENV_NAME=myenv PYTHON_VERSION=3.11 bash scripts/setup_sol.sh
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-hips-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
REQUIRED_MODULES="torch vllm wandb requests PIL numpy cryptography transformers peft qwen_vl_utils"

verify() {
    python - "$@" <<'PY'
import sys
from importlib.util import find_spec
mods = sys.argv[1:] or ["torch", "vllm", "wandb", "requests", "PIL", "numpy"]
missing = [m for m in mods if find_spec(m) is None]
ok = [m for m in mods if m not in missing]
print("python  :", sys.version.split()[0], "(" + sys.executable + ")")
print("present :", ", ".join(ok) if ok else "(none)")
if missing:
    print("MISSING :", ", ".join(missing))
    sys.exit(1)
print("environment OK")
PY
}

# --- verify-only short-circuit (also handy to re-check an existing env) -------
if [ "${1:-}" = "--verify-only" ]; then
    verify ${REQUIRED_MODULES}
    exit $?
fi

# --- 1. make mamba available -------------------------------------------------
if ! command -v mamba >/dev/null 2>&1; then
    module load mamba 2>/dev/null || {
        echo "ERROR: 'mamba' not found. Run this on Sol (where 'module load mamba'" >&2
        echo "       works), or install mambaforge first." >&2
        exit 1
    }
fi

# --- 2. create the env if it doesn't exist -----------------------------------
if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "env '${ENV_NAME}' already exists; reusing it."
else
    echo "creating env '${ENV_NAME}' (python ${PYTHON_VERSION}) ..."
    mamba create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

# --- 3. activate -------------------------------------------------------------
# shellcheck disable=SC1091
source activate "${ENV_NAME}"

# --- 4. install --------------------------------------------------------------
python -m pip install --upgrade pip
# vLLM pulls a matching torch/CUDA build on Sol, so we don't pin torch here.
echo "installing vllm + project deps (this can take a few minutes) ..."
pip install vllm wandb requests pillow numpy scipy cryptography
# VLM fine-tuning stack (transformers/accelerate come with vllm, but pin extras)
pip install "transformers>=4.49" peft accelerate datasets qwen-vl-utils
# Optional QLoRA on 40GB GPUs:  pip install bitsandbytes

# --- 5. verify ---------------------------------------------------------------
echo
verify ${REQUIRED_MODULES}

echo
echo "Done. Re-activate later with:"
echo "    module load mamba && source activate ${ENV_NAME}"
echo "Then submit jobs, e.g.:  sbatch scripts/run_sol_qwenvl.sbatch"
