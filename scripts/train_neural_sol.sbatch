#!/bin/bash
# =============================================================================
# Hidden in Plain Sight -- train the neural steganography channel on ASU Sol,
# logging to Weights & Biases.
#
# Runs `stego.py train-neural`. The model is tiny (64x64), so a single small GPU
# (or even CPU) is enough -- a100:1 here is just for consistency with the other
# jobs; feel free to request a smaller GPU.
#
# W&B: Sol compute nodes often have no outbound internet, so this defaults to
# WANDB_MODE=offline and prints the `wandb sync` command to run afterward from a
# login node. If your nodes DO have internet, submit with WANDB_MODE=online and
# make sure WANDB_API_KEY is set.
#
# Submit:   sbatch scripts/train_neural_sol.sbatch
# Override: sbatch --export=ALL,STEPS=5000,WANDB_MODE=online scripts/train_neural_sol.sbatch
#
# >>> EDIT the #SBATCH lines to match your Sol allocation.
# =============================================================================
#SBATCH --job-name=hips-train
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p general                 # partition
#SBATCH -q public                  # QoS
#SBATCH -G a100:1                  # 1 GPU is overkill for this model; fine
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-01:00:00              # walltime D-HH:MM:SS
##SBATCH -A <your_account>         # uncomment + set if your QoS needs an account

set -uo pipefail

# ---- user-tunable configuration ---------------------------------------------
: "${CHECKPOINT:=results/neural-${SLURM_JOB_ID:-local}.pt}"
: "${STEPS:=3000}"
: "${BATCH_SIZE:=16}"
: "${LR:=1e-3}"
: "${WANDB_PROJECT:=hidden-in-plain-sight}"
: "${WANDB_RUN_NAME:=neural-${SLURM_JOB_ID:-local}}"
: "${WANDB_MODE:=offline}"        # offline | online | disabled
: "${ENV_ACTIVATE:=}"
# WANDB_API_KEY: set it in your ~/.bashrc (or --export) for online mode.
# -----------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results
export WANDB_DIR="results"        # offline runs land under results/wandb/
export WANDB_MODE                 # let the wandb SDK read the mode from env

echo "=== $(date) | job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "checkpoint=${CHECKPOINT} steps=${STEPS} bs=${BATCH_SIZE} lr=${LR}"
echo "wandb: project=${WANDB_PROJECT} run=${WANDB_RUN_NAME} mode=${WANDB_MODE}"

# ---- activate the Python environment (same env as the serving jobs) ---------
if [ -n "${ENV_ACTIVATE}" ]; then
    eval "${ENV_ACTIVATE}"
else
    module load mamba 2>/dev/null || true
    source activate hips-vllm 2>/dev/null || {
        echo "ERROR: could not activate env 'hips-vllm'. Set ENV_ACTIVATE or create the env." >&2
        exit 1
    }
fi

python -c "import torch, wandb" 2>/dev/null || {
    echo "ERROR: torch/wandb not importable in this env (pip install torch wandb)." >&2
    exit 1; }

# ---- train ------------------------------------------------------------------
python stego.py train-neural \
    --checkpoint "${CHECKPOINT}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN_NAME}"
status=$?

echo "=== $(date) | training exit=${status} | checkpoint: ${CHECKPOINT} ==="
if [ "${WANDB_MODE}" = "offline" ]; then
    echo "W&B ran offline. From a login node WITH internet, upload the run:"
    echo "  wandb sync ${WANDB_DIR}/wandb/offline-run-*"
fi
exit "${status}"
