#!/bin/bash
# =============================================================================
# Hidden in Plain Sight -- LoRA fine-tune Qwen2.5-VL on the synthetic
# content-aware injection dataset, on ASU Sol (Slurm), logging to W&B.
#
# Prereqs (run once on a login node):
#   bash scripts/setup_sol.sh                  # env with vllm/transformers/peft/...
#   python -m data.generate_synthetic --n 400 --out-dir data/synthetic
#
# Submit:   sbatch scripts/train_vlm_sol.sbatch
# Smoke:    sbatch --export=ALL,MAX_STEPS=2 scripts/train_vlm_sol.sbatch
# 3B model: sbatch --export=ALL,MODEL=Qwen/Qwen2.5-VL-3B-Instruct scripts/train_vlm_sol.sbatch
#
# >>> EDIT the #SBATCH lines to match your Sol allocation.
# =============================================================================
#SBATCH --job-name=hips-vlm-ft
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p general
#SBATCH -q public
#SBATCH -G a100:1                  # 7B LoRA fits one 80GB A100; use --load-4bit on 40GB
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-06:00:00
##SBATCH -A <your_account>

set -uo pipefail

: "${MODEL:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${DATA_DIR:=data/synthetic}"
: "${OUTPUT_DIR:=runs/qwenvl-lora-${SLURM_JOB_ID:-local}}"
: "${EPOCHS:=2}"
: "${BATCH_SIZE:=2}"
: "${GRAD_ACCUM:=8}"
: "${LR:=1e-4}"
: "${MAX_STEPS:=-1}"
: "${LOAD_4BIT:=0}"                # set 1 to QLoRA on a 40GB GPU
: "${WANDB_PROJECT:=hidden-in-plain-sight}"
: "${WANDB_MODE:=offline}"
: "${ENV_ACTIVATE:=}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results runs
export WANDB_DIR="results"
export WANDB_MODE
export WANDB_PROJECT

echo "=== $(date) | job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "model=${MODEL} data=${DATA_DIR} out=${OUTPUT_DIR} 4bit=${LOAD_4BIT} max_steps=${MAX_STEPS}"

if [ -n "${ENV_ACTIVATE}" ]; then
    eval "${ENV_ACTIVATE}"
else
    module load mamba 2>/dev/null || true
    source activate hips-vllm 2>/dev/null || {
        echo "ERROR: could not activate env 'hips-vllm'. Run scripts/setup_sol.sh first." >&2
        exit 1
    }
fi

python -c "import torch, transformers, peft, qwen_vl_utils" 2>/dev/null || {
    echo "ERROR: training deps missing (transformers/peft/qwen-vl-utils). Re-run scripts/setup_sol.sh." >&2
    exit 1; }

if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "ERROR: ${DATA_DIR}/train.jsonl not found. Run: python -m data.generate_synthetic --out-dir ${DATA_DIR}" >&2
    exit 1
fi

EXTRA=""
[ "${LOAD_4BIT}" = "1" ] && EXTRA="--load-4bit"

python -m training.train_vlm_lora \
    --data-dir "${DATA_DIR}" \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max-steps "${MAX_STEPS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --run-name "vlm-ft-${SLURM_JOB_ID:-local}" \
    ${EXTRA}
status=$?

echo "=== $(date) | training exit=${status} | adapter: ${OUTPUT_DIR} ==="
if [ "${WANDB_MODE}" = "offline" ]; then
    echo "sync W&B from a login node:  wandb sync ${WANDB_DIR}/wandb/offline-run-*"
fi
exit "${status}"
