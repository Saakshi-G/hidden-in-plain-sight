#!/bin/bash
# =============================================================================
# STEP 2 -- LoRA fine-tune Qwen2.5-VL on the synthetic dataset (GPU).
#
# Needs STEP 1 done (data/synthetic/train.jsonl).
# Submit under YOUR account:
#   sbatch -A <your_account> scripts/script2.slurm
# Smoke-test the training first (RECOMMENDED -- it is untested on GPU):
#   sbatch -A <acct> --export=ALL,MAX_STEPS=2 scripts/script2.slurm
# Smaller GPU / model:
#   sbatch -A <acct> --export=ALL,MODEL=Qwen/Qwen2.5-VL-3B-Instruct scripts/script2.slurm
#   sbatch -A <acct> --export=ALL,LOAD_4BIT=1 scripts/script2.slurm   # QLoRA on 40GB
# =============================================================================
#SBATCH --job-name=hips-2-train
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p general                 # <- change to your partition if different
#SBATCH -q class                   # <- change to your QoS if different
#SBATCH -G a100:1                  # 7B LoRA fits one 80GB A100; use LOAD_4BIT=1 on 40GB
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-06:00:00

set -uo pipefail
: "${MODEL:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${DATA_DIR:=data/synthetic}"
: "${OUTPUT_DIR:=runs/qwenvl-lora-${SLURM_JOB_ID:-local}}"
: "${EPOCHS:=2}"
: "${BATCH_SIZE:=2}"
: "${GRAD_ACCUM:=8}"
: "${LR:=1e-4}"
: "${MAX_STEPS:=-1}"
: "${LOAD_4BIT:=0}"
: "${WANDB_PROJECT:=hidden-in-plain-sight}"
: "${WANDB_MODE:=offline}"
: "${ENV_ACTIVATE:=}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results runs
export WANDB_DIR="results"; export WANDB_MODE; export WANDB_PROJECT

echo "=== $(date) | job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "model=${MODEL} data=${DATA_DIR} out=${OUTPUT_DIR} 4bit=${LOAD_4BIT} max_steps=${MAX_STEPS}"

if [ -n "${ENV_ACTIVATE}" ]; then
    eval "${ENV_ACTIVATE}"
else
    module load mamba 2>/dev/null || true
    source activate hips-vllm 2>/dev/null || {
        echo "ERROR: activate the env first (run scripts/script0_setup.sh), or set ENV_ACTIVATE." >&2
        exit 1
    }
fi

python -c "import torch, transformers, peft, qwen_vl_utils" 2>/dev/null || {
    echo "ERROR: training deps missing. Re-run scripts/script0_setup.sh." >&2; exit 1; }
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "ERROR: ${DATA_DIR}/train.jsonl not found. Run STEP 1 (scripts/script1.slurm) first." >&2
    exit 1
fi

EXTRA=""; [ "${LOAD_4BIT}" = "1" ] && EXTRA="--load-4bit"

python -m training.train_vlm_lora \
    --data-dir "${DATA_DIR}" --model "${MODEL}" --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}" --max-steps "${MAX_STEPS}" \
    --wandb-project "${WANDB_PROJECT}" --run-name "vlm-ft-${SLURM_JOB_ID:-local}" ${EXTRA}
status=$?

echo "=== $(date) | training exit=${status} | adapter: ${OUTPUT_DIR} ==="
[ "${WANDB_MODE}" = "offline" ] && echo "sync logs:  wandb sync ${WANDB_DIR}/wandb/offline-run-*"
exit "${status}"
