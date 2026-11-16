#!/bin/bash
# =============================================================================
# ONE-SHOT pipeline (GPU): data -> [optional train] -> experiment -> metrics.
# Runs the whole thing in order in a single job. Metrics are printed by the
# experiment step itself (run_survey), so there is no separate metrics stage.
#
# Prereq (once, login node):  bash scripts/script0_setup.sh
# Submit under YOUR account:
#   sbatch -A <your_account> scripts/run_all.slurm
# Also fine-tune first, merge the adapter, and evaluate the fine-tuned model:
#   sbatch -A <acct> --export=ALL,RUN_TRAIN=1 scripts/run_all.slurm
# Smaller model:
#   sbatch -A <acct> --export=ALL,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,SERVED_NAME=qwen2.5-vl-3b scripts/run_all.slurm
#
# For granular control / parallel runs, the per-stage scripts (script1..4.slurm)
# still exist; this just chains them.
# =============================================================================
#SBATCH --job-name=hips-all
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p htc                 # <- change to your partition if different
#SBATCH -q class                   # <- change to your QoS if different
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00

set -uo pipefail

# ---- config -----------------------------------------------------------------
: "${MODEL:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${SERVED_NAME:=qwen2.5-vl-7b}"
: "${DATA_DIR:=data/synthetic}"
: "${N:=400}"                      # dataset size (only used if data is missing)
: "${MANIFEST:=${DATA_DIR}/manifest_eval.json}"
: "${OUT:=results/run-${SLURM_JOB_ID:-local}.jsonl}"
: "${RUN_TRAIN:=0}"               # 1 = LoRA fine-tune before the experiment
: "${OUTPUT_DIR:=runs/qwenvl-lora-${SLURM_JOB_ID:-local}}"
: "${MAX_STEPS:=-1}"
: "${LOAD_4BIT:=0}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEM_UTIL:=0.90}"
: "${SERVER_TIMEOUT:=900}"
: "${WANDB_PROJECT:=hidden-in-plain-sight}"
: "${WANDB_MODE:=offline}"
: "${ENV_ACTIVATE:=}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results runs
export WANDB_DIR="results"; export WANDB_MODE; export WANDB_PROJECT

echo "=== $(date) | job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "model=${MODEL} data=${DATA_DIR} run_train=${RUN_TRAIN} out=${OUT}"

# ---- activate env -----------------------------------------------------------
if [ -n "${ENV_ACTIVATE}" ]; then
    eval "${ENV_ACTIVATE}"
else
    module load mamba/latest 2>/dev/null || true
    eval "$(mamba shell hook --shell bash)"
    mamba activate hips-vllm || {
        echo "ERROR: activate the env first (run scripts/script0_setup.sh), or set ENV_ACTIVATE." >&2
        exit 1

    }

fi

# Verify batch job is actually using hips-vllm
echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
import scipy
import vllm

print("torch:", torch.__version__)
print("scipy:", scipy.__version__)
print("vllm:", vllm.__version__)
print("environment check OK")
PY

python -c "import vllm, requests, PIL" 2>/dev/null || {
    echo "ERROR: core deps missing. Re-run scripts/script0_setup.sh." >&2; exit 1; }

# ---- STEP 1: dataset (generate only if missing) -----------------------------
if [ ! -f "${MANIFEST}" ]; then
    echo "--- STEP 1: generating dataset (${N} scenes) -> ${DATA_DIR}"
    python -m data.generate_synthetic --n "${N}" --out-dir "${DATA_DIR}" || {
        echo "ERROR: dataset generation failed." >&2; exit 1; }
else
    echo "--- STEP 1: dataset already present at ${MANIFEST} (skipping)"
fi

# ---- STEP 2: optional LoRA fine-tune ----------------------------------------
if [ "${RUN_TRAIN}" = "1" ]; then
    echo "--- STEP 2: LoRA fine-tune ${MODEL} -> ${OUTPUT_DIR}"
    EXTRA=""; [ "${LOAD_4BIT}" = "1" ] && EXTRA="--load-4bit"
    python -m training.train_vlm_lora \
        --data-dir "${DATA_DIR}" --model "${MODEL}" --output-dir "${OUTPUT_DIR}" \
        --max-steps "${MAX_STEPS}" --wandb-project "${WANDB_PROJECT}" \
        --run-name "vlm-ft-${SLURM_JOB_ID:-local}" ${EXTRA} || {
        echo "ERROR: training failed." >&2; exit 1; }
    MERGED_DIR="${OUTPUT_DIR}-merged"
    echo "--- STEP 2b: merging adapter -> ${MERGED_DIR}"
    python -m training.merge_lora --adapter-dir "${OUTPUT_DIR}" --output-dir "${MERGED_DIR}" || {
        echo "ERROR: LoRA merge failed." >&2; exit 1; }
    MODEL="${MERGED_DIR}"; SERVED_NAME="${SERVED_NAME}-ft"
    echo "evaluating fine-tuned model: ${MODEL} (served as ${SERVED_NAME})"
else
    echo "--- STEP 2: training skipped (set RUN_TRAIN=1 to enable)"
fi

# ---- STEP 3: serve + experiment (+ metrics, printed by run_survey) ----------
echo "--- STEP 3: serve ${MODEL} and run the experiment"
TP="$(nvidia-smi -L 2>/dev/null | wc -l)"; TP="${TP:-1}"; [ "$TP" -ge 1 ] || TP=1
PORT=$(( 8000 + ( ${SLURM_JOB_ID:-0} % 1000 ) ))
export GLM_BASE_URL="http://127.0.0.1:${PORT}/v1"; export GLM_MODEL="${SERVED_NAME}"; export GLM_API_KEY="EMPTY"

SERVER_LOG="results/vllm-${SLURM_JOB_ID:-local}.log"
vllm serve "${MODEL}" --served-model-name "${SERVED_NAME}" \
    --host 127.0.0.1 --port "${PORT}" --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --limit-mm-per-prompt '{"image":1}' --trust-remote-code > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting up to ${SERVER_TIMEOUT}s for the server ..."
deadline=$(( SECONDS + SERVER_TIMEOUT ))
until curl -sf "${GLM_BASE_URL}/models" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited early:" >&2; tail -n 40 "${SERVER_LOG}" >&2; exit 1; fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "ERROR: server not healthy in time:" >&2; tail -n 40 "${SERVER_LOG}" >&2; exit 1; fi
    sleep 5
done
echo "server is up."

python -m experiments.run_survey --manifest "${MANIFEST}" --out "${OUT}" --model "${SERVED_NAME}"
status=$?
echo "=== $(date) | pipeline exit=${status} | results+metrics: ${OUT} ==="
[ "${WANDB_MODE}" = "offline" ] && [ "${RUN_TRAIN}" = "1" ] && echo "sync logs: wandb sync ${WANDB_DIR}/wandb/offline-run-*"
exit "${status}"
