#!/bin/bash
# =============================================================================
# STEP 3 -- run the injection/detection experiment (GPU).
# Serves a vision-language model locally with vLLM, runs a manifest of survey
# items, and writes results/run-<jobid>.jsonl (+ .summary.json).
#
# Needs STEP 1 done (data/synthetic/manifest_eval.json).
# Submit under YOUR account:
#   sbatch -A <your_account> scripts/script3.slurm
# Options:
#   # smaller model:
#   sbatch -A <acct> --export=ALL,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,SERVED_NAME=qwen2.5-vl-3b scripts/script3.slurm
#   # evaluate a FINE-TUNED checkpoint (merged LoRA dir from STEP 2):
#   sbatch -A <acct> --export=ALL,MODEL=runs/qwenvl-merged,SERVED_NAME=qwenvl-ft scripts/script3.slurm
# =============================================================================
#SBATCH --job-name=hips-3-exp
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p general                 # <- change to your partition if different
#SBATCH -q class                   # <- change to your QoS if different
#SBATCH -G a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-02:00:00

set -uo pipefail
: "${MODEL:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${SERVED_NAME:=qwen2.5-vl-7b}"
: "${MANIFEST:=data/synthetic/manifest_eval.json}"
: "${OUT:=results/run-${SLURM_JOB_ID:-local}.jsonl}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEM_UTIL:=0.90}"
: "${SERVER_TIMEOUT:=900}"
: "${ENV_ACTIVATE:=}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results

TP="$(nvidia-smi -L 2>/dev/null | wc -l)"; TP="${TP:-1}"; [ "$TP" -ge 1 ] || TP=1
PORT=$(( 8000 + ( ${SLURM_JOB_ID:-0} % 1000 ) ))
export GLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export GLM_MODEL="${SERVED_NAME}"
export GLM_API_KEY="EMPTY"

echo "=== $(date) | job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "model=${MODEL} served_as=${SERVED_NAME} TP=${TP} port=${PORT} manifest=${MANIFEST}"

if [ -n "${ENV_ACTIVATE}" ]; then
    eval "${ENV_ACTIVATE}"
else
    module load mamba 2>/dev/null || true
    source activate hips-vllm 2>/dev/null || {
        echo "ERROR: activate the env first (run scripts/script0_setup.sh), or set ENV_ACTIVATE." >&2
        exit 1
    }
fi

python -c "import vllm, requests, PIL" 2>/dev/null || {
    echo "ERROR: vllm/requests/pillow not importable. Re-run scripts/script0_setup.sh." >&2; exit 1; }
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: ${MANIFEST} not found. Run STEP 1 (scripts/script1.slurm) first." >&2; exit 1
fi

SERVER_LOG="results/vllm-${SLURM_JOB_ID:-local}.log"
echo "starting vLLM server -> ${SERVER_LOG}"
vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host 127.0.0.1 --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --limit-mm-per-prompt image=1 \
    --trust-remote-code \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { echo "stopping vLLM server (pid ${SERVER_PID})"; kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting up to ${SERVER_TIMEOUT}s for the server ..."
deadline=$(( SECONDS + SERVER_TIMEOUT ))
until curl -sf "${GLM_BASE_URL}/models" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited early. Tail of ${SERVER_LOG}:" >&2; tail -n 40 "${SERVER_LOG}" >&2; exit 1
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "ERROR: server not healthy within ${SERVER_TIMEOUT}s." >&2; tail -n 40 "${SERVER_LOG}" >&2; exit 1
    fi
    sleep 5
done
echo "server is up."

python -m experiments.run_survey --manifest "${MANIFEST}" --out "${OUT}" --model "${SERVED_NAME}"
status=$?
echo "=== $(date) | experiment exit=${status} | results: ${OUT} ==="
exit "${status}"
