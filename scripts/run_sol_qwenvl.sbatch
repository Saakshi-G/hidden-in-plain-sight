#!/bin/bash
# =============================================================================
# Hidden in Plain Sight -- run the survey agent on ASU Sol (Slurm) with a
# SMALL open-source vision-language model instead of GLM-5.
#
# Default model: Qwen/Qwen2.5-VL-7B-Instruct
#   - open & ungated, single-GPU, well supported by vLLM
#   - strong OCR, so it can actually perceive the low-perceptibility directive
#     that the visual attack renders (a weak-OCR model would fail the channel
#     for the wrong reason).
# For a tighter GPU (~12-16 GB) use Qwen/Qwen2.5-VL-3B-Instruct instead.
#
# Same pipeline as scripts/run_sol.sbatch: serve locally with vLLM, wait for
# health, run experiments/run_survey.py, tear the server down on exit. The
# agent is model-agnostic -- only the model name/size and GPU request change.
#
# Submit:   sbatch scripts/run_sol_qwenvl.sbatch
# Override: sbatch --export=ALL,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,SERVED_NAME=qwen2.5-vl-3b scripts/run_sol_qwenvl.sbatch
#
# >>> EDIT the #SBATCH lines to match your Sol allocation. Confirm partition/
#     QoS/GPU syntax against current ASU RC docs for Sol.
# =============================================================================
#SBATCH --job-name=hips-qwenvl
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p general                 # partition (Sol: e.g. general / highmem)
#SBATCH -q class                  # QoS      (Sol: e.g. public / grp_<name>)
#SBATCH -G a100:1                  # one GPU is plenty for a 7B VLM
#SBATCH -c 4                       # CPU cores
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00              # walltime D-HH:MM:SS
##SBATCH -A <your_account>         # uncomment + set if your QoS needs an account

set -uo pipefail

# ---- user-tunable configuration (override via --export or environment) ------
: "${MODEL:=Qwen/Qwen2.5-VL-7B-Instruct}"   # HF repo id OR local path
: "${SERVED_NAME:=qwen2.5-vl-7b}"           # name the client/agent uses
: "${MANIFEST:=experiments/sample_manifest.json}"
: "${OUT:=results/run-${SLURM_JOB_ID:-local}.jsonl}"
: "${ENV_ACTIVATE:=}"              # command to activate your Python env (see below)
: "${MAX_MODEL_LEN:=8192}"         # context cap; raise if VRAM allows
: "${GPU_MEM_UTIL:=0.90}"          # vLLM --gpu-memory-utilization
: "${SERVER_TIMEOUT:=900}"         # seconds to wait for the server (7B loads fast)
# -----------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results

# Tensor-parallel size = number of GPUs allocated (7B needs only 1).
TP="$(nvidia-smi -L 2>/dev/null | wc -l)"; TP="${TP:-1}"; [ "$TP" -ge 1 ] || TP=1

PORT=$(( 8000 + ( ${SLURM_JOB_ID:-0} % 1000 ) ))
export GLM_BASE_URL="http://127.0.0.1:${PORT}/v1"   # agent reads these env vars
export GLM_MODEL="${SERVED_NAME}"
export GLM_API_KEY="EMPTY"

echo "=== $(date) | job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "model=${MODEL} served_as=${SERVED_NAME} TP=${TP} port=${PORT}"
echo "manifest=${MANIFEST} out=${OUT}"

# ---- activate the Python environment ----------------------------------------
# Build a vLLM env ONCE (same env serves GLM-5 or Qwen-VL):
#   module load mamba
#   mamba create -n hips-vllm python=3.11 -y && source activate hips-vllm
#   pip install vllm requests pillow numpy
# Qwen2.5-VL needs a recent vLLM (>= 0.6.3) and transformers; a current vllm
# pip install pulls compatible deps.
if [ -n "${ENV_ACTIVATE}" ]; then
    eval "${ENV_ACTIVATE}"
else
    module load mamba 2>/dev/null || true
    source activate hips-vllm 2>/dev/null || {
        echo "ERROR: could not activate env 'hips-vllm'. Set ENV_ACTIVATE or create the env." >&2
        exit 1
    }
fi

python -c "import vllm, requests, PIL" 2>/dev/null || {
    echo "ERROR: vllm/requests/pillow not importable in this env." >&2; exit 1; }

# ---- launch the vLLM OpenAI-compatible server in the background -------------
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

cleanup() {
    echo "stopping vLLM server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- wait until the server is healthy ---------------------------------------
echo "waiting up to ${SERVER_TIMEOUT}s for ${GLM_BASE_URL}/models ..."
deadline=$(( SECONDS + SERVER_TIMEOUT ))
until curl -sf "${GLM_BASE_URL}/models" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited early. Tail of ${SERVER_LOG}:" >&2
        tail -n 40 "${SERVER_LOG}" >&2
        exit 1
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "ERROR: server did not become healthy within ${SERVER_TIMEOUT}s." >&2
        tail -n 40 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 5
done
echo "server is up."

# ---- run the experiment -----------------------------------------------------
python -m experiments.run_survey \
    --manifest "${MANIFEST}" \
    --out "${OUT}" \
    --model "${SERVED_NAME}"
status=$?

echo "=== $(date) | experiment exit=${status} ==="
exit "${status}"
