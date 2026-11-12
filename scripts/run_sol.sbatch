#!/bin/bash
# =============================================================================
# Hidden in Plain Sight -- run the GLM-5 survey agent on ASU Sol (Slurm).
#
# Serves GLM-5 locally with vLLM (OpenAI-compatible), waits until it is healthy,
# then runs experiments/run_survey.py over a manifest and writes results/.
#
# Submit with:   sbatch scripts/run_sol.sbatch
# Override bits: sbatch --export=ALL,MANIFEST=experiments/my.json scripts/run_sol.sbatch
#
# >>> EDIT the #SBATCH lines (account / partition / QoS / GPUs / time) to match
#     your Sol allocation, and set MODEL below to the GLM-5 weights you have.
#     Confirm partition/QoS/GPU syntax against current ASU RC docs for Sol.
# =============================================================================
#SBATCH --job-name=hips-glm5
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p general                 # partition (Sol: e.g. general / highmem)
#SBATCH -q public                  # QoS      (Sol: e.g. public / grp_<name>)
#SBATCH -G a100:2                  # GPUs: <type>:<count>; TP size = count
#SBATCH -c 16                      # CPU cores
#SBATCH --mem=128G
#SBATCH -t 0-04:00:00              # walltime D-HH:MM:SS
##SBATCH -A <your_account>         # uncomment + set if your QoS needs an account

set -uo pipefail

# ---- user-tunable configuration (override via --export or environment) ------
: "${MODEL:=zai-org/GLM-5}"        # HF repo id OR local path to GLM-5 weights
: "${SERVED_NAME:=glm-5}"          # name the client/agent uses for the model
: "${MANIFEST:=experiments/sample_manifest.json}"
: "${OUT:=results/run-${SLURM_JOB_ID:-local}.jsonl}"
: "${ENV_ACTIVATE:=}"              # command to activate your Python env (see below)
: "${MAX_MODEL_LEN:=8192}"         # cap context to fit VRAM; raise if you can
: "${GPU_MEM_UTIL:=0.90}"          # vLLM --gpu-memory-utilization
: "${SERVER_TIMEOUT:=1800}"        # seconds to wait for the server to come up
# -----------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results

# Tensor-parallel size = number of GPUs allocated to the job.
TP="$(nvidia-smi -L 2>/dev/null | wc -l)"; TP="${TP:-1}"; [ "$TP" -ge 1 ] || TP=1

# Pick a per-job port so concurrent jobs don't collide.
PORT=$(( 8000 + ( ${SLURM_JOB_ID:-0} % 1000 ) ))
export GLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export GLM_MODEL="${SERVED_NAME}"
export GLM_API_KEY="EMPTY"

echo "=== $(date) | job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "model=${MODEL} served_as=${SERVED_NAME} TP=${TP} port=${PORT}"
echo "manifest=${MANIFEST} out=${OUT}"

# ---- activate the Python environment ----------------------------------------
# Sol uses mamba modules. Build a vLLM env ONCE on a login/compute node, e.g.:
#   module load mamba
#   mamba create -n hips-vllm python=3.11 -y && source activate hips-vllm
#   pip install vllm requests pillow numpy
# Then either set ENV_ACTIVATE below, or edit these two lines to match.
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
    --trust-remote-code \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# Always stop the server when the job ends (success, failure, or cancel).
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
