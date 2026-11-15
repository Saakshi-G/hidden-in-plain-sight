#!/bin/bash
# =============================================================================
# STEP 4 -- compute detection metrics from the experiment results (CPU).
# Reads results/*.jsonl and prints TPR(injection success)/FPR/precision per
# (model, mode), plus a cross-model transferability table.
#
# Submit under YOUR account:
#   sbatch -A <your_account> scripts/script4.slurm
# (Also fine to run directly on a login node after activating the env.)
# =============================================================================
#SBATCH --job-name=hips-4-metrics
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p general                 # <- change to your partition if different
#SBATCH -q class                   # <- change to your QoS if different
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-00:15:00

set -uo pipefail
: "${METRICS_JSON:=results/metrics.json}"
: "${ENV_ACTIVATE:=}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results

if [ -n "${ENV_ACTIVATE}" ]; then
    eval "${ENV_ACTIVATE}"
else
    module load mamba 2>/dev/null || true
    source activate hips-vllm 2>/dev/null || {
        echo "ERROR: activate the env first (run scripts/script0_setup.sh), or set ENV_ACTIVATE." >&2
        exit 1
    }
fi

python -m evaluation.metrics --json "${METRICS_JSON}"
echo "=== $(date) | metrics written to ${METRICS_JSON} ==="
