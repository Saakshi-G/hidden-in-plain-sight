#!/bin/bash
# =============================================================================
# STEP 1 -- generate the synthetic content-aware injection dataset (CPU, fast).
#
# Submit under YOUR account:
#   sbatch -A <your_account> scripts/script1.slurm
# Override size/output:
#   sbatch -A <acct> --export=ALL,N=800,OUT_DIR=data/synthetic scripts/script1.slurm
#
# Produces data/synthetic/{images, manifest_eval.json, train.jsonl, val.jsonl, keys}.
# (This is quick enough to also just run on a login node after activating the env.)
# =============================================================================
#SBATCH --job-name=hips-1-data
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH -p general                 # <- change to your partition if different
#SBATCH -q class                   # <- change to your QoS if different
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-00:20:00

set -uo pipefail
: "${N:=400}"
: "${OUT_DIR:=data/synthetic}"
: "${SEED:=17}"
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

python -m data.generate_synthetic --n "${N}" --out-dir "${OUT_DIR}" --seed "${SEED}"
echo "=== $(date) | dataset written to ${OUT_DIR} ==="
