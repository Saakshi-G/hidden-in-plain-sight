#!/usr/bin/env bash
# =============================================================================
# STEP 0 (run ONCE, on a Sol LOGIN node -- NOT via sbatch; needs internet).
# Builds the shared Python env (hips-vllm) every job script uses.
#
#   bash scripts/script0_setup.sh
#
# Then log in to Weights & Biases (or export WANDB_API_KEY in ~/.bashrc):
#   module load mamba && source activate hips-vllm && python -m wandb login
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/setup_sol.sh "$@"
