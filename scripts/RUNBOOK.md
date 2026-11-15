# Run on Sol — step-by-step (for the team)

Numbered scripts, run in order. **Submit under your own account** so it uses your
fairshare: add `-A <your_account>` to each `sbatch`. If your partition/QoS differ
from the defaults (`-p general -q class`), edit the two `#SBATCH` lines at the top
of each script, or pass `sbatch -p <part> -q <qos> ...`.

| Step | Command | Where | GPU? |
|------|---------|-------|------|
| 0. Setup (once) | `bash scripts/script0_setup.sh` | login node | no |
| — W&B login | `module load mamba && source activate hips-vllm && python -m wandb login` | login node | no |
| 1. Make dataset | `sbatch -A <acct> scripts/script1.slurm` | batch | no |
| 2. Fine-tune | `sbatch -A <acct> --export=ALL,MAX_STEPS=2 scripts/script2.slurm` (smoke) then without MAX_STEPS | batch | yes |
| 3. Experiment | `sbatch -A <acct> scripts/script3.slurm` | batch | yes |
| 4. Metrics | `sbatch -A <acct> scripts/script4.slurm` | batch | no |

Notes:
- **Step 0 must run on a login node** (it installs packages and needs internet);
  it is NOT an sbatch job. Everyone runs it once to build their own `hips-vllm` env.
- **Do the Step 2 smoke test first** (`MAX_STEPS=2`) — the training script hasn't
  been run on a GPU yet.
- Step 3 runs the **base** model by default. To evaluate the Step 2 fine-tune,
  merge the LoRA adapter and pass `MODEL=<merged_dir>` (see script3 header).
- Offline W&B: after a GPU job, `wandb sync results/wandb/offline-run-*` from a login node.

Monitor / manage:
```bash
squeue --me
tail -f results/slurm-<jobid>.out      # job log
tail -f results/vllm-<jobid>.log       # model-server log (step 3)
scancel <jobid>
```
