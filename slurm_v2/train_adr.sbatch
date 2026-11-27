#!/bin/bash
#SBATCH --job-name=nniga_adr
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/adr_%A_%a.out
#SBATCH --error=logs/adr_%A_%a.err
#
# Revision v2, item 3.2: Experiment 6 (1D advection-diffusion-reaction).
# 8 tasks = p in {2,3} x seeds 0..3 -> data_results_v2/adr/p<p>/.
# Submit from the repository root:  mkdir -p logs && sbatch slurm_v2/train_adr.sbatch
# Local run of one task:            TASK=3 PYTHON=python bash slurm_v2/train_adr.sbatch
# Env: CONDA_ENV (default p1d), PYTHON (default python), SMOKE=1, OUT_ROOT.

set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
mkdir -p logs
CONDA_ENV="${CONDA_ENV:-p1d}"
if [ -d "$HOME/.conda/envs/${CONDA_ENV}/bin" ]; then
    export PATH="$HOME/.conda/envs/${CONDA_ENV}/bin:$PATH"
fi
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

TASK="${TASK:-${SLURM_ARRAY_TASK_ID:-0}}"
P=$(( 2 + TASK / 4 ))
SEED=$(( TASK % 4 ))
SMOKE_FLAG=""
[ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
OUT="${OUT_ROOT:-data_results_v2}/adr/p${P}"
echo "=== adr task ${TASK}: p=${P} seed=${SEED} $(date -Iseconds) host=$(hostname) ==="
"${PYTHON}" -u 1D/scripts/train_adr.py --p "${P}" --seed "${SEED}" --output-dir "${OUT}" ${SMOKE_FLAG}
echo "=== adr task ${TASK} done $(date -Iseconds) ==="
