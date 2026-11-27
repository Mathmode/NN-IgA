#!/bin/bash
#SBATCH --job-name=nniga_alg1_2d
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/alg1_2d_%A_%a.out
#SBATCH --error=logs/alg1_2d_%A_%a.err
#
# Revision v2, items 3.1 and 3.4: Algorithm 1 in 2D with the continuation ladder
# 4 -> 8 -> 16 -> 32 and both losses (eta^2 and Ritz), plus the trained networks
# (eta from data_results/, Ritz from data_results_v2/*_ritz/ once available) at the
# same levels and instances. 16 tasks = {arctan, lshape} x p in {2,3} x 4 chunks
# of 4 instances (16 spread test parameters per experiment and degree).
# Output: data_results_v2/alg1_2d/<exp>_p<p>_chunk<k>.csv
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm_v2/alg1_2d.sbatch
# (after slurm_v2/train_2d_ritz.sbatch to include the Ritz networks).
# Local run of one task:            TASK=0 SMOKE=1 PYTHON=python bash slurm_v2/alg1_2d.sbatch
# Env: CONDA_ENV (default p1d), PYTHON (default python), SMOKE=1, OUT_DIR, MAXITER (default 200).

set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
mkdir -p logs
CONDA_ENV="${CONDA_ENV:-p1d}"
if [ -d "$HOME/.conda/envs/${CONDA_ENV}/bin" ]; then
    export PATH="$HOME/.conda/envs/${CONDA_ENV}/bin:$PATH"
fi
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

TASK="${TASK:-${SLURM_ARRAY_TASK_ID:-0}}"
if [ "${TASK}" -lt 8 ]; then EXP=arctan; else EXP=lshape; fi
P=$(( 2 + (TASK % 8) / 4 ))
CHUNK=$(( TASK % 4 ))
LO=$(( 4 * CHUNK )); HI=$(( LO + 4 ))
LEVELS="4 8 16 32"; MAXITER="${MAXITER:-200}"
if [ "${SMOKE:-0}" = "1" ]; then LEVELS="4 8"; MAXITER=5; LO=0; HI=1; fi

echo "=== alg1_2d task ${TASK}: ${EXP} p=${P} instances [${LO},${HI}) $(date -Iseconds) host=$(hostname) ==="
"${PYTHON}" -u 2D/scripts/alg1_2d.py --experiment "${EXP}" --p "${P}" --levels ${LEVELS} \
    --maxiter "${MAXITER}" --instances 16 --instance-range "${LO}" "${HI}" \
    --losses eta ritz --networks eta ritz --tag "chunk${CHUNK}" --out-dir "${OUT_DIR:-data_results_v2/alg1_2d}"
echo "=== alg1_2d task ${TASK} done $(date -Iseconds) ==="
