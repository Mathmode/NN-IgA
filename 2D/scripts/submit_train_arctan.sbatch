#!/bin/bash
#SBATCH --job-name=arctan_train
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/arctan_train_%A_%a.out
#SBATCH --error=logs/arctan_train_%A_%a.err
#
# Train the `arctan` experiment (2D single-patch arctangent front) under the
# validation protocol, for BOTH polynomial degrees p in {2,3}; one SLURM array
# task per random seed (0..3). Warm-start coarse-to-fine continuation (the
# method) is the default.
#
# SUBMIT FROM THE 2D TREE so the relative logs/ path resolves:
#     cd 2D && sbatch scripts/submit_train_arctan.sbatch
#
# Env overrides:
#     P_DEGREES="2 3"             degrees to run in this job   (default: both)
#     CONDA_ENV=p1d               conda env with JAX/NGSolve   (default: p1d)
#     INIT_MODE=warmstart|uniform continuation weight transfer (default: warmstart)
#
# Outputs -> <repo>/data_results/arctan/p<degree>/  (the train_arctan.py default;
# the uniform-init study variant writes to .../arctan/p<degree>/arctan_theta0/).

set -eo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR:-$PWD}"        # the 2D tree (submit from here)
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"   # repository root (holds common/, data_results/)
cd "${EXP_DIR}"
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-p1d}"
if [ -d "$HOME/.conda/envs/${CONDA_ENV}/bin" ]; then
    export PATH="$HOME/.conda/envs/${CONDA_ENV}/bin:$PATH"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
INIT_MODE="${INIT_MODE:-warmstart}"

for P_FE in ${P_DEGREES:-2 3}; do
    echo "=== arctan train seed=${SEED} p=${P_FE} init=${INIT_MODE} on $(hostname) at $(date -Iseconds) ==="
    echo "    protocol=val -> ${REPO_ROOT}/data_results/arctan/p${P_FE}/"
    python -u scripts/train_arctan.py \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --protocol val \
        --init-mode "${INIT_MODE}"
    echo "=== arctan train seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
