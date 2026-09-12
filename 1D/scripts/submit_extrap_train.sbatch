#!/bin/bash
#SBATCH --job-name=helm_extrap_train
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/helm_extrap_train_%A_%a.out
#SBATCH --error=logs/helm_extrap_train_%A_%a.err
#
# Arm B of the contrast-EXTRAPOLATION study (exploratory; NOT production). Trains the
# helmholtz network FLAT (no contrast curriculum, warm_epochs=0) on LOW contrasts only
# (c < c_split), keeping the coarse-to-fine continuation in N. For BOTH degrees
# p in {2,3}; one array task per seed (0..3). Output goes ONLY to the *_extrap tree;
# production data_results/helmholtz/ is never written.
#
# SUBMIT FROM THE 1D TREE:
#     cd 1D && sbatch scripts/submit_extrap_train.sbatch
#
# Env overrides:
#     P_DEGREES="2 3"  degrees to run   (default: both)
#     CONDA_ENV=p1d    conda env        (default: p1d)
#
# Writes -> <repo>/data_results/helmholtz_extrap/p<degree>/

set -eo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR:-$PWD}"        # the 1D tree (submit from here)
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"   # repository root
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

for P_FE in ${P_DEGREES:-2 3}; do
    OUT="${REPO_ROOT}/data_results/helmholtz_extrap/p${P_FE}"
    echo "=== arm B (extrapolation) train seed=${SEED} p=${P_FE} on $(hostname) at $(date -Iseconds) ==="
    echo "    flat low-c training (warm_epochs=0) -> ${OUT}/"
    python -u scripts/train_helmholtz_extrap.py \
        --p "${P_FE}" \
        --seed "${SEED}" \
        --device cpu \
        --output-dir "${OUT}"
    echo "=== arm B train seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
