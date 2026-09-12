#!/bin/bash
#SBATCH --job-name=lshape_train
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/lshape_train_%A_%a.out
#SBATCH --error=logs/lshape_train_%A_%a.err
#
# Train the `lshape` experiment (2D two-patch L-shape re-entrant corner) under
# the validation protocol, for BOTH polynomial degrees p in {2,3}; one SLURM
# array task per random seed (0..3). Warm-start coarse-to-fine continuation
# (the method) is the default.
#
# SUBMIT FROM THE 2D TREE so the relative logs/ path resolves:
#     cd 2D && sbatch scripts/submit_train_lshape.sbatch
#
# Env overrides:
#     P_DEGREES="2 3"             degrees to run in this job   (default: both)
#     CONDA_ENV=p1d               conda env with JAX/NGSolve   (default: p1d)
#     INIT_MODE=warmstart|uniform continuation weight transfer (default: warmstart)
#     H1_REF_CACHE=<path>         immersed IGA self-reference for the H¹ training
#                                 monitor (default: references/reference_lshape.pkl).
#                                 Optional: if absent the monitor column is NaN and
#                                 training is unaffected.
#
# Outputs -> <repo>/data_results/lshape/p<degree>/  (the train_lshape.py default;
# the uniform-init study variant writes to .../lshape/p<degree>/lshape_theta0/).

set -eo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR:-$PWD}"        # the 2D tree (submit from here)
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"   # repository root (holds common/, references/, data_results/)
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
H1_REF_CACHE="${H1_REF_CACHE:-${REPO_ROOT}/references/reference_lshape.pkl}"

for P_FE in ${P_DEGREES:-2 3}; do
    echo "=== lshape train seed=${SEED} p=${P_FE} init=${INIT_MODE} on $(hostname) at $(date -Iseconds) ==="
    echo "    protocol=val  h1-ref=${H1_REF_CACHE}  -> ${REPO_ROOT}/data_results/lshape/p${P_FE}/"
    python -u scripts/train_lshape.py \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --protocol val \
        --init-mode "${INIT_MODE}" \
        --h1-ref-cache "${H1_REF_CACHE}"
    echo "=== lshape train seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
