#!/bin/bash
#SBATCH --job-name=advdiff_eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/advdiff_eval_%A_%a.out
#SBATCH --error=logs/advdiff_eval_%A_%a.err
#
# Evaluate the `advdiff` experiment on the held-out test split, for BOTH degrees
# p in {2,3}; one SLURM array task per seed (0..3). Reads the final checkpoint
# written by submit_train_advdiff.sbatch and writes the per-seed summary CSV.
#
# SUBMIT FROM THE 2D TREE (after the matching train job finished):
#     cd 2D && sbatch scripts/submit_eval_advdiff.sbatch
#
# Env overrides:
#     P_DEGREES="2 3"  degrees to evaluate            (default: both)
#     CONDA_ENV=p1d    conda env with JAX/NGSolve     (default: p1d)
#     CORR=0           --corrector-max-iter (default 0 = corrector-free)
#     FORCE=0          set FORCE=1 to re-evaluate seeds whose summary already exists
#
# Reads  <repo>/data_results/advdiff/p<degree>/checkpoints/seed<seed>/checkpoint_final.npz
# Writes <repo>/data_results/advdiff/p<degree>/eval/

set -eo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR:-$PWD}"        # the 2D tree (submit from here)
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
CORR="${CORR:-0}"
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

for P_FE in ${P_DEGREES:-2 3}; do
    DATA_DIR="${REPO_ROOT}/data_results/advdiff/p${P_FE}"
    CKPT="${DATA_DIR}/checkpoints/seed${SEED}/checkpoint_final.npz"
    OUTPUT_DIR="${DATA_DIR}/eval"
    echo "=== advdiff eval seed=${SEED} p=${P_FE} corr=${CORR} on $(hostname) at $(date -Iseconds) ==="
    echo "    checkpoint=${CKPT}  ->  ${OUTPUT_DIR}"
    [ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT} (run submit_train_advdiff.sbatch first)"; exit 1; }
    python -u scripts/eval_advdiff.py \
        --protocol val \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --checkpoint "${CKPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --N-levels 4 8 16 32 64 \
        --corrector-max-iter "${CORR}" \
        ${FORCE_FLAG}
    echo "=== advdiff eval seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
