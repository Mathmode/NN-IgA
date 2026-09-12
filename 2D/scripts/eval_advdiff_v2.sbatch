#!/bin/bash
#SBATCH --job-name=advdiff_v2_eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/advdiff_v2_eval_%A_%a.out
#SBATCH --error=logs/advdiff_v2_eval_%A_%a.err
#
# Evaluate the OFFICIAL advdiff_v2 run on the held-out test split, both degrees
# p in {2,3}, one SLURM array task per seed (0..3). Reads the final checkpoint
# written by train_advdiff_v2.sbatch and writes the per-seed summary_*.csv the
# figures consume, at evaluation levels N = 4 8 16 32 64  (N=64 is extrapolation
# beyond the training range 4..32).
#
# --protocol val so the test split matches the training split (SPLIT_SEED, 70/15/15).
# Corrector-free by default (CORR=0), as in the production eval; only the 'uniform'
# and 'positional' methods feed the figures. Reads/writes ONLY under advdiff_v2/;
# data_results/advdiff/ is never touched.
#
# SUBMIT FROM THE 2D TREE, AFTER train_advdiff_v2 finished (chain with afterok):
#     cd 2D && sbatch --dependency=afterok:<train_jobid> scripts/eval_advdiff_v2.sbatch
#
# Reads  <repo>/data_results/advdiff_v2/p<degree>/checkpoints/seed<seed>/checkpoint_final.npz
# Writes <repo>/data_results/advdiff_v2/p<degree>/eval/
# Env overrides:
#     P_DEGREES="2 3"  degrees to evaluate         (default: both)
#     CONDA_ENV=p1d    conda env with JAX          (default: p1d)
#     CORR=0           --corrector-max-iter        (default 0 = corrector-free)
#     FORCE=0          FORCE=1 re-evaluates seeds whose summary already exists

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
    DATA_DIR="${REPO_ROOT}/data_results/advdiff_v2/p${P_FE}"
    CKPT="${DATA_DIR}/checkpoints/seed${SEED}/checkpoint_final.npz"
    OUTPUT_DIR="${DATA_DIR}/eval"
    echo "=== advdiff_v2 eval seed=${SEED} p=${P_FE} corr=${CORR} on $(hostname) at $(date -Iseconds) ==="
    echo "    checkpoint=${CKPT}  ->  ${OUTPUT_DIR}"
    [ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT} (run train_advdiff_v2.sbatch first)"; exit 1; }
    python -u scripts/eval_advdiff.py \
        --protocol val \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --checkpoint "${CKPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --N-levels 4 8 16 32 64 \
        --corrector-max-iter "${CORR}" \
        ${FORCE_FLAG}
    echo "=== advdiff_v2 eval seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
