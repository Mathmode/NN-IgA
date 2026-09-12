#!/bin/bash
#SBATCH --job-name=extrap_eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# EXTRAPOLATION STUDY -- evaluation arm.
# Evaluate the single-level-supervised density (trained by submit_extrap_train
# at SUP_LEVEL) at ALL mesh levels {4,8,16,32,64}. Because the density network
# is N-independent, this samples the SAME trained density at every level; the
# N=64 column is the extrapolation we care about (it is beyond every trained
# supervision level 8/16/32). One array task per seed (0..3).
#
# It reads the final checkpoint written by the matching training run:
#   <repo>/data_results/extrap_study/only_N<SUP_LEVEL>/p2/checkpoints/seed<seed>/checkpoint_final.npz
# and writes the per-seed summary CSV (uniform / positional rows per (sigma,N)):
#   <repo>/data_results/extrap_study/only_N<SUP_LEVEL>/p2/eval/summary_p2_seed<seed>.csv
#
# Corrector-free by default (CORR=0), matching how the warmstart control was
# evaluated under --protocol val, so the factor is computed on identical terms.
#
# SUBMIT FROM THE 2D TREE, once per supervision level, AFTER its training array
# finished (set SUP_LEVEL):
#     cd 2D
#     SUP_LEVEL=8  sbatch --job-name=extrap_eval_N8  scripts/submit_extrap_eval.sbatch
#     SUP_LEVEL=16 sbatch --job-name=extrap_eval_N16 scripts/submit_extrap_eval.sbatch
#     SUP_LEVEL=32 sbatch --job-name=extrap_eval_N32 scripts/submit_extrap_eval.sbatch
#
# Env overrides:
#     SUP_LEVEL=8|16|32   supervision level to evaluate (REQUIRED; fail loud)
#     CONDA_ENV=p1d       conda env with JAX/NGSolve    (default: p1d)
#     P_FE=2              FE degree                      (default: 2 -- study scope)
#     CORR=0              --corrector-max-iter           (default 0 = corrector-free)
#     FORCE=0             set FORCE=1 to re-evaluate seeds whose summary exists

set -eo pipefail

: "${SUP_LEVEL:?set SUP_LEVEL=8|16|32 before sbatch (e.g. SUP_LEVEL=8 sbatch ...)}"
case "${SUP_LEVEL}" in
    8|16|32) ;;
    *) echo "ERROR: SUP_LEVEL must be 8, 16, or 32 (got '${SUP_LEVEL}')"; exit 1 ;;
esac

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
P_FE="${P_FE:-2}"
CORR="${CORR:-0}"
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

DATA_DIR="${REPO_ROOT}/data_results/extrap_study/only_N${SUP_LEVEL}/p${P_FE}"
CKPT="${DATA_DIR}/checkpoints/seed${SEED}/checkpoint_final.npz"
OUTPUT_DIR="${DATA_DIR}/eval"

echo "=== extrap eval: supervise N=${SUP_LEVEL} -> eval {4,8,16,32,64} | seed=${SEED} p=${P_FE} corr=${CORR} on $(hostname) at $(date -Iseconds) ==="
echo "    checkpoint=${CKPT}  ->  ${OUTPUT_DIR}"
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT} (run submit_extrap_train.sbatch with SUP_LEVEL=${SUP_LEVEL} first)"; exit 1; }
python -u scripts/eval_arctan.py \
    --seed "${SEED}" \
    --p "${P_FE}" \
    --checkpoint "${CKPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --N-levels 4 8 16 32 64 \
    --corrector-max-iter "${CORR}" \
    ${FORCE_FLAG}
echo "=== extrap eval: supervise N=${SUP_LEVEL} seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
