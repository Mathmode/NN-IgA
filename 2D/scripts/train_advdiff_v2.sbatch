#!/bin/bash
#SBATCH --job-name=advdiff_v2_train
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/advdiff_v2_train_%A_%a.out
#SBATCH --error=logs/advdiff_v2_train_%A_%a.err
#
# OFFICIAL advdiff (P5) re-training -> data_results/advdiff_v2/  (NEW folder; the
# existing data_results/advdiff/ is left UNTOUCHED for comparison).
#
# Protocol = val (identical to the paper's arctan/lshape): deterministic 70/15/15
# split (SPLIT_SEED), per-level early stopping on the held-out validation eta^2,
# best-val checkpoint. Coarse-to-fine warm-start continuation (weights inherited
# across levels) is the method (--init-mode warmstart). Both degrees p in {2,3};
# one SLURM array task per seed (0..3) -> 8 configurations total.
#
# Learning rate: forced to lr_init=1e-3 -> lr_end=1e-4 via --lr-init/--lr-end
#   (the config default is 1e-2; these flags override ONLY this run, the config
#   is not modified, and every other experiment/invocation is unchanged).
# Schedule: the project-wide warmup -> exponential-decay over each level's epochs
#   (shared train_one_level; same schedule the other experiments use).
# Loss denominator: eta^2_unif (config.LOSS_NORM = "uniform_same_level").
# Evaluation levels are applied later by eval_advdiff_v2.sbatch (N up to 64).
#
# SUBMIT FROM THE 2D TREE so logs/ and REPO_ROOT resolve:
#     cd 2D && sbatch scripts/train_advdiff_v2.sbatch
#
# Env overrides:
#     P_DEGREES="2 3"             degrees to run in each task (default: both)
#     CONDA_ENV=p1d               conda env with JAX          (default: p1d)
#     INIT_MODE=warmstart|uniform continuation weight transfer(default: warmstart)

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
    OUT_DIR="${REPO_ROOT}/data_results/advdiff_v2/p${P_FE}"
    # safety: never write outside the advdiff_v2 tree (protects data_results/advdiff/)
    case "${OUT_DIR}" in
        *"/data_results/advdiff_v2/"*) : ;;
        *) echo "ERROR: output '${OUT_DIR}' is not under advdiff_v2/ -- refusing"; exit 1 ;;
    esac
    echo "=== advdiff_v2 train seed=${SEED} p=${P_FE} init=${INIT_MODE} on $(hostname) at $(date -Iseconds) ==="
    echo "    protocol=val  lr=1e-3->1e-4  levels=4,8,16,32  ->  ${OUT_DIR}"
    python -u scripts/train_advdiff.py \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --protocol val \
        --init-mode "${INIT_MODE}" \
        --N-levels 4 8 16 32 \
        --batch-size 16 \
        --output-dir "${OUT_DIR}"
    echo "=== advdiff_v2 train seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
