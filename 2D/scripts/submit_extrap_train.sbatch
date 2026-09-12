#!/bin/bash
#SBATCH --job-name=extrap_train
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# EXTRAPOLATION STUDY -- training arm.
# Train an `arctan` density supervised at a SINGLE mesh level (SUP_LEVEL),
# with --init-mode uniform, --protocol val, p=2; one array task per seed (0..3).
# The density network is N-independent, so a density supervised at N=SUP_LEVEL
# can later be evaluated at ANY level {4,8,16,32,64} (see submit_extrap_eval).
#
# Scientific question: how much SUPERVISION resolution does the density need to
# extrapolate to N=64? Coarse supervision (N=8) sees a blurry estimator (the
# N=8 mesh resolves the arctan front poorly); fine supervision (N=32) sees a
# sharp estimator. This arm trains the densities; the eval arm measures the
# N=64 factor vs the warmstart control (4.74 for p2).
#
# WHY this is a FAIR experiment: under --protocol val every level gets the SAME
# max_epochs cap and the SAME val-eta2 plateau early-stop (config.TRAIN_VAL), so
# "only N=8" and "only N=32" receive comparable optimisation budget -- the ONLY
# difference is the supervision resolution. (train_arctan.py L213-216.)
#
# NOTE on uniform-init at a SINGLE level: the per-level theta0 re-init in the
# continuation loop is guarded by `idx > start_idx`, so for a single level it
# never fires -- warmstart and uniform are byte-identical here. The point of
# --init-mode uniform in THIS study is only the (default) isolated output dir
# convention; we override the dir explicitly below anyway. Each SUP_LEVEL run
# trains ONE level from the seed's LeCun-random init, which is exactly the
# "single-level supervision" we want.
#
# ISOLATION (non-negotiable): writes ONLY to
#   <repo>/data_results/extrap_study/only_N<SUP_LEVEL>/p2/
# It NEVER touches the validated Branch-A tree <repo>/data_results/arctan/.
#
# SUBMIT FROM THE 2D TREE, once per supervision level (set SUP_LEVEL):
#     cd 2D
#     SUP_LEVEL=8  sbatch --job-name=extrap_train_N8  scripts/submit_extrap_train.sbatch
#     SUP_LEVEL=16 sbatch --job-name=extrap_train_N16 scripts/submit_extrap_train.sbatch
#     SUP_LEVEL=32 sbatch --job-name=extrap_train_N32 scripts/submit_extrap_train.sbatch
# (The --job-name override tags both the SLURM name and the log files, since
#  #SBATCH lines cannot read the SUP_LEVEL env var. Logs -> logs/<name>_%A_%a.*)
#
# Env overrides:
#     SUP_LEVEL=8|16|32   supervision level (REQUIRED; no default -- fail loud)
#     CONDA_ENV=p1d       conda env with JAX           (default: p1d)
#     P_FE=2              FE degree                     (default: 2 -- study scope)

set -eo pipefail

# --- SUP_LEVEL is mandatory and constrained: a wrong value would silently
#     train/write the wrong folder, so fail loudly instead. ---
: "${SUP_LEVEL:?set SUP_LEVEL=8|16|32 before sbatch (e.g. SUP_LEVEL=8 sbatch ...)}"
case "${SUP_LEVEL}" in
    8|16|32) ;;
    *) echo "ERROR: SUP_LEVEL must be 8, 16, or 32 (got '${SUP_LEVEL}')"; exit 1 ;;
esac

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
P_FE="${P_FE:-2}"

# ISOLATED output dir -- one tree per supervision level. NEVER data_results/arctan/.
OUT_DIR="${REPO_ROOT}/data_results/extrap_study/only_N${SUP_LEVEL}/p${P_FE}"

echo "=== extrap train: supervise N=${SUP_LEVEL} only | seed=${SEED} p=${P_FE} init=uniform proto=val on $(hostname) at $(date -Iseconds) ==="
echo "    -> ${OUT_DIR}"
python -u scripts/train_arctan.py \
    --seed "${SEED}" \
    --p "${P_FE}" \
    --protocol val \
    --init-mode uniform \
    --N-levels "${SUP_LEVEL}" \
    --output-dir "${OUT_DIR}"
echo "=== extrap train: supervise N=${SUP_LEVEL} seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
echo "    checkpoint -> ${OUT_DIR}/checkpoints/seed${SEED}/checkpoint_final.npz"
