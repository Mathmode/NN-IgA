#!/bin/bash
#SBATCH --job-name=helm_denom
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/helm_denom_%A_%a.out
#SBATCH --error=logs/helm_denom_%A_%a.err
#
# DENOMINATOR STUDY for the `helmholtz` experiment: re-train from scratch with ONE
# of three loss denominators, everything else IDENTICAL to production (contrast
# curriculum, warm phase A, val-protocol early-stopping, LR schedule, batch, seeds,
# levels, epochs). The ONLY change is --loss-denom. One (denom, degree) pair per
# job; one SLURM array task per seed (0..3).
#
#   DENOM=uniform : eta^2(theta_unif^{(N)}; c)  -- the eq-26 production denominator
#   DENOM=exact   : |u*_c|^2_{H1} (analytic)     -- the previous (stable) denominator
#   DENOM=robust  : uniform, clipped at the per-level train p90 (val shares the thr)
#
# PARAMETRIZED: expects DENOM and P_FE in the environment (run_denom_study.sh sets
# them and submits the 6 = 3 denom x 2 deg jobs):
#     DENOM = uniform | exact | robust
#     P_FE  = 2 | 3
#
# Output -> <repo>/data_results/helmholtz_denom_${DENOM}/p${P_FE}/  -- a SEPARATE
# tree; the production data_results/helmholtz/ is NEVER written.
#
# SUBMIT FROM THE 1D TREE (the launcher does this):
#     cd 1D && DENOM=robust P_FE=3 sbatch scripts/submit_denom_study.sbatch
#
# Env overrides:
#     CONDA_ENV=p1d   conda env with JAX   (default: p1d)

set -eo pipefail

: "${DENOM:?set DENOM=uniform|exact|robust (run_denom_study.sh does this)}"
: "${P_FE:?set P_FE=2|3 (the FE degree)}"
case "${DENOM}" in uniform|exact|robust) ;; *) echo "bad DENOM=${DENOM} (uniform|exact|robust)"; exit 2;; esac

EXP_DIR="${SLURM_SUBMIT_DIR:-$PWD}"        # the 1D tree (submit from here)
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
OUT="${REPO_ROOT}/data_results/helmholtz_denom_${DENOM}/p${P_FE}"

echo "=== helmholtz denom-study denom=${DENOM} p=${P_FE} seed=${SEED} on $(hostname) at $(date -Iseconds) ==="
echo "    protocol=val  loss_denom=${DENOM}  ->  ${OUT}/"
python -u scripts/train_helmholtz.py \
    --p "${P_FE}" \
    --seed "${SEED}" \
    --protocol val \
    --device cpu \
    --loss-denom "${DENOM}" \
    --output-dir "${OUT}"
echo "=== denom=${DENOM} p=${P_FE} seed=${SEED} DONE at $(date -Iseconds) ==="
