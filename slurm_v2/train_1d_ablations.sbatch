#!/bin/bash
#SBATCH --job-name=nniga_1d_abl
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/abl1d_%A_%a.out
#SBATCH --error=logs/abl1d_%A_%a.err
#
# Revision v2, items 3.1 and 3.6 (1D trainings). 24 tasks = 3 variants x p in {2,3} x seeds 0..3:
#   0-7    Exp. 1, Ritz loss            -> data_results_v2/singular_ritz/p<p>
#   8-15   Exp. 1, no log(xi+1/N) input -> data_results_v2/singular_nolog/p<p>
#   16-23  Exp. 2, no h_max cap         -> data_results_v2/helmholtz_nohmax/p<p> (+ eval)
# Each task takes ~5-15 min on one CPU node. Evaluation of the Exp. 1 networks is
# done afterwards by 1D/scripts/loss_ablation_1d.py --part networks.
#
# Submit from the repository root:   mkdir -p logs && sbatch slurm_v2/train_1d_ablations.sbatch
# Local run of one task:             TASK=3 PYTHON=python bash slurm_v2/train_1d_ablations.sbatch
# Env: CONDA_ENV (default p1d), PYTHON (default python), SMOKE=1 (reduced run).

set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
mkdir -p logs
CONDA_ENV="${CONDA_ENV:-p1d}"
if [ -d "$HOME/.conda/envs/${CONDA_ENV}/bin" ]; then
    export PATH="$HOME/.conda/envs/${CONDA_ENV}/bin:$PATH"
fi
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

TASK="${TASK:-${SLURM_ARRAY_TASK_ID:-0}}"
VARIANT=$(( TASK / 8 ))
P=$(( 2 + (TASK % 8) / 4 ))
SEED=$(( TASK % 4 ))
SMOKE_FLAG=""
[ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"

echo "=== task ${TASK}: variant=${VARIANT} p=${P} seed=${SEED} smoke=${SMOKE:-0} $(date -Iseconds) host=$(hostname) ==="
case "${VARIANT}" in
  0) OUT="${OUT_ROOT:-data_results_v2}/singular_ritz/p${P}"
     "${PYTHON}" -u 1D/scripts/train_singular.py --p "${P}" --seed "${SEED}" --protocol val \
         --loss ritz --eval-levels --output-dir "${OUT}" ${SMOKE_FLAG} ;;
  1) OUT="${OUT_ROOT:-data_results_v2}/singular_nolog/p${P}"
     "${PYTHON}" -u 1D/scripts/train_singular.py --p "${P}" --seed "${SEED}" --protocol val \
         --no-log-feature --eval-levels --output-dir "${OUT}" ${SMOKE_FLAG} ;;
  2) OUT="${OUT_ROOT:-data_results_v2}/helmholtz_nohmax/p${P}"
     "${PYTHON}" -u 1D/scripts/train_helmholtz.py --p "${P}" --seed "${SEED}" --protocol val \
         --no-hmax --output-dir "${OUT}" ${SMOKE_FLAG}
     LEVELS="32 48 64 96 128 256"
     [ "${SMOKE:-0}" = "1" ] && LEVELS="32"
     "${PYTHON}" -u 1D/scripts/eval_helmholtz.py --p "${P}" --seed "${SEED}" --no-hmax --force \
         --checkpoint "${OUT}/checkpoints/p${P}_seed${SEED}/checkpoint_final.npz" \
         --test-cs "${OUT}/test_cs_p${P}_seed${SEED}.npy" --N-levels ${LEVELS} \
         --output-dir "${OUT}/eval" ;;
esac
echo "=== task ${TASK} done $(date -Iseconds) ==="
