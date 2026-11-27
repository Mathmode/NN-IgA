#!/bin/bash
#SBATCH --job-name=nniga_2d_ritz
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ritz2d_%A_%a.out
#SBATCH --error=logs/ritz2d_%A_%a.err
#
# Revision v2, item 3.1: networks trained on the normalized Ritz energy J/|J_unif|
# for Experiments 3 (arctan) and 4 (L-shape), with the protocol of the paper runs
# (recorded in data_results/*/p*/meta_train_*.json): val protocol, levels 4 8 16 32,
# lr 1e-3 -> 1e-4, batch 16. Only the loss changes (and es_tol = 0, see
# 2D/scripts/train_*.py --loss ritz).
#   tasks 0-7   arctan  p in {2,3} x seeds 0..3 -> data_results_v2/arctan_ritz/p<p>/
#   tasks 8-15  lshape  p in {2,3} x seeds 0..3 -> data_results_v2/lshape_ritz/p<p>/
# Expected cost: like the eta runs (arctan 7-14 h, lshape 1-3 h per task).
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm_v2/train_2d_ritz.sbatch
# Local run of one task:            TASK=8 SMOKE=1 PYTHON=python bash slurm_v2/train_2d_ritz.sbatch
# Env: CONDA_ENV (default p1d), PYTHON (default python), SMOKE=1, OUT_ROOT.

set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
mkdir -p logs
CONDA_ENV="${CONDA_ENV:-p1d}"
if [ -d "$HOME/.conda/envs/${CONDA_ENV}/bin" ]; then
    export PATH="$HOME/.conda/envs/${CONDA_ENV}/bin:$PATH"
fi
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

TASK="${TASK:-${SLURM_ARRAY_TASK_ID:-0}}"
if [ "${TASK}" -lt 8 ]; then EXP=arctan; else EXP=lshape; fi
P=$(( 2 + (TASK % 8) / 4 ))
SEED=$(( TASK % 4 ))
OUT="${OUT_ROOT:-data_results_v2}/${EXP}_ritz/p${P}"
EXTRA=""
LEVELS="4 8 16 32"
if [ "${SMOKE:-0}" = "1" ]; then EXTRA="--smoke --es-max-epochs 5"; LEVELS="4 8"; fi

echo "=== ritz2d task ${TASK}: ${EXP} p=${P} seed=${SEED} $(date -Iseconds) host=$(hostname) ==="
"${PYTHON}" -u "2D/scripts/train_${EXP}.py" --seed "${SEED}" --p "${P}" --protocol val --loss ritz \
    --N-levels ${LEVELS} --lr-init 1e-3 --lr-end 1e-4 --output-dir "${OUT}" ${EXTRA}
echo "=== ritz2d task ${TASK} done $(date -Iseconds) ==="
