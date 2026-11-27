#!/bin/bash
#SBATCH --job-name=nniga_meshdiag
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/meshdiag_%A_%a.out
#SBATCH --error=logs/meshdiag_%A_%a.err
#
# Revision v2, items 3.4 and 3.5: realized mesh constants, conditioning and online
# time (network forward pass + Galerkin solve) of the trained networks, on the
# cluster nodes, so that T_online comes from the same machine as T_train.
#   0  1D/scripts/mesh_diagnostics_1d.py   (Exps. 1, 2, 6)
#   1  2D/scripts/mesh_diagnostics_2d.py   (Exps. 3, 4, 5)
# Output: ${OUT_DIR}/{constants,conditioning_online}_{1d,2d}*.csv
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm_v2/mesh_diagnostics.sbatch
# Env: CONDA_ENV (default p1d), PYTHON (default python), SMOKE=1, OUT_DIR
#      (default data_results_v2/mesh_diagnostics).

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
OUT_DIR="${OUT_DIR:-data_results_v2/mesh_diagnostics}"
A1=""; A2=""
if [ "${SMOKE:-0}" = "1" ]; then A1="--seeds 0 --cond-levels 16"; A2="--experiments lshape --n-params 1 --seeds 0 --cond-levels 16 --cond-params 1"; fi

echo "=== meshdiag task ${TASK} $(date -Iseconds) host=$(hostname) smoke=${SMOKE:-0} ==="
if [ "${TASK}" = "0" ]; then
    "${PYTHON}" -u 1D/scripts/mesh_diagnostics_1d.py --out-dir "${OUT_DIR}" ${A1}
else
    "${PYTHON}" -u 2D/scripts/mesh_diagnostics_2d.py --out-dir "${OUT_DIR}" ${A2}
fi
echo "=== meshdiag task ${TASK} done $(date -Iseconds) ==="
