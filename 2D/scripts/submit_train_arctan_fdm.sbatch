#!/bin/bash
#SBATCH --job-name=arctan_fdm_train
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/arctan_fdm_train_%A_%a.out
#SBATCH --error=logs/arctan_fdm_train_%A_%a.err
#
# Train `arctan` with the exact O(N^3) Kronecker / Fast-Diagonalization solver
# (--solver fdm), validation protocol, BOTH degrees p in {2,3}; one array task
# per seed (0..3). The FDM is verified bit-identical to the dense solver in
# forward AND gradient (REPORT_optB_fdm_arctan.md), and a same-seed smoke
# reproduces the dense loss trajectory exactly (REPORT_fdm_production_wiring.md).
#
# ISOLATION (non-negotiable): this writes to a SEPARATE tree
#   <repo>/data_results/arctan_fdm/p<degree>/    (… /arctan_theta0/ under uniform)
# so it NEVER collides with a dense campaign in <repo>/data_results/arctan/.
#
# SUBMIT FROM THE 2D TREE so the relative logs/ path resolves:
#     cd 2D && sbatch scripts/submit_train_arctan_fdm.sbatch
#
# Env overrides:
#     P_DEGREES="2 3"             degrees to run in this job   (default: both)
#     CONDA_ENV=p1d               conda env with JAX           (default: p1d)
#     INIT_MODE=warmstart|uniform continuation weight transfer (default: warmstart)
#
# Run the two campaigns to SEPARATE folders (warmstart is the paper method):
#     cd 2D && sbatch scripts/submit_train_arctan_fdm.sbatch                       # warmstart
#     cd 2D && INIT_MODE=uniform sbatch scripts/submit_train_arctan_fdm.sbatch     # uniform-init ablation

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
    # Build the ISOLATED output dir; mirror train_arctan.py's theta0 convention
    # for the uniform-init study. NEVER under data_results/arctan/.
    BASE="${REPO_ROOT}/data_results/arctan_fdm/p${P_FE}"
    if [ "${INIT_MODE}" = "uniform" ]; then
        OUT_DIR="${BASE}/arctan_theta0"
    else
        OUT_DIR="${BASE}"
    fi
    echo "=== arctan FDM train seed=${SEED} p=${P_FE} init=${INIT_MODE} solver=fdm on $(hostname) at $(date -Iseconds) ==="
    echo "    protocol=val -> ${OUT_DIR}"
    python -u scripts/train_arctan.py \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --protocol val \
        --init-mode "${INIT_MODE}" \
        --solver fdm \
        --output-dir "${OUT_DIR}"
    echo "=== arctan FDM train seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
