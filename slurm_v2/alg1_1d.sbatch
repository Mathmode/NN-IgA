#!/bin/bash
#SBATCH --job-name=nniga_alg1_1d
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/alg1_1d_%A_%a.out
#SBATCH --error=logs/alg1_1d_%A_%a.err
#
# Revision v2, items 3.1 and 3.4: Algorithm 1 for Experiment 1 on Theta_ad
# (1D/scripts/loss_ablation_1d.py), on the cluster nodes, so that T_alg1 and T_train
# come from the same machine.
#   0-7  continuation ladder 2 -> 64, both losses, p = 2, 3, test exponents
#        [9k, 9k+9) of the 67 (sorted), one CSV per chunk
#   8    cold start at N=16 (pilot), three exponents
# The objective is compiled once per (loss, p, N) with beta as an argument
# (1D/src/nonparametric/alg1_ladder.py), so the JIT cache does not grow per instance.
# Output: ${OUT_DIR}/exp1_alg1_ladder_chunk<k>.csv, ${OUT_DIR}/exp1_alg1_cold_N16.csv
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm_v2/alg1_1d.sbatch
# Env: CONDA_ENV (default p1d), PYTHON (default python), SMOKE=1, OUT_DIR
#      (default data_results_v2/loss_ablation_bounded).

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
OUT_DIR="${OUT_DIR:-data_results_v2/loss_ablation_bounded}"
mkdir -p "${OUT_DIR}"
CHUNK=9; LO=$(( CHUNK * TASK )); HI=$(( LO + CHUNK )); ITERS_LADDER=1000; ITERS_PILOT=3000; EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then HI=$(( LO + 1 )); ITERS_LADDER=5; ITERS_PILOT=5; EXTRA="--n-betas 2"; fi

echo "=== alg1_1d task ${TASK} $(date -Iseconds) host=$(hostname) smoke=${SMOKE:-0} ==="
if [ "${TASK}" -lt 8 ]; then
    "${PYTHON}" -u 1D/scripts/loss_ablation_1d.py --part alg1 --iters "${ITERS_LADDER}" --out-dir "${OUT_DIR}" \
        --beta-range "${LO}" "${HI}" --tag "chunk${TASK}"
else
    "${PYTHON}" -u 1D/scripts/loss_ablation_1d.py --part pilot --iters "${ITERS_PILOT}" --out-dir "${OUT_DIR}" ${EXTRA}
fi
echo "=== alg1_1d task ${TASK} done $(date -Iseconds) ==="
