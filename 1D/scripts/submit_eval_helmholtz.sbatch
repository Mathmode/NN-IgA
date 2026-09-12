#!/bin/bash
#SBATCH --job-name=helmholtz_eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/helmholtz_eval_%A_%a.out
#SBATCH --error=logs/helmholtz_eval_%A_%a.err
#
# Evaluate the `helmholtz` experiment on the held-out test contrasts, for BOTH
# degrees p in {2,3}; one SLURM array task per seed (0..3). Reads the final
# checkpoint and the saved test-contrast split written by
# submit_train_helmholtz.sbatch, and writes the per-seed summary CSV.
#
# SUBMIT FROM THE 1D TREE (after the matching train job finished):
#     cd 1D && sbatch scripts/submit_eval_helmholtz.sbatch
#
# Env overrides:
#     P_DEGREES="2 3"  degrees to evaluate        (default: both)
#     CONDA_ENV=p1d    conda env with JAX         (default: p1d)
#     FORCE=0          set FORCE=1 to re-evaluate seeds whose summary already exists
#
# Reads  <repo>/data_results/helmholtz/p<degree>/checkpoints/p<degree>_seed<seed>/checkpoint_final.npz
#        <repo>/data_results/helmholtz/p<degree>/test_cs_p<degree>_seed<seed>.npy
# Writes <repo>/data_results/helmholtz/p<degree>/eval/

set -eo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR:-$PWD}"        # the 1D tree (submit from here)
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
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

for P_FE in ${P_DEGREES:-2 3}; do
    DATA_DIR="${REPO_ROOT}/data_results/helmholtz/p${P_FE}"
    CKPT="${DATA_DIR}/checkpoints/p${P_FE}_seed${SEED}/checkpoint_final.npz"
    TEST_CS="${DATA_DIR}/test_cs_p${P_FE}_seed${SEED}.npy"
    OUTPUT_DIR="${DATA_DIR}/eval"
    echo "=== helmholtz eval seed=${SEED} p=${P_FE} on $(hostname) at $(date -Iseconds) ==="
    echo "    checkpoint=${CKPT}"
    echo "    test-cs=${TEST_CS}  ->  ${OUTPUT_DIR}"
    [ -f "${CKPT}" ]    || { echo "ERROR: checkpoint not found: ${CKPT} (run submit_train_helmholtz.sbatch first)"; exit 1; }
    [ -f "${TEST_CS}" ] || { echo "ERROR: test_cs not found: ${TEST_CS} (run submit_train_helmholtz.sbatch first)"; exit 1; }
    python -u scripts/eval_helmholtz.py \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --checkpoint "${CKPT}" \
        --test-cs "${TEST_CS}" \
        --output-dir "${OUTPUT_DIR}" \
        ${FORCE_FLAG}
    echo "=== helmholtz eval seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
