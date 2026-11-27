#!/bin/bash
#SBATCH --job-name=helm_extrap_eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/helm_extrap_eval_%A_%a.out
#SBATCH --error=logs/helm_extrap_eval_%A_%a.err
#
# Evaluate one arm of the contrast-extrapolation study on the SHARED high-contrast
# test set (data_results/helmholtz_extrap/high_c_test_cs.npy), for BOTH degrees;
# one array task per seed. Each eval writes both methods (positional + uniform),
# so arm C (uniform Galerkin floor) is read off the SAME rows. Parametrized by ARM:
#
#   ARM=B : the extrapolation network -- reads  data_results/helmholtz_extrap/p<d>/checkpoints/...
#                                        writes data_results/helmholtz_extrap/p<d>/eval/
#   ARM=A : the production network (interpolation baseline) -- reads the PRODUCTION
#           checkpoint data_results/helmholtz/p<d>/checkpoints/... but writes ONLY to
#           data_results/helmholtz_extrap_evalA/p<d>/eval/ (production is never written).
#
# Both arms evaluate the SAME high-c contrasts at the SAME N levels -> a fair,
# same-test-set comparison (arm A's eval is re-run here on the high-c set because the
# production eval used the full-range test split, not this high-c subset).
#
# SUBMIT FROM THE 1D TREE (arm B after submit_extrap_train.sbatch finished):
#     cd 1D && ARM=B sbatch scripts/submit_extrap_eval.sbatch
#     cd 1D && ARM=A sbatch scripts/submit_extrap_eval.sbatch
#
# Env overrides:
#     ARM=B|A            which arm to evaluate            (default: B)
#     P_DEGREES="2 3"    degrees                          (default: both)
#     NLEVELS="64 96 128"  mesh levels to evaluate at     (default: the training levels)
#     CONDA_ENV=p1d      conda env                        (default: p1d)
#     FORCE=0            set FORCE=1 to re-evaluate

set -eo pipefail

ARM="${ARM:-B}"
case "${ARM}" in B|A) ;; *) echo "bad ARM=${ARM} (B|A)"; exit 2;; esac

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
TEST_CS="${REPO_ROOT}/data_results/helmholtz_extrap/high_c_test_cs.npy"
[ -f "${TEST_CS}" ] || { echo "ERROR: shared high-c test set not found: ${TEST_CS}"; exit 1; }
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

for P_FE in ${P_DEGREES:-2 3}; do
    if [ "${ARM}" = "B" ]; then
        CKPT="${REPO_ROOT}/data_results/helmholtz_extrap/p${P_FE}/checkpoints/p${P_FE}_seed${SEED}/checkpoint_final.npz"
        OUTPUT_DIR="${REPO_ROOT}/data_results/helmholtz_extrap/p${P_FE}/eval"
    else
        CKPT="${REPO_ROOT}/data_results/helmholtz/p${P_FE}/checkpoints/p${P_FE}_seed${SEED}/checkpoint_final.npz"
        OUTPUT_DIR="${REPO_ROOT}/data_results/helmholtz_extrap_evalA/p${P_FE}/eval"
    fi
    echo "=== extrap eval ARM=${ARM} seed=${SEED} p=${P_FE} on $(hostname) at $(date -Iseconds) ==="
    echo "    checkpoint=${CKPT}"
    echo "    test-cs=${TEST_CS} (shared high-c)  N=[${NLEVELS:-64 96 128}]  ->  ${OUTPUT_DIR}"
    [ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
    python -u scripts/eval_helmholtz.py \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --checkpoint "${CKPT}" \
        --test-cs "${TEST_CS}" \
        --N-levels ${NLEVELS:-64 96 128} \
        --output-dir "${OUTPUT_DIR}" \
        ${FORCE_FLAG}
    echo "=== extrap eval ARM=${ARM} seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
