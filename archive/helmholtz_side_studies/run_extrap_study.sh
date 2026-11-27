#!/bin/bash
# Launch the contrast-EXTRAPOLATION study (exploratory; NOT production). Submits, in
# order, with SLURM dependencies:
#   1. arm B training  (flat low-c, 4 seeds x p2/p3)      -> data_results/helmholtz_extrap/
#   2. arm B eval on the shared high-c test set           -> data_results/helmholtz_extrap/.../eval/
#      (runs after the arm-B training array finishes ok)
#   3. arm A re-eval (production ckpt) on the SAME high-c  -> data_results/helmholtz_extrap_evalA/.../eval/
#      (no dependency: the production checkpoint already exists)
# arm C (uniform Galerkin floor) needs no run -- it is the `method=uniform` rows in
# the same eval CSVs. Production data_results/helmholtz/ is never written.
#
# Run from the 1D tree:
#     cd 1D && bash scripts/run_extrap_study.sh
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/.."   # the 1D tree, so logs/ + scripts/ resolve

# Sanity: the shared high-c test set must exist before evals are submitted.
REPO_ROOT="$(cd .. && pwd)"
TEST_CS="${REPO_ROOT}/data_results/helmholtz_extrap/high_c_test_cs.npy"
[ -f "${TEST_CS}" ] || { echo "ERROR: missing ${TEST_CS} (generate it first)"; exit 1; }

echo "1) submitting arm B training (4 seeds x p2/p3) ..."
TRAIN_B=$(sbatch --parsable scripts/submit_extrap_train.sbatch)
echo "   arm B train job: ${TRAIN_B}"

echo "2) submitting arm B eval (afterok:${TRAIN_B}) ..."
EVAL_B=$(ARM=B sbatch --parsable --dependency="afterok:${TRAIN_B}" scripts/submit_extrap_eval.sbatch)
echo "   arm B eval job:  ${EVAL_B}"

echo "3) submitting arm A re-eval on the shared high-c set (no dependency) ..."
EVAL_A=$(ARM=A sbatch --parsable scripts/submit_extrap_eval.sbatch)
echo "   arm A eval job:  ${EVAL_A}"

echo ""
echo "submitted. After all finish, analyze with:"
echo "    cd 1D && python scripts/analyze_extrap_study.py"
