#!/bin/bash
# Launch the helmholtz DENOMINATOR STUDY: 6 jobs = 3 denominators x 2 FE degrees,
# each a SLURM array of 4 seeds (0..3). Everything except the loss denominator is
# identical to production. Run from the 1D tree:
#
#     cd 1D && bash scripts/run_denom_study.sh
#
# Outputs land in <repo>/data_results/helmholtz_denom_{uniform,exact,robust}/p{2,3}/.
# The production data_results/helmholtz/ tree is NEVER touched.
#
# Env overrides forwarded to every job:
#     CONDA_ENV=p1d   conda env with JAX   (default: p1d)
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/.."   # the 1D tree, so logs/ and scripts/ resolve as in production

DENOMS=(uniform exact robust)
DEGREES=(2 3)

n=0
for DENOM in "${DENOMS[@]}"; do
    for P in "${DEGREES[@]}"; do
        echo "submitting denom=${DENOM} p=${P} (array seeds 0-3) ..."
        sbatch --job-name="helm_denom_${DENOM}_p${P}" \
               --export=ALL,DENOM="${DENOM}",P_FE="${P}" \
               scripts/submit_denom_study.sbatch
        n=$((n + 1))
    done
done
echo "submitted ${n} jobs (3 denom x 2 deg), 4 seeds each -> 24 training runs total."
