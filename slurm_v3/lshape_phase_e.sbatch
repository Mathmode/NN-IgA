#!/bin/bash
#SBATCH --job-name=lshape_E
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --constraint=icelake
#SBATCH --exclude=hyperion-[225-233]
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --qos=bcam-exclusive
#SBATCH --output=logs/lshape_E_%A_%a.out
#SBATCH --error=logs/lshape_E_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.conda/envs/${CONDA_ENV:-p1d_v2}/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
if [[ -n "${SLURM_JOB_ID:-}" && "${SMOKE:-0}" != 1 ]]; then
    if ! grep -q 'model name.*Platinum 8362 ' /proc/cpuinfo; then
        echo "Phase E requires Platinum 8362" >&2
        exit 42
    fi
fi
task="${SLURM_ARRAY_TASK_ID:-0}"
p=$((2 + task % 2))
seed=$((task / 2 % 4))
loss=eta
if (( task >= 8 )); then loss=ritz; fi
base="${OUT_ROOT:-data_results_v3/lshape}"
levels=(4 8 16 32)
split=test
extra=()
if [[ "${SMOKE:-0}" == 1 ]]; then
    base="$base/smoke"
    levels=(4 8)
    split=val
    extra=(--smoke --n-train 4 --n-test 2 --batch-size 2 --epochs-per-level 2)
fi
case "${STAGE:-train}" in
    train)
        out="$base/$loss/p$p"
        test ! -f "$out/meta_train_lshape_seed$seed.json"
        "${PYTHON:-python}" -u 2D/scripts/train_lshape.py --p "$p" --seed "$seed" --loss "$loss" --level-feature \
            --protocol val --lr-init 1e-3 --lr-end 1e-4 --batch-size 16 --N-levels "${levels[@]}" "${extra[@]}" --output-dir "$out"
        ;;
    evaluate)
        if (( task >= 16 )); then loss=uniform; seed=0; fi
        "${PYTHON:-python}" -u 2D/scripts/lshape_phase_e_eval.py --p "$p" --seed "$seed" --method "$loss" --root "$base" --split "$split"
        ;;
    alg1)
        index=$((task / 2))
        cap=1000
        if [[ "$split" == test ]]; then levels+=(64); else cap=4; fi
        out="$base/alg1"
        "${PYTHON:-python}" -u 2D/scripts/alg1_2d.py --experiment lshape --split "$split" --instances 16 \
            --instance-range "$index" "$((index+1))" --levels "${levels[@]}" --p "$p" --maxiter "$cap" \
            --losses eta ritz --tag "i$index" --out-dir "$out" --mesh-schedule data_results_v3/lshape/v1_policy.csv \
            --lshape-reference references_v3/lshape_energy_ngsolve.csv --alg-only
        ;;
    *) exit 2;;
esac
