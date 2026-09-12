"""Single source of truth for the ARCTAN (Aballay 4.3.1) and LSHAPE (Aballay 4.3.2)
experiments under <repo>.

LOSS POLICY: residual a-posteriori estimator (η²/η²_uniform), NOT Ritz energy.
This is the methodological difference vs. Aballay et al. 2025.

Key differences vs. Aballay (also recorded in README.md):

- Loss: residual estimator (generalisable to non-coercive problems).
- Batch size: 16 (cluster CPU efficiency) vs Aballay's 1.
- Iteration budgets: calibrated so the number of "effective updates"
  matches Aballay's published count divided by 16.
- Mesh continuation: (4, 8, 16, 32, 64) for ARCTAN and (10, 20, 30) for LSHAPE (v2, per-patch).
- Polynomial degree LSHAPE: p = 2 (bspline_basis floor) vs Aballay's p = 1.
"""
from __future__ import annotations

from typing import Dict, Tuple

# --------------------------------------------------------------------------
# Common
# --------------------------------------------------------------------------
P_VALUES: Tuple[int, ...] = (2, 3)
SEEDS: Tuple[int, ...] = (0, 1, 2, 3)
PRECISION: str = "float64"


# --------------------------------------------------------------------------
# ARCTAN — Arctangent 2D (Aballay 4.3.1)
# --------------------------------------------------------------------------
ARCTAN: Dict[str, object] = {
    "experiment": "arctan",
    # Mesh
    "levels": (4, 8, 16, 32),
    "anchor_N": 64,
    # Parameter grid
    "alpha_min": 1.0,
    "alpha_max": 20.0,
    "n_alpha": 20,                                # 20-pt reversed-log-base-2
    "s_min": 0.1,
    "s_max": 0.9,
    "n_s1": 10,
    "n_s2": 10,                                   # total = 20*10*10 = 2000
    "train_frac": 0.7,                            # -> 1400 train, 600 test
    # Quadrature
    "quad_forcing": 50,                           # GL nq per dim (matches Aballay)
    "quad_stiffness": None,                       # default p+1 (polynomial-exact)
    "quad_metric": 50,                            # GL nq for H1 evaluation
    "quad_estimator": 50,                         # GL nq for residual estimator
    # Architecture
    "sigma_dim": 3,
    "hidden_dims": (32, 32),                      # capacity ablation (v2): matched to LSHAPE (was (10,10), ~181 -> ~1.3k params)
    "activation": "tanh",
    # Iteration calibration:
    #   Aballay N=64: 500 ep * 1400 / batch=1 = 700k iter.
    #   Our equiv at batch=16: 700k / 16 = 43,750 effective updates.
    #   Continuation across (4,8,16,32,64) shares that budget.
    "p": 3,                                        # FE polynomial degree
}


# --------------------------------------------------------------------------
# LSHAPE — L-shape singular (Aballay 4.3.2)
# --------------------------------------------------------------------------
LSHAPE: Dict[str, object] = {
    "experiment": "lshape",
    # Mesh
    "levels": (10, 20, 30),                       # per-patch; v2 continuation, eval extrapolates to N=40
    "anchor_N": 64,
    "fixed_x": 0.5,
    "fixed_y": 0.5,
    # Parameter grid
    "sigma_exp_min": -1.0,                        # sigma_j = 10^beta_j, beta in [-1, 1]
    "sigma_exp_max": 1.0,
    "n_sigma1": 20,
    "n_sigma2": 20,                               # total 400
    "train_frac": 0.7,                            # -> 280 train, 120 test
    # Quadrature
    "quad_forcing": 4,
    "quad_stiffness": None,                       # default p+1
    "quad_estimator": 4,                          # polynomial integrand for f=1
    # Reference Ritz energy (FOR EVALUATION DIAGNOSTIC ONLY)
    "reference_N": 128,                           # cluster: 256 if budget allows
    "reference_p": 2,                             # bspline_basis only supports p >= 2
    "reference_max_iter": 300,
    # Architecture
    "sigma_dim": 2,
    "hidden_dims": (32, 32),                      # v2.5: reduced from (64,64) to balance quality and runtime
    "activation": "tanh",
    # Aballay N=64: 12,500 ep * 280 / 1 = 3.5M iter. Our equiv: ~220k effective updates.
    "p": 2,
}


# --------------------------------------------------------------------------
# ADVDIFF — Advection-diffusion boundary layer.
#   PDE: -eps(nu) * Lap u + b(nu) * u_x = f_nu, u=0 on the whole boundary.
#   nu = (logeps, b); eps(nu) = 10^logeps. Manufactured solution with an
#   anisotropic boundary layer (width delta = eps/b) at the outflow x=1.
# Standard jnp.linalg.solve + jax.grad pipeline (NOT custom_vjp; see
# REPORT_p2_custom_vjp_2d.md). Single-stage continuation, like ARCTAN.
# --------------------------------------------------------------------------
ADVDIFF: Dict[str, object] = {
    "experiment": "advdiff",
    # Mesh continuation (eval extrapolates to N=64, like ARCTAN).
    "levels": (4, 8, 16, 32),
    "anchor_N": 32,
    # Parameter space nu = (logeps, b).
    "sigma_dim": 2,
    "n_logeps": 20,
    "n_b": 20,
    "logeps_min": -2.0,
    "logeps_max": -1.5,
    "b_min": 0.5,
    "b_max": 2.0,
    "train_frac": 0.7,                            # 70% train / 30% test
    # Network (same width as ARCTAN's wider variant).
    "hidden_dims": (64, 64, 64),
    "activation": "tanh",
    # Quadrature.
    "quad_K": 4,                                  # GL nq/dim for stiffness
    "quad_forcing": 50,                           # GL nq/dim for forcing
    "quad_eta": 50,                               # GL nq/dim for the estimator
    "quad_metric": 50,                            # GL nq/dim for the H1 metric
    # FE polynomial degree (default; overridden by --p / the P_FE sweep).
    "p": 2,
}


# --------------------------------------------------------------------------
# Training defaults (per-level epoch counts dial in "effective updates").
# --------------------------------------------------------------------------
TRAIN: Dict[str, object] = {
    "optimizer": "adam",
    "lr1": 1e-2,
    "lr2": 1e-3,
    "lr2_lshape": 1e-4,                               # v2: LSHAPE-specific lr_end (finer than ARCTAN's 1e-3)
    "batch_size": 16,
    "log_every": 50,
    "noise_scale": 0.0,

    # ARCTAN budget — total ~14k effective updates ((1400/16) iter/ep summed across levels).
    # In Aballay-equivalent batched-iters this is ~14k batched ≈ 225k batch-1 iters,
    # about 1/3 of Aballay's 700k. We choose the LOW end for the bootstrap; the
    # cluster has headroom (8 h budget) to triple this if convergence is weak.
    "epochs_arctan": {4: 100, 8: 100, 16: 100, 32: 100},

    # LSHAPE budget — total ~18k effective updates. Aballay batch-1 equivalent: ~290k iters,
    # vs Aballay's 3.5M. LSHAPE is the more demanding one; we expect to scale this up
    # after the first cluster run if needed.
    "epochs_lshape": {10: 600, 20: 400, 30: 300},          # v2: continuation 10->20->30

    # ADVDIFF budget — uniform 100 epochs per level, mirroring ARCTAN's final schedule.
    "epochs_advdiff": {4: 100, 8: 100, 16: 100, 32: 100},
}


# --------------------------------------------------------------------------
# Standardized 2D training protocol (ARCTAN / LSHAPE / ADVDIFF) — per-level EARLY STOPPING.
#
# Motivation (training_budget_audit.txt): the ad-hoc fixed epochs/level above
# are anti-correlated with need — coarse levels and warm-started fine levels
# over-budgeted, the hard problem's (ADVDIFF) mid-levels under-trained. The fix is
# a single STOPPING CRITERION applied identically to all three benchmarks
# (plus a unified LR tail), so the paper states one protocol instead of three
# epoch tables.
#
# This is SELECTED AT RUN TIME by ``main_train_p{2,3,4}.py --protocol std``.
# The default (``--protocol fixed``) reproduces the legacy per-experiment
# schedules in ``TRAIN`` byte-for-byte; nothing here changes the fixed path.
#
# Per-level rule (monitor the per-epoch mean TRAINING loss = the residual
# objective already tracked):
#   * an epoch is an "improvement" only if the loss drops by > ``tol`` (1%)
#     relative to the best-so-far at this level;
#   * stop the level once ``patience`` epochs pass with no improvement
#     (``min_epochs`` floor so early-epoch noise can't trip it);
#   * ``max_epochs`` is the safety cap — early stopping should fire first.
#
# Unified LR: two-endpoint exponential decay 1e-2 -> 1e-4 for ALL three
# (adopts LSHAPE's finer 1e-4 tail for ARCTAN/ADVDIFF; harmless under early stopping —
# levels stop when flat). The decay horizon = ``max_epochs`` worth of steps
# (i.e. transition_steps = max_epochs * (n_train // batch), unchanged policy),
# so a level that keeps descending keeps annealing toward 1e-4, while a level
# that plateaus simply stops earlier at whatever LR it had reached.
# --------------------------------------------------------------------------
TRAIN_STD: Dict[str, object] = {
    "lr_init": 1e-2,          # unified across ARCTAN/LSHAPE/ADVDIFF
    "lr_end": 1e-4,           # unified fine tail (LSHAPE already used this; ARCTAN/ADVDIFF adopt it)
    "max_epochs": 400,        # per-level safety cap + LR decay horizon
    "min_epochs": 40,         # never stop before this many epochs
    "patience": 40,           # epochs without > tol relative improvement -> stop
    "tol": 0.01,              # 1% relative-improvement threshold
}


# --------------------------------------------------------------------------
# UNIFIED VALIDATION-BASED protocol (--protocol val) — singular..HELMHOLTZ, both degrees.
#
# Supersedes TRAIN_STD for the paper. The std protocol monitored the (noisy)
# TRAINING loss, which cannot detect over-fitting; this protocol instead
# monitors the mean eta^2 over a held-out VALIDATION split, every epoch, at the
# current continuation level, with the SAME loss normalization each experiment
# already uses. Three-way split 70/15/15 (test held out for final metrics only),
# best-val-eta2 checkpoint per level (warm-start next level from best-val).
#
# SPLIT_SEED is a single, documented, FIXED seed shared across all training
# seeds, so every seed sees the SAME reproducible, disjoint test set (the 4
# training seeds vary only network init + batch shuffling). Selected at run time
# by main_train_p{2,3,4}.py --protocol val; the default (fixed) and std paths
# are byte-identical to before.
# --------------------------------------------------------------------------
SPLIT_SEED: int = 20240601    # unified --protocol val split seed (documented, fixed)

TRAIN_VAL: Dict[str, object] = {
    "train_frac": 0.70,       # 70 / 15 / 15 train / val / test (test = remainder)
    "val_frac": 0.15,
    "lr_init": 1e-2,          # unified two-stage LR
    "lr_end": 1e-4,
    "max_epochs": 400,        # per-level cap + LR decay horizon
    "min_epochs": 40,
    "patience": 40,           # epochs without > tol val-eta2 improvement -> stop
    "tol": 0.01,              # 1% relative-improvement threshold (on val eta^2)
}


# --------------------------------------------------------------------------
# Unified mesh policy — per-experiment bounded-logit cap T (decision 7-C)
# and minimum element size h_min (decision 7-D / unified with 1D).
#
# Decision 7-C: T is a per-experiment constant. Start at T=5.0 for every
# experiment (singular, ARCTAN, LSHAPE); keep them independent so one can be tuned
# without affecting the others. No T schedule, no N-dependence.
#
# Decision 7-D / unification: enforce h_min on cell sizes in 2D too (1D
# already does this via H_MIN_SCHEDULE). Per-experiment to mirror the
# pattern of T; uniform across N for simplicity.
# --------------------------------------------------------------------------
T_ARCTAN: float = 5.0          # decision 7-C
T_LSHAPE: float = 5.0          # decision 7-C
H_MIN_ARCTAN: float = 1e-7     # unification / decision 7-D
H_MIN_LSHAPE: float = 1e-7     # unification / decision 7-D

# --------------------------------------------------------------------------
# OPTIONAL per-degree overrides for the L-shape mesh policy (p=3 calibration
# study). EMPTY by default -> every p falls back to the scalars above, so the
# behaviour is BYTE-IDENTICAL to the validated setup (T=5, h_min=1e-7) and p=2
# is untouched. Populate e.g. ``T_LSHAPE_BY_P = {3: 20.0}`` to make ONLY p=3
# use a different temperature WITHOUT affecting p=2: the L-shape knot builder
# (``knots_p3_from_network_axis``) consults these per-p maps first and falls
# back to the scalar. Read by BOTH training and evaluation (both go through the
# same knot builder), so train/eval stay consistent automatically.
#
# CAVEAT: the L-BFGS-B online corrector (``corrector_v2`` / ``evaluation_v2``)
# reads the SCALAR ``T_LSHAPE``; it is OFF in the val protocol
# (``--corrector-max-iter 0``). If you ever enable it together with a per-p T,
# thread the per-p value there too (out of scope for this calibration setup).
T_LSHAPE_BY_P: Dict[int, float] = {}
H_MIN_LSHAPE_BY_P: Dict[int, float] = {}


# --------------------------------------------------------------------------
# Loss denominator ε regulariser (decision 7-A) — small additive safety
# constant applied to the per-σ loss denominator in BOTH ARCTAN (analytic
# H^1 norm) and LSHAPE (uniform-mesh estimator).
# --------------------------------------------------------------------------
EPSILON_DENOM: float = 1e-12


# --------------------------------------------------------------------------
# Unified eq-26 loss + optimizer (Repository Version release).
#
# LOSS_NORM — single source of truth for the training-loss denominator.
#   "uniform_same_level": J_res = (1/2n) Σ η²(θ_φ(ν);ν) / (η²(θ_unif^{(N)};ν)+ε)
#   with θ_unif the uniform mesh AT THE CURRENT continuation level N
#   (manuscript eq. 26). This is the ONLY supported value; the legacy
#   solution-norm denominators (analytic ‖u*‖²_{H¹}, decision 7-H) are
#   DEPRECATED for training and kept solely as H¹-error METRIC denominators.
#
# CLIP_NORM / WARMUP_EPOCHS — optimizer: global-norm gradient clipping and
# a linear LR warmup (0 → lr_init over WARMUP_EPOCHS epochs) prefixed to the
# exponential decay, re-initialised at each continuation level.
# --------------------------------------------------------------------------
LOSS_NORM: str = "uniform_same_level"
CLIP_NORM: float = 1.0
WARMUP_EPOCHS: int = 2


# --------------------------------------------------------------------------
# L-BFGS-B online corrector defaults (used by corrector_v2 in evaluation).
# Decision 7-B: global τ residual gate after training, κ=1.5.
# Decision 7-D: per-axis box shape gate, AR_max=1e3 default.
# --------------------------------------------------------------------------
CORRECTOR: Dict[str, object] = {
    "method": "L-BFGS-B",
    "budget": 80,
    "ftol": 1e-10,
    "gtol": 1e-7,
    "gate_kappa": 1.5,             # decision 7-B
    "shape_gate_AR_max": 1e3,      # decision 7-D (per-axis ratio)
}


def split_seeds(seed: int) -> Dict[str, int]:
    """Derive deterministic per-split seeds from the master seed."""
    return {
        "train": int(seed) * 1000 + 1,
        "val": int(seed) * 1000 + 2,
        "test": int(seed) * 1000 + 3,
    }


__all__ = [
    "P_VALUES",
    "SEEDS",
    "PRECISION",
    "ARCTAN",
    "LSHAPE",
    "ADVDIFF",
    "TRAIN",
    "TRAIN_STD",
    "TRAIN_VAL",
    "SPLIT_SEED",
    "CORRECTOR",
    "T_ARCTAN",
    "T_LSHAPE",
    "H_MIN_ARCTAN",
    "H_MIN_LSHAPE",
    "T_LSHAPE_BY_P",
    "H_MIN_LSHAPE_BY_P",
    "EPSILON_DENOM",
    "LOSS_NORM",
    "CLIP_NORM",
    "WARMUP_EPOCHS",
    "split_seeds",
]
