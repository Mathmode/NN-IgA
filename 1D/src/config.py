"""Single source of truth for all hyperparameters of the parametric `singular`
experiment (1D power-type point singularity) and the `helmholtz` experiment.

Modifying any value here is the only way to change the experiment configuration.
All other modules import from this file. The values match exactly the
configuration described in the paper for these two experiments.
"""
from __future__ import annotations

from typing import Dict, Tuple

# --- Experiment scope ----------------------------------------------------
P_VALUES: Tuple[int, ...] = (2, 3)
SEEDS: Tuple[int, ...] = (0, 1, 2, 3)
EVAL_LEVELS: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
CONTINUATION_LEVELS: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)
ANCHOR_N: int = 64
BETA_RANGE: Tuple[float, float] = (1.55, 1.95)

# --- Architecture (positional density network) ---------------------------
ARCH: Dict[str, object] = {
    "type": "positional_density_network",
    "hidden_dims": (32, 32),
    "activation": "tanh",
    "input_dim": 3,                    # (beta, xi, log(xi + 1/N))
    "input_encoding": "log",
}

# --- Training ------------------------------------------------------------
TRAIN: Dict[str, object] = {
    "loss": "residual_normalized",
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "batch_size": 64,
    "max_epochs": 400,
    "patience": 80,
    "max_optimizer_restarts": 2,
    "train_size": 256,
    "val_size": 64,
    "test_size": 128,
}

# --- Per-level schedule --------------------------------------------------
# T_SCHEDULE[p][N] is the saturation parameter T (logit cap) for that (p, N).
T_SCHEDULE: Dict[int, Dict[int, float]] = {
    2: {2: 2.0, 4: 2.0, 8: 2.0, 16: 2.0, 32: 2.0, 64: 3.0, 128: 5.0, 256: 5.0},
    3: {2: 6.0, 4: 6.0, 8: 6.0, 16: 6.0, 32: 6.0, 64: 6.0, 128: 6.0, 256: 7.0},
}

H_MIN_SCHEDULE: Dict[int, Dict[int, float]] = {
    2: {N: (1e-7 if N <= 32 else 1e-8) for N in EVAL_LEVELS},
    3: {N: 1e-8 for N in EVAL_LEVELS},
}

# --- Online correction ---------------------------------------------------
# Per decision 7-B, the residual gate uses a GLOBAL τ = κ · median r_η
# over a held-out validation subset (calibrated once after training).
# Per decision 7-D, the shape gate is a per-axis box on element-size
# ratios; 1D has a single axis so it reduces to ``max_h / min_h``.
CORRECTOR: Dict[str, object] = {
    "method": "L-BFGS-B",
    "budget": 80,
    "ftol": 1e-10,
    "gtol": 1e-7,
    "gate_kappa": 1.5,                # decision 7-B
    "shape_gate_AR_max": 1e5,         # decision 7-D; 1D = max_h / min_h
}

# --- Loss denominator ε regulariser (decision 7-A) ----------------------
# Small additive safety constant on the loss denominator to guard division
# by tiny numbers.
EPSILON_DENOM: float = 1e-12

# --- Unified eq-26 loss + optimizer (Repository Version release) ---------
# LOSS_NORM: the training loss is J_res = (1/2n) Σ η²(θ_φ(ν);ν) /
# (η²(θ_unif^{(N)};ν)+ε) with the UNIFORM mesh AT THE CURRENT level
# (manuscript eq. 26); the legacy analytic solution-norm denominators
# (‖u_β‖²_{H¹} for singular, |u*_c|²_{H¹} for helmholtz) are DEPRECATED for training.
# CLIP_NORM / WARMUP_EPOCHS: global-norm gradient clipping + linear LR
# warmup (0 -> lr_init over WARMUP_EPOCHS epochs), re-initialised per level.
LOSS_NORM: str = "uniform_same_level"
CLIP_NORM: float = 1.0
WARMUP_EPOCHS: int = 2

# --- Visualization ----------------------------------------------------
VIZ: Dict[str, object] = {
    "betas": (1.55, 1.65, 1.75, 1.85, 1.95),
    "viz_seed": 0,
    "x_eval_grid_size": 1000,
    "x_log_min": 1e-7,
    "x_log_max": 1e-2,
}

# --- Numerical precision -------------------------------------------------
PRECISION: str = "float64"


# =========================================================================
# HELMHOLTZ experiment -- parametric 1D Helmholtz transmission family.
#   -(sigma u')' + alpha(x;omega) u = 0,  alpha = -omega^2 rho,  on (0,1),
#   interface x_I=1/2, Dirichlet u(0)=0, Neumann sigma(1)u'(1)=g_N=10 pi.
# Self-contained block (alongside the singular config above, untouched).
# Consumed by src.helmholtz.* and scripts/{train_helmholtz,eval_helmholtz}.py.
# =========================================================================
import math as _math

HELMHOLTZ: Dict[str, object] = {
    "experiment": "helmholtz",
    # Fixed PDE data (mirrors src.helmholtz.pde_helmholtz constants).
    "domain": (0.0, 1.0),
    "x_interface": 0.5,
    "sigma": (1.0, 4.0),
    "rho": ((31.0 * _math.pi / 2.0) ** 2, 100.0 * _math.pi ** 2),
    "g_N": 10.0 * _math.pi,
    # Parameter (frequency) range + sampling.
    "omega_min": 1.0,
    "omega_max": 4.0,
    "n_omega": 100,                # target safe frequencies after resonance pruning
                                   #   (~100 survive the +/-0.02 bands; 80-120 window)
    "train_frac": 0.7,             # 70/30 train/test (endpoints kept if safe)
    "resonance_tol": 0.02,         # exclude +/- this band around each resonance
    # Architecture (positional density network); input (omega, xi, axis_id).
    "hidden_dims": (32, 32),
    "activation": "tanh",
    "input_dim": 3,
    "input_encoding": "omega_xi_axis",   # singular's (beta,xi,log) -> (omega,xi,axis_id)
    # Mesh / continuation (eval extrapolates to N=256).
    "levels": (16, 32, 64, 128),
    "eval_levels": (16, 32, 64, 128, 256),
    "anchor_N": 128,
    "p_values": (2, 3),
    "seeds": (0, 1, 2, 3),
    # Bounded-logit cap T and minimum cell size (per-half softmax floor).
    "T_cap": 5.0,
    "h_min": 1e-7,
    # Training (mirrors singular's residual-normalized AdamW recipe).
    "loss": "residual_normalized",        # eta^2 / |u*_omega|^2_{H1}
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "batch_size": 16,
    "epochs_per_level": {16: 150, 32: 150, 64: 120, 128: 100},

    # ---- helmholtz reparametrization: parameter = wavenumber CONTRAST c = k1/k2 ----
    # (omega/sigma/rho2/g_N above are FIXED; rho_1 = (c*k2)^2*sigma1 -> k1 = c*k2,
    #  k2 = 5*pi.  base case c=3.1 reproduces the original instance.)
    "param": "contrast",                  # mode flag: "contrast" (helmholtz) vs legacy "omega"
    "c_min": 1.5,
    "c_max": 6.0,
    "n_c": 100,                           # target safe contrasts after resonance pruning
    "resonance_tol_c": 0.02,              # exclude +/- band around matching-determinant zeros
    # Global mesh (free split across the C^0 interface) + Nyquist h_max safeguard.
    "mesh": "global",                     # "global" (helmholtz) vs legacy "two_patch"
    "g_min_target": 2.5,                  # points-per-wavelength cap: k*h <= 2*pi/g_min
    "input_dim_c": 3,                     # network input (c, xi, side)
    # Level ladder STARTS at 64: the contrast-continuation warm phase runs at the
    # coarsest level, and the anti-stall smoke showed the high-c (c~6, k1~30*pi)
    # warm phase only escapes the 50:50 stall once high-c is resolved (N>=64:
    # L_frac@c6 -> 0.64, loss 1.5e-3; N=32/48 stall at ~0.52). Network is
    # N-independent so eval extrapolates down to 32/48 and up to 256.
    "levels_c": (64, 96, 128),
    "eval_levels_c": (32, 48, 64, 96, 128, 256),
    # Contrast-continuation curriculum (anti-stall).
    "warm_epochs": 60,                    # high-c exploration epochs at coarsest level (=64)
    "c_high_frac": 0.34,                  # top-third of c used in the warm phase
    "lr_explore": 4e-3,                   # warm-phase LR (then base lr above)
    "epochs_per_level_c": {64: 150, 96: 120, 128: 100},
}


# =========================================================================
# UNIFIED VALIDATION-BASED protocol (--protocol val) — shared by singular and helmholtz.
#
# Replaces each 1D experiment's status-quo stopping rule with η²-VALIDATION
# early stopping on a held-out 70/15/15 split, mirroring the 2D config
# (2D/src/config.py: SPLIT_SEED + TRAIN_VAL). Monitors mean η² over the val
# split every epoch at the current level (same residual-normalized loss used
# in training), best-val-η² checkpoint per level. SPLIT_SEED is the single
# documented, fixed split seed shared across all training seeds (every seed
# sees the SAME reproducible disjoint test set). Gated behind --protocol val;
# the legacy AdamW/fixed-epoch paths are byte-identical under their defaults.
# =========================================================================
SPLIT_SEED: int = 20240601    # unified --protocol val split seed (documented, fixed)

VAL: Dict[str, object] = {
    "train_frac": 0.70,        # 70 / 15 / 15 train / val / test
    "val_frac": 0.15,
    "p1_total": 448,           # singular β-LHS total -> 314 / 67 / 67
    "lr_init": 1e-2,           # unified two-stage LR (singular/helmholtz were single 1e-3)
    "lr_end": 1e-4,
    "max_epochs": 400,         # per-level cap + LR decay horizon
    "min_epochs": 40,
    "patience": 40,            # epochs without > tol val-η² improvement -> stop
    "tol": 0.01,               # 1% relative-improvement threshold (on val η²)
    "batch_size": 16,          # unified (singular was 64; helmholtz already 16)
}


def split_seeds(seed: int) -> Dict[str, int]:
    """Derive deterministic per-split seeds from the master seed."""
    return {
        "train": int(seed) * 1000 + 1,
        "val": int(seed) * 1000 + 2,
        "test": int(seed) * 1000 + 3,
    }


def val_split_seeds() -> Dict[str, int]:
    """Per-split seeds for the unified val protocol, derived from the single
    fixed SPLIT_SEED (shared across all training seeds)."""
    return {
        "train": int(SPLIT_SEED) * 1000 + 1,
        "val": int(SPLIT_SEED) * 1000 + 2,
        "test": int(SPLIT_SEED) * 1000 + 3,
    }


__all__ = [
    "P_VALUES",
    "SEEDS",
    "EVAL_LEVELS",
    "CONTINUATION_LEVELS",
    "ANCHOR_N",
    "BETA_RANGE",
    "ARCH",
    "TRAIN",
    "T_SCHEDULE",
    "H_MIN_SCHEDULE",
    "CORRECTOR",
    "EPSILON_DENOM",
    "VIZ",
    "PRECISION",
    "HELMHOLTZ",
    "SPLIT_SEED",
    "VAL",
    "split_seeds",
    "val_split_seeds",
]
