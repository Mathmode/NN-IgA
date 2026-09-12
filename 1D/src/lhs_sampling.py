"""Deterministic Latin Hypercube sampling over beta in BETA_RANGE.

Used to generate (train / val / test) parameter splits per master seed.
Splits are guaranteed to be disjoint (no shared beta value).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from src.config import BETA_RANGE, TRAIN, VAL, split_seeds, val_split_seeds


def lhs_beta(n: int, lo: float, hi: float, seed: int) -> np.ndarray:
    """Standard 1D LHS on [lo, hi] with the given seed."""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=1, seed=int(seed))
    pts = sampler.random(n=int(n)).flatten()
    return (float(lo) + (float(hi) - float(lo)) * pts).astype(np.float64)


def generate_splits(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic (train, val, test) beta splits derived from a master seed.

    Sizes come from src.config.TRAIN. Ranges from src.config.BETA_RANGE.
    The function checks for split-leakage and raises if any beta is shared.
    """
    seeds = split_seeds(int(seed))
    lo, hi = float(BETA_RANGE[0]), float(BETA_RANGE[1])
    train = lhs_beta(int(TRAIN["train_size"]), lo, hi, seeds["train"])  # type: ignore[index]
    val = lhs_beta(int(TRAIN["val_size"]), lo, hi, seeds["val"])        # type: ignore[index]
    test = lhs_beta(int(TRAIN["test_size"]), lo, hi, seeds["test"])     # type: ignore[index]

    all_beta = np.concatenate([train, val, test])
    eps = 1e-9 * (hi - lo)
    if all_beta.size > 1:
        ordered = np.sort(all_beta)
        diffs = np.diff(ordered)
        if np.any(diffs < eps):
            raise RuntimeError(
                "LHS leakage: two beta values are within {:.1e}. "
                "Re-derive seeds or expand BETA_RANGE.".format(eps)
            )
    return train, val, test


def generate_splits_val() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified val-protocol β split (70/15/15), shared across ALL training seeds.

    Draws a SINGLE LHS set of ``VAL['p1_total']`` betas using the fixed
    ``SPLIT_SEED`` (via ``val_split_seeds``), then partitions it deterministically
    into 70/15/15 train/val/test. Unlike ``generate_splits`` (independent per-seed
    LHS draws), this gives every training seed the SAME reproducible, disjoint
    test set — only init/shuffle vary with the training seed. Disjointness is by
    construction (a partition of one set of distinct LHS points).
    """
    lo, hi = float(BETA_RANGE[0]), float(BETA_RANGE[1])
    total = int(VAL["p1_total"])              # type: ignore[index]
    seeds = val_split_seeds()
    betas = lhs_beta(total, lo, hi, seeds["train"])
    n_train = int(round(float(VAL["train_frac"]) * total))   # type: ignore[index]
    n_val = int(round(float(VAL["val_frac"]) * total))       # type: ignore[index]
    rng = np.random.default_rng(seeds["val"])
    perm = rng.permutation(total)
    train = np.sort(betas[perm[:n_train]])
    val = np.sort(betas[perm[n_train : n_train + n_val]])
    test = np.sort(betas[perm[n_train + n_val :]])
    # Disjoint by construction; guard against accidental coincidence.
    eps = 1e-9 * (hi - lo)
    ordered = np.sort(np.concatenate([train, val, test]))
    if ordered.size > 1 and np.any(np.diff(ordered) < eps):
        raise RuntimeError("val-split leakage: two beta values coincide.")
    return train, val, test


__all__ = ["lhs_beta", "generate_splits", "generate_splits_val"]
