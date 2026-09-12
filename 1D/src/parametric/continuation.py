"""Coarse-to-fine continuation: train rho_phi sequentially over CONTINUATION_LEVELS.

The positional density network is N-INDEPENDENT — the same parameters can be
evaluated at any N. Continuation exploits this: we initialize once at the
coarsest level and keep training the *same* parameters as N grows. The
saturation cap T(p, N) per (p, N) is the only N-dependent piece.

The anchor level is N = ANCHOR_N = 64 (per src.config). After training all
levels, the *anchor checkpoint* is the canonical artefact used by
evaluation.py for both train- and out-of-sample evaluation, including
levels finer than the anchor (N in {128, 256}).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from common._precision import ensure_double_precision

ensure_double_precision()

import jax.numpy as jnp
import numpy as np

from src.config import CONTINUATION_LEVELS
from src.parametric.positional_density_network import (
    PDNParams,
    init_params,
    params_from_flat_dict,
    params_to_flat_dict,
)
from src.parametric.training import LevelTrainResult, train_at_level


@dataclass
class ContinuationResult:
    params_final: PDNParams
    history: List[dict]                 # concatenated per-level history
    per_level: List[LevelTrainResult]   # in continuation order
    elapsed_sec: float
    levels_done: List[int]


def save_checkpoint(params: PDNParams, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params_to_flat_dict(params))


def load_checkpoint(path: Path) -> PDNParams:
    return params_from_flat_dict(np.load(Path(path)))


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[tuple[Path, int]]:
    """Return (path, level_N) of the latest checkpoint in the dir, or None."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_N*.npz"))
    if not candidates:
        return None
    # Parse the level from the filename
    def _level(p: Path) -> int:
        return int(p.stem.replace("checkpoint_N", ""))
    candidates.sort(key=_level)
    last = candidates[-1]
    return last, _level(last)


def train_with_continuation(
    *,
    p: int,
    seed: int,
    train_betas: np.ndarray,
    val_betas: np.ndarray,
    levels: tuple[int, ...] = CONTINUATION_LEVELS,
    checkpoint_dir: Optional[Path] = None,
    resume: bool = False,
    uniform_init: bool = False,
    init_mode: str = "warmstart",
    epochs_per_level: Optional[int] = None,
    protocol: str = "fixed",
) -> ContinuationResult:
    """Run the full coarse-to-fine continuation for one (p, seed).

    Parameters
    ----------
    p : int
        IGA polynomial degree.
    seed : int
        Master seed for parameter init AND batch shuffling.
    train_betas, val_betas : np.ndarray
        Beta splits (already produced by `generate_splits(seed)`).
    levels : tuple
        Levels to train through (in increasing order). Default is
        ``CONTINUATION_LEVELS = (2, 4, 8, 16, 32, 64)``.
    checkpoint_dir : Path, optional
        Where to save per-level checkpoints. If ``None``, no I/O.
    resume : bool
        If True and ``checkpoint_dir`` contains a previous run, resume from
        the latest checkpoint and skip levels already completed.
    uniform_init : bool, optional (default False)
        Decision 7-E ablation: if True, skip the coarse-to-fine
        continuation and train ONLY at the finest level in ``levels``.
        Network parameters are zero-initialised (every layer's weights
        and biases set to 0) so the policy starts at the uniform mesh
        (logits z = 0 → q = T·tanh(0) = 0 → softmax uniform → uniform
        h). This isolates the value of continuation vs. uniform warm
        start; reproduces the 1D analogue of the 2D --uniform-init
        ablation.
    """
    levels_sorted = tuple(int(N) for N in levels)
    assert all(levels_sorted[i] < levels_sorted[i + 1] for i in range(len(levels_sorted) - 1)), (
        "CONTINUATION_LEVELS must be strictly increasing"
    )
    # Note: the default config guarantees ANCHOR_N == 64 is in
    # CONTINUATION_LEVELS. Callers that pass explicit ``levels`` (e.g., the
    # --smoke flag of main_parametric_1D_singular.py) intentionally override
    # this contract, so we do not assert on ANCHOR_N here.

    # Initialize or resume
    params = init_params(int(seed))
    skip_until: Optional[int] = None
    if resume and checkpoint_dir is not None:
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            ckpt_path, last_level = latest
            params = load_checkpoint(ckpt_path)
            skip_until = int(last_level)
            print(
                f"  [resume] loaded {ckpt_path.name}; skipping <= N={last_level}",
                flush=True,
            )

    # Decision 7-E: --uniform-init ablation. Zero every layer's weights
    # and biases (so the PDN produces z = 0 → q = 0 → uniform h after
    # the softmax + h_min floor) and skip every coarser level — train
    # ONLY at the finest level.
    if bool(uniform_init):
        zeroed_layers: list[tuple[jnp.ndarray, jnp.ndarray]] = [
            (jnp.zeros_like(W), jnp.zeros_like(b))
            for (W, b) in params.layers
        ]
        params = PDNParams(layers=zeroed_layers)
        levels_to_train: tuple[int, ...] = (levels_sorted[-1],)
        print(
            f"  [uniform-init] zeroed network; will train ONLY at N={levels_to_train[0]}",
            flush=True,
        )
    else:
        levels_to_train = levels_sorted

    history: List[dict] = []
    per_level: List[LevelTrainResult] = []
    levels_done: List[int] = []
    t0 = time.perf_counter()

    _trained_any = False
    for N in levels_to_train:
        if skip_until is not None and N <= skip_until:
            continue
        # init_mode="uniform" (study variant): re-initialise θ0 each level after
        # the first (no warm-start transfer); "warmstart" (default) inherits.
        if str(init_mode) == "uniform" and _trained_any:
            params = init_params(int(seed))
            print(f"  [continuation] init_mode=uniform: re-initialised θ0 at "
                  f"level N={N} (no warm-start transfer)", flush=True)
        t_lvl = time.perf_counter()
        result = train_at_level(
            params,
            p=int(p),
            N=int(N),
            train_betas=train_betas,
            val_betas=val_betas,
            seed=int(seed) * 100 + int(N),
            protocol=str(protocol),
            epochs_override=epochs_per_level,
        )
        params = result.params_best
        _trained_any = True
        history.extend(result.history)
        per_level.append(result)
        levels_done.append(int(N))
        if checkpoint_dir is not None:
            save_checkpoint(params, Path(checkpoint_dir) / f"checkpoint_N{int(N)}.npz")
        wall = time.perf_counter() - t_lvl
        print(
            f"  [N={int(N):4d}] best_val={result.best_val_loss:.3e}  "
            f"epochs={len(result.history)}  restarts={result.optimizer_restarts}  "
            f"t={wall:.1f}s",
            flush=True,
        )

    return ContinuationResult(
        params_final=params,
        history=history,
        per_level=per_level,
        elapsed_sec=float(time.perf_counter() - t0),
        levels_done=levels_done,
    )


__all__ = [
    "ContinuationResult",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "train_with_continuation",
]
