"""Coarse-to-fine continuation for the 2D parametric network.

Trains the same ``PDN2DParams`` instance across a sequence of mesh levels,
saving a checkpoint at each level. Because the network is N-independent
(input ``xi`` lives in [0, 1] regardless of N), the same parameters can
be evaluated at any N.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from src.parametric.positional_density_network_2d import (
    PDN2DParams,
    init_params,
    params_from_flat_dict,
    params_to_flat_dict,
)
from src.parametric.training import (
    LevelTrainResult,
    TrainConfig,
    make_batch_loss_arctan,
    make_batch_loss_lshape,
    mem_probe,
    train_one_level,
)


@dataclass
class ContinuationResult:
    params_final: PDN2DParams
    per_level: List[LevelTrainResult]
    levels_done: List[int]


def save_checkpoint(params: PDN2DParams, path: Path) -> None:
    """Atomically save params to ``path`` (write to .tmp, then rename).

    np.savez does NOT write atomically on its own; a crash mid-write
    leaves a truncated file. We write to a temp file and use an
    os-level atomic rename so an interrupted save (e.g. SLURM TIMEOUT)
    never corrupts an existing checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, **params_to_flat_dict(params))
    # np.savez appends ".npz" only when the path lacks that exact suffix.
    written = tmp if tmp.exists() else tmp.with_name(tmp.name + ".npz")
    written.replace(path)


def load_checkpoint(path: Path) -> PDN2DParams:
    return params_from_flat_dict(np.load(Path(path)))


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Tuple[Path, int]]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_N*.npz"))
    if not candidates:
        return None
    def _level(p: Path) -> int:
        return int(p.stem.replace("checkpoint_N", ""))
    candidates.sort(key=_level)
    return candidates[-1], _level(candidates[-1])


def train_with_continuation(
    *,
    experiment: str,                    # "arctan" or "lshape"
    p: int,
    seed: int,
    train_sigmas: np.ndarray,
    train_denoms: np.ndarray,
    levels: Tuple[int, ...],
    epochs_per_level: dict,
    batch_size: int = 16,
    lr_init: float = 1e-2,
    lr_end: float = 1e-3,
    log_every: int = 50,
    q_K: int = 4,
    q_F: int = 50,
    q_est: int = 50,
    checkpoint_dir: Optional[Path] = None,
    resume: bool = False,
    sigma_dim: int = 3,
    hidden_dims: Tuple[int, ...] = (10, 10),
    epsilon: float = 0.0,
    uniform_init: bool = False,
    init_mode: str = "warmstart",
    solver: str = "dense",
    early_stopping: bool = False,
    es_patience: int = 40,
    es_tol: float = 0.01,
    es_min_epochs: int = 40,
    es_abs_tol: float = 0.0,
    es_plateau_rel_tol: float = 0.0,
    val_sigmas: Optional[np.ndarray] = None,
    val_denoms: Optional[np.ndarray] = None,
    monitor: str = "train",
    best_checkpoint: bool = False,
    history_csv_path: Optional[Path] = None,
    history_is_val: bool = False,
    denoms_fn: Optional[Callable] = None,
    val_forward_factory: Optional[Callable] = None,
    h1_metric_factory: Optional[Callable] = None,
    clip_norm: float = 1.0,
    warmup_epochs: int = 2,
    history_header_note: str = "",
) -> ContinuationResult:
    """Run coarse-to-fine continuation (decision 7-E default).

    Parameters
    ----------
    epsilon : float
        Loss-denominator regulariser (decision 7-A). Callers should pass
        ``EPSILON_DENOM`` from ``src.config``.
    uniform_init : bool
        If True (decision 7-E ``--uniform-init`` flag): zero-init the
        network weights (so ``N_φ(y, ξ) ≡ 0`` ⇒ θ_φ(y) = uniform mesh
        on the first iteration) AND skip continuation, training only at
        the deepest ``levels[-1]`` for the full epoch budget assigned to
        that level. If False (default): standard coarse-to-fine
        continuation through every level in ``levels`` from a LeCun-
        random initialisation.
    early_stopping : bool
        Standardized protocol (config.TRAIN_STD). If True, each level stops
        once ``es_patience`` epochs pass with no > ``es_tol`` relative
        improvement of the per-epoch mean training loss (floored by
        ``es_min_epochs``); ``epochs_per_level`` then acts as the per-level
        ``max_epochs`` safety cap. If False (default), runs the fixed epoch
        budget exactly as before.
    denoms_fn : callable, optional
        ``denoms_fn(N) -> (train_denoms, val_denoms)``. Eq-26 release: the
        loss denominators are ``η²(θ_unif^{(N)}; ν)`` on the uniform mesh AT
        THE CURRENT LEVEL, so they are recomputed at the start of every
        level (this replaces the legacy fixed arrays, including the L-shape
        coarsest-level freeze). When given it overrides ``train_denoms`` /
        ``val_denoms``.
    val_forward_factory / h1_metric_factory : callable, optional
        ``val_forward_factory(N)`` builds the combined val forward (per-ν
        eq-26 terms + u_h + knots from ONE solve); ``h1_metric_factory(N)``
        builds the host-side H¹-error metric ``(u_hs, kxs, kys, sigmas) ->
        h1_rel array``. Together they populate the per-epoch
        ``h1_rel_val_median`` history column.
    clip_norm / warmup_epochs : float / int
        Optimizer knobs (config.CLIP_NORM / config.WARMUP_EPOCHS).

    Returns the FINAL parameters (anchor checkpoint).
    """
    if experiment not in ("arctan", "lshape"):
        raise ValueError(f"experiment must be 'arctan' or 'lshape'; got {experiment}")

    levels_sorted = tuple(int(N) for N in levels)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume detection ----
    params: PDN2DParams
    start_idx = 0
    if resume and checkpoint_dir is not None:
        last = find_latest_checkpoint(checkpoint_dir)
        if last is not None:
            _path, _level = last
            params = load_checkpoint(_path)
            # Skip already-completed levels.
            try:
                start_idx = levels_sorted.index(_level) + 1
            except ValueError:
                start_idx = 0
            print(
                f"[continuation] resume from {_path}, "
                f"completed up to N={_level} (start at idx {start_idx})"
            )

    if start_idx == 0:
        params = init_params(seed=int(seed), sigma_dim=int(sigma_dim), hidden_dims=tuple(hidden_dims))
        if bool(uniform_init):
            # Decision 7-E: zero-init so the network produces the uniform
            # mesh on the first iteration (N_φ(y, ξ) ≡ 0 ⇒ θ_φ(y) = unif).
            zeroed_layers: List[Tuple[jnp.ndarray, jnp.ndarray]] = [
                (jnp.zeros_like(W), jnp.zeros_like(b))
                for (W, b) in params.layers
            ]
            params = PDN2DParams(layers=zeroed_layers)
            print("[continuation] uniform-init: network zero-initialised "
                  "(--uniform-init flag, decision 7-E)")

    # Decision 7-E: --uniform-init bypasses continuation, training only at
    # the deepest level. The continuation loop below then iterates a single
    # element.
    if bool(uniform_init):
        levels_to_train = (levels_sorted[-1],)
        start_idx = 0
    else:
        levels_to_train = levels_sorted

    per_level: List[LevelTrainResult] = []
    levels_done: List[int] = []
    # Incremental per-level history streaming (bounds RAM: per-epoch rows are
    # flushed to disk as each level finishes, not held for the whole run).
    hist_iter = 0
    if history_csv_path is not None:
        history_csv_path = Path(history_csv_path)
        history_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_csv_path, "w") as _hf:
            if history_header_note:
                _hf.write(f"# {history_header_note}\n")
            _hf.write("level_N,iteration,train_loss,val_eta2,h1_rel_val_median\n"
                      if history_is_val else "level_N,iteration,loss\n")
    for idx in range(start_idx, len(levels_to_train)):
        N = levels_to_train[idx]
        # init_mode="uniform" (study variant): RE-INITIALISE the network to its
        # θ0 init at the start of every level instead of inheriting the previous
        # level's converged weights — the network re-learns the deformation from
        # the uniform mesh at each level, with NO hierarchical warm-start
        # transfer. init_mode="warmstart" (default, the paper method) leaves
        # ``params`` as the inherited value from the prior level (set below).
        if str(init_mode) == "uniform" and idx > start_idx:
            params = init_params(seed=int(seed), sigma_dim=int(sigma_dim),
                                 hidden_dims=tuple(hidden_dims))
            print(f"[continuation] init_mode=uniform: re-initialised θ0 at level "
                  f"N={N} (no warm-start transfer)")
        ep = int(epochs_per_level.get(N, 5))
        print(f"\n[continuation] training level N={N}, epochs={ep}")
        # Eq-26 release: per-level uniform-mesh denominators (same-level
        # normalisation). Recomputed at EVERY level start for train and val.
        if denoms_fn is not None:
            train_denoms, val_denoms = denoms_fn(int(N))
            print(f"[continuation] eq-26 denominators at N={N}: "
                  f"train median={float(np.median(train_denoms)):.4e}"
                  + (f", val median={float(np.median(val_denoms)):.4e}"
                     if val_denoms is not None else ""), flush=True)
        val_forward = (val_forward_factory(int(N))
                       if val_forward_factory is not None else None)
        h1_metric = (h1_metric_factory(int(N))
                     if h1_metric_factory is not None else None)
        if experiment == "arctan":
            batch_loss = make_batch_loss_arctan(
                p=int(p), n_elem=int(N), q_K=int(q_K), q_F=int(q_F), q_est=int(q_est),
                epsilon=float(epsilon), solver=str(solver),
            )
        else:
            batch_loss = make_batch_loss_lshape(
                p=int(p), n_elem=int(N), q_K=int(q_K), q_F=int(q_F), q_est=int(q_est),
                epsilon=float(epsilon),
            )

        cfg = TrainConfig(
            n_epochs=ep,
            batch_size=int(batch_size),
            lr_init=float(lr_init),
            lr_end=float(lr_end),
            log_every=int(log_every),
            seed=int(seed) + idx,           # different shuffling each level
            early_stopping=bool(early_stopping),
            es_patience=int(es_patience),
            es_tol=float(es_tol),
            es_min_epochs=int(es_min_epochs),
            es_abs_tol=float(es_abs_tol),
            es_plateau_rel_tol=float(es_plateau_rel_tol),
            monitor=str(monitor),
            best_checkpoint=bool(best_checkpoint),
            clip_norm=float(clip_norm),
            warmup_epochs=int(warmup_epochs),
        )
        result = train_one_level(
            params, train_sigmas, train_denoms,
            batch_loss_fn=batch_loss, cfg=cfg,
            val_sigmas=val_sigmas, val_denoms=val_denoms,
            val_forward_fn=val_forward, h1_metric_fn=h1_metric,
        )
        params = result.params_final
        per_level.append(result)
        levels_done.append(N)
        if bool(early_stopping):
            print(f"[continuation] level N={N}: stop_reason={result.stop_reason} "
                  f"epochs_used={result.epochs_used}/{ep} monitor={monitor} "
                  f"best={result.best_val:.4e}@ep{result.best_epoch}")

        if checkpoint_dir is not None:
            ckpt = checkpoint_dir / f"checkpoint_N{N}.npz"
            save_checkpoint(params, ckpt)
            # Also overwrite checkpoint_final.npz so it always points at the
            # most recently completed level. If a later level times out, the
            # downstream evaluation can still load a valid checkpoint without
            # manual symlinks. The save is atomic enough on POSIX that a
            # crash mid-write leaves the previous checkpoint_final.npz
            # intact (np.savez writes to a temp file then renames).
            final_ckpt = checkpoint_dir / "checkpoint_final.npz"
            save_checkpoint(params, final_ckpt)
            print(f"  -> checkpoint saved: {ckpt}")
            print(f"  -> checkpoint_final.npz updated -> N={N}")

        # Flush this level's per-epoch history to disk, then release it so RAM
        # stays bounded regardless of total epochs (defensive — small for 2D).
        if history_csv_path is not None:
            with open(history_csv_path, "a") as _hf:
                for h in result.epoch_history:
                    hist_iter += 1
                    if history_is_val:
                        _hf.write(f"{int(N)},{hist_iter},{float(h['loss']):.6e},"
                                  f"{float(h['val_eta2']):.6e},"
                                  f"{float(h.get('h1_rel_val_median', float('nan'))):.6e}\n")
                    else:
                        _hf.write(f"{int(N)},{hist_iter},{float(h['loss']):.6e}\n")
            result.epoch_history.clear()

        # OOM FIX: release this level's JAX state before the next (deeper) level
        # allocates. clear_caches drops the prior level's compiled step/val
        # executables; gc.collect breaks the ref-cycles pinning their device
        # buffers. Memory only — no effect on the trained result.
        jax.clear_caches()
        gc.collect()
        mem_probe(f"after N={N} clear")

    return ContinuationResult(
        params_final=params,
        per_level=per_level,
        levels_done=levels_done,
    )


__all__ = [
    "ContinuationResult",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "train_with_continuation",
]
