"""Parametric training for advdiff (advection–diffusion boundary layer).

Mirrors ``training.py`` (arctan/lshape) but for the adv-diff PDE:
  * forward solve via ``galerkin_solve_advdiff`` (jnp.linalg.solve path),
  * estimator via ``eta_squared_advdiff``,
  * parameter ``nu = (logeps, b)`` of dim 2; ``eps = 10^nu[0]``, ``b = nu[1]``,
  * loss denominator = uniform-mesh estimator ``eta^2(theta_unif^{(N)}; nu)`` at the
    current level (``LOSS_NORM="uniform_same_level"``, eq. 28), recomputed per level.
    (The legacy analytic ``|u*_nu|^2_{H1}`` denominator of decision 7-H is DEPRECATED
    for training -- see ``train_advdiff.py`` and ``eq26.py`` -- and kept only as the
    H1-error metric denominator; corrected in remediation T10.)

The methodology is the standard ``jnp.linalg.solve`` + ``jax.grad``
pipeline (NOT the explicit custom_vjp adjoint — see
``REPORT_p2_custom_vjp_2d.md``: it is ~1.5–1.7× slower under vmap).

KNOTS FUNCTION — reused from arctan, not adapted. The positional network's
``_input_features`` concatenates the WHOLE ``sigma`` vector (any length)
with ``xi`` and ``axis_id`` — it never indexes specific components — so
``knots_p2_from_network_axis`` is generic in ``sigma_dim``. advdiff's mesh is
single-patch on [0,1] (no L-shape interface, no multiplicity-p knot),
exactly like arctan. Hence ``knots_p2_from_network_axis`` is the correct
builder for advdiff with ``sigma_dim=2``; we import-alias it as
``knots_p4_from_network_axis``. (Requires the net to be initialised with
``sigma_dim=2`` so the first layer's fan-in is ``2 + 2 = 4``.)

CONTINUATION — ``continuation.train_with_continuation`` hardcodes
``experiment ∈ {"arctan","lshape"}`` (it raises on anything else and selects
``make_batch_loss_arctan/p3`` internally), so it cannot be reused unchanged
for advdiff without modifying existing source. To keep existing files
untouched, this module provides ``train_with_continuation_advdiff`` — a
byte-faithful copy of that single-stage coarse-to-fine loop that
dispatches ``make_batch_loss_advdiff`` instead. All other machinery
(``train_one_level``, ``TrainConfig``, ``init_params``,
``save_checkpoint``, ``ContinuationResult``) is imported and reused.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from src.parametric.positional_density_network_2d import (
    PDN2DParams,
    init_params,
    knots_p2_from_network_axis as knots_p4_from_network_axis,
)
from src.parametric.training import (
    LevelTrainResult,
    TrainConfig,
    mem_probe,
    train_one_level,
)
from src.parametric.continuation import (
    ContinuationResult,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.nonparametric.advdiff.solver_advdiff import galerkin_solve_advdiff
from src.nonparametric.advdiff.eta_estimator_advdiff import eta_squared_advdiff

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Per-nu residual loss
# --------------------------------------------------------------------------


def loss_p4_single_nu(
    params: PDN2DParams,
    nu: Array,
    denom_sq: Array,
    *,
    p: int,
    n_elem: int,
    q_K: int,
    q_F: int,
    q_est: int,
    epsilon: float = 0.0,
) -> Array:
    """Residual loss for one ν-tuple in advdiff:

        L(ν) = η²(θ_φ(ν); ν) / (denom_sq(ν) + ε).

    Per decision 7-H, ``denom_sq`` is the **analytic H¹ seminorm squared**
    of the manufactured solution u*_ν (precomputed once per ν before
    training). ν = (logeps, b) is 2D; ε(ν) = 10^ν[0], b(ν) = ν[1].
    Per decision 7-A, ``ε`` is a small additive safety regulariser.
    """
    eps = jnp.power(jnp.asarray(10.0, dtype=DEFAULT_DTYPE), nu[0])
    b = nu[1]
    knots_x = knots_p4_from_network_axis(params, nu, n_elem, p, axis_id=0.0)
    knots_y = knots_p4_from_network_axis(params, nu, n_elem, p, axis_id=1.0)
    res = galerkin_solve_advdiff(
        knots_x, knots_y, p, n_elem, n_elem,
        eps, b,
        q_K=q_K, q_F=q_F,
    )
    eta_sq = eta_squared_advdiff(
        res.u_h, knots_x, knots_y, p, n_elem, n_elem,
        eps, b, q_est=q_est,
    )
    return eta_sq / (
        jnp.asarray(denom_sq, dtype=DEFAULT_DTYPE)
        + jnp.asarray(epsilon, dtype=DEFAULT_DTYPE)
    )


def make_batch_loss_advdiff(
    *,
    p: int,
    n_elem: int,
    q_K: int = None,
    q_F: int = 50,
    q_est: int = 50,
    epsilon: float = 0.0,
):
    """Build a batched advdiff loss implementing manuscript eq. (26) literally
    (see ``training.make_batch_loss_arctan``): ``denoms`` must be
    ``η²(θ_unif^{(N)}; ν)`` on the uniform mesh at THIS level N
    (config.LOSS_NORM == "uniform_same_level")."""
    if q_K is None:
        q_K = p + 1

    def batch_loss(params: PDN2DParams, nus: Array, denoms: Array) -> Array:
        per = jax.vmap(
            lambda nu, d: loss_p4_single_nu(
                params, nu, d,
                p=p, n_elem=n_elem, q_K=q_K, q_F=q_F, q_est=q_est,
                epsilon=float(epsilon),
            )
        )(nus, denoms)
        return 0.5 * jnp.mean(per)          # the 1/(2n) of eq. (26)

    return batch_loss


def make_val_forward_advdiff(*, p: int, n_elem: int, q_K: int = None, q_F: int = 50,
                        q_est: int = 50, epsilon: float = 0.0):
    """advdiff analogue of ``training.make_val_forward_arctan``: one solve per ν
    returning the per-ν eq-26 term plus (u_h, knots_x, knots_y) for the
    H¹-error monitor."""
    if q_K is None:
        q_K = p + 1

    def one(params, nu, denom):
        eps = jnp.power(jnp.asarray(10.0, dtype=DEFAULT_DTYPE), nu[0])
        b = nu[1]
        knots_x = knots_p4_from_network_axis(params, nu, n_elem, p, axis_id=0.0)
        knots_y = knots_p4_from_network_axis(params, nu, n_elem, p, axis_id=1.0)
        res = galerkin_solve_advdiff(
            knots_x, knots_y, p, n_elem, n_elem, eps, b, q_K=q_K, q_F=q_F,
        )
        eta_sq = eta_squared_advdiff(
            res.u_h, knots_x, knots_y, p, n_elem, n_elem, eps, b, q_est=q_est,
        )
        per = eta_sq / (jnp.asarray(denom, dtype=DEFAULT_DTYPE)
                        + jnp.asarray(epsilon, dtype=DEFAULT_DTYPE))
        return per, res.u_h, knots_x, knots_y

    def val_forward(params: PDN2DParams, nus: Array, denoms: Array):
        return jax.vmap(lambda nu, d: one(params, nu, d))(nus, denoms)

    return val_forward


# --------------------------------------------------------------------------
# Coarse-to-fine continuation for advdiff (parallel to
# continuation.train_with_continuation; that one hardcodes p2/p3).
# --------------------------------------------------------------------------


def train_with_continuation_advdiff(
    *,
    p: int,
    seed: int,
    train_nus: np.ndarray,
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
    sigma_dim: int = 2,
    hidden_dims: Tuple[int, ...] = (64, 64, 64),
    epsilon: float = 0.0,
    uniform_init: bool = False,
    init_mode: str = "warmstart",
    early_stopping: bool = False,
    es_patience: int = 40,
    es_tol: float = 0.01,
    es_min_epochs: int = 40,
    val_nus: Optional[np.ndarray] = None,
    val_denoms: Optional[np.ndarray] = None,
    monitor: str = "train",
    best_checkpoint: bool = False,
    denoms_fn=None,
    val_forward_factory=None,
    h1_metric_factory=None,
    clip_norm: float = 1.0,
    warmup_epochs: int = 2,
) -> ContinuationResult:
    """Single-stage coarse-to-fine continuation for advdiff.

    Faithful copy of ``continuation.train_with_continuation``'s loop
    (resume detection, optional --uniform-init, per-level Adam, per-level
    + final checkpointing) but dispatches ``make_batch_loss_advdiff``. Returns
    the FINAL parameters (anchor checkpoint) in a ``ContinuationResult``.
    """
    levels_sorted = tuple(int(N) for N in levels)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    params: PDN2DParams
    start_idx = 0
    if resume and checkpoint_dir is not None:
        last = find_latest_checkpoint(checkpoint_dir)
        if last is not None:
            _path, _level = last
            params = load_checkpoint(_path)
            try:
                start_idx = levels_sorted.index(_level) + 1
            except ValueError:
                start_idx = 0
            print(f"[continuation-p4] resume from {_path}, "
                  f"completed up to N={_level} (start at idx {start_idx})")

    if start_idx == 0:
        params = init_params(seed=int(seed), sigma_dim=int(sigma_dim),
                             hidden_dims=tuple(hidden_dims))
        if bool(uniform_init):
            zeroed_layers: List[Tuple[jnp.ndarray, jnp.ndarray]] = [
                (jnp.zeros_like(W), jnp.zeros_like(b))
                for (W, b) in params.layers
            ]
            params = PDN2DParams(layers=zeroed_layers)
            print("[continuation-p4] uniform-init: network zero-initialised "
                  "(--uniform-init, decision 7-E)")

    if bool(uniform_init):
        levels_to_train = (levels_sorted[-1],)
        start_idx = 0
    else:
        levels_to_train = levels_sorted

    per_level: List[LevelTrainResult] = []
    levels_done: List[int] = []
    for idx in range(start_idx, len(levels_to_train)):
        N = levels_to_train[idx]
        # init_mode="uniform" (study variant): re-initialise θ0 each level (no
        # warm-start transfer); "warmstart" (default) inherits the prior level.
        if str(init_mode) == "uniform" and idx > start_idx:
            params = init_params(seed=int(seed), sigma_dim=int(sigma_dim),
                                 hidden_dims=tuple(hidden_dims))
            print(f"[continuation-advdiff] init_mode=uniform: re-initialised θ0 "
                  f"at level N={N} (no warm-start transfer)")
        ep = int(epochs_per_level.get(N, 5))
        print(f"\n[continuation-p4] training level N={N}, epochs={ep}")
        # Eq-26 release: per-level uniform-mesh denominators (same-level
        # normalisation), recomputed at every level start for train and val.
        if denoms_fn is not None:
            train_denoms, val_denoms = denoms_fn(int(N))
            print(f"[continuation-p4] eq-26 denominators at N={N}: "
                  f"train median={float(np.median(train_denoms)):.4e}"
                  + (f", val median={float(np.median(val_denoms)):.4e}"
                     if val_denoms is not None else ""), flush=True)
        val_forward = (val_forward_factory(int(N))
                       if val_forward_factory is not None else None)
        h1_metric = (h1_metric_factory(int(N))
                     if h1_metric_factory is not None else None)
        batch_loss = make_batch_loss_advdiff(
            p=int(p), n_elem=int(N), q_K=int(q_K), q_F=int(q_F),
            q_est=int(q_est), epsilon=float(epsilon),
        )
        cfg = TrainConfig(
            n_epochs=ep,
            batch_size=int(batch_size),
            lr_init=float(lr_init),
            lr_end=float(lr_end),
            log_every=int(log_every),
            seed=int(seed) + idx,
            early_stopping=bool(early_stopping),
            es_patience=int(es_patience),
            es_tol=float(es_tol),
            es_min_epochs=int(es_min_epochs),
            monitor=str(monitor),
            best_checkpoint=bool(best_checkpoint),
            clip_norm=float(clip_norm),
            warmup_epochs=int(warmup_epochs),
        )
        result = train_one_level(
            params, train_nus, train_denoms,
            batch_loss_fn=batch_loss, cfg=cfg,
            val_sigmas=val_nus, val_denoms=val_denoms,
            val_forward_fn=val_forward, h1_metric_fn=h1_metric,
        )
        params = result.params_final
        per_level.append(result)
        levels_done.append(N)
        if bool(early_stopping):
            print(f"[continuation-p4] level N={N}: stop_reason={result.stop_reason} "
                  f"epochs_used={result.epochs_used}/{ep} monitor={monitor} "
                  f"best={result.best_val:.4e}@ep{result.best_epoch}")

        if checkpoint_dir is not None:
            ckpt = checkpoint_dir / f"checkpoint_N{N}.npz"
            save_checkpoint(params, ckpt)
            final_ckpt = checkpoint_dir / "checkpoint_final.npz"
            save_checkpoint(params, final_ckpt)
            print(f"  -> checkpoint saved: {ckpt}")
            print(f"  -> checkpoint_final.npz updated -> N={N}")

        # OOM FIX (shared with the arctan/lshape trainer): release this level's JAX
        # state before the next level allocates. Memory only — no effect on the
        # trained result.
        jax.clear_caches()
        gc.collect()
        mem_probe(f"after N={N} clear")

    return ContinuationResult(
        params_final=params,
        per_level=per_level,
        levels_done=levels_done,
    )


__all__ = [
    "knots_p4_from_network_axis",
    "loss_p4_single_nu",
    "make_batch_loss_advdiff",
    "make_val_forward_advdiff",
    "train_with_continuation_advdiff",
]
