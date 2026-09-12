"""Parametric training for arctan (arctangent 2D) and lshape (L-shape).

LOSS: residual a-posteriori estimator squared, σ-normalised by
``η²(θ_uniform; σ)``. The Ritz energy is **NOT** used as a training
signal — this is the defining methodological difference versus
Aballay et al. 2025.

The public API matches ``continuation.train_with_continuation``:

    make_batch_loss_arctan(p, n_elem, q_K, q_F, q_est) -> batch_loss(params, sigmas, denoms)
    make_batch_loss_lshape(p, n_elem, q_K, q_F, q_est) -> batch_loss(params, sigmas, denoms)
    train_one_level(params0, train_sigmas, train_denoms, *, batch_loss_fn, cfg)

The ``denoms`` array is the per-σ pre-computed ``η²(θ_uniform; σ)``
(see ``src.nonparametric.eta_estimator_2d.eta_uniform_squared_*``).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.parametric.positional_density_network_2d import (
    PDN2DParams,
    knots_p2_from_network_axis,
    knots_p3_from_network_axis,
)
from src.nonparametric.eta_estimator_2d import (
    eta_squared_arctan,
    eta_squared_lshape,
)
from src.nonparametric.r_adapt import p3_effective_n_elem
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape
from src.nonparametric.solver_2d_fdm import galerkin_solve_arctan_fdm

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    n_epochs: int = 5
    batch_size: int = 16
    lr_init: float = 1e-2
    lr_end: float = 1e-3
    log_every: int = 50
    seed: int = 0
    # --- standardized protocol: per-level EARLY STOPPING (additive; gated) ---
    # When ``early_stopping`` is False (default) the loop runs the full
    # ``n_epochs`` exactly as before — the legacy fixed-epoch path is byte
    # identical. When True, the level stops once ``es_patience`` epochs pass
    # with no > ``es_tol`` relative improvement of the per-epoch mean training
    # loss (floored by ``es_min_epochs``); ``n_epochs`` then doubles as the
    # ``max_epochs`` safety cap.
    early_stopping: bool = False
    es_patience: int = 40
    es_tol: float = 0.01          # 1% relative-improvement threshold
    es_min_epochs: int = 40
    # --- robust plateau detection (additive; OFF by default) ---
    # es_abs_tol > 0 makes the STOP decision also require an ABSOLUTE drop in the
    # monitored quantity to count as progress (the patience window only resets on a
    # "significant" improvement = relative > es_tol AND absolute > es_abs_tol). This
    # fixes the converged-but-never-stops case (arctan val-η²: the fine level keeps
    # minting fresh all-time lows by a slow sub-noise drift as the LR decays, so the
    # patience window never closes and the level runs to the max_epochs cap).
    # With es_abs_tol == 0.0 (default) "significant" == the relative test, so the
    # stop anchor coincides with best_epoch and the stop decision is BYTE-IDENTICAL
    # to the previous behaviour (verified across 50 random traces). best-val
    # checkpoint SELECTION (best_mon / best_epoch) is UNCHANGED in all cases — only
    # the patience anchor is affected.
    es_abs_tol: float = 0.0
    # --- scale-adaptive (windowed-relative) plateau (additive; OFF by default) ---
    # es_plateau_rel_tol > 0 REPLACES the patience-anchor stop test with a
    # scale-FREE one: stop once the best monitored value has improved by less than
    # es_plateau_rel_tol (RELATIVE) over the last es_patience epochs. Being a
    # fractional drop of the running best, it fires correctly at ANY magnitude — at
    # val-η²~3.5 (coarse N) the effective bar is ~rel_tol*3.5, at val~6e-6 (fine N)
    # ~rel_tol*6e-6 — unlike the absolute es_abs_tol floor, which is tuned to ONE
    # scale and is inert at the others (it was 7 orders too small at coarse N, so
    # coarse levels never stopped). With es_plateau_rel_tol == 0.0 (default) the
    # legacy per-event anchor test is used → BYTE-IDENTICAL to before (singular/lshape/advdiff/helmholtz
    # unaffected; only the arctan-fast launcher sets it >0). best-val checkpoint
    # SELECTION (best_mon / best_epoch) is the global argmin in BOTH paths.
    es_plateau_rel_tol: float = 0.0
    # --- unified VALIDATION protocol (--protocol val) ---
    # monitor="val": early stopping watches the mean η² over a held-out
    # validation set (passed to train_one_level), not the training loss; this
    # is the only quantity that can detect over-fitting. best_checkpoint=True:
    # return the params at the best-monitored epoch (not the last), so the
    # continuation warm-starts the next level from the best-val weights.
    monitor: str = "train"        # "train" (std) | "val" (val protocol)
    best_checkpoint: bool = False
    # --- unified optimizer (eq. 26 release): global-norm gradient clipping +
    # linear LR warmup into the exponential decay. ``clip_norm`` bounds the
    # global gradient norm (Adam overshoot guard; the arctangent loss spike was
    # diagnosed as un-clipped Adam overshoot). ``warmup_epochs`` ramps the LR
    # 0 -> lr_init over the first epochs of EACH level (the optimizer is
    # re-initialised per level, so the warmup also restarts per level).
    clip_norm: float = 1.0        # config.CLIP_NORM
    warmup_epochs: int = 2        # config.WARMUP_EPOCHS


@dataclass
class LevelTrainResult:
    params_final: PDN2DParams
    history: list[Dict[str, Any]]
    elapsed_sec: float
    final_loss: float
    # --- additive: per-epoch history + early-stop bookkeeping.
    # ``epoch_history`` is always populated (one record/epoch with the per-epoch
    # mean train loss, computed from the already-synced per-batch loss — no extra
    # device sync, no numerical effect; plus ``val_eta2`` when a val set is given).
    # ``stop_reason`` is "plateau" (early stop fired) or "cap" (ran to n_epochs).
    # ``best_val`` / ``best_epoch`` record the best monitored value and its epoch.
    # These default-valued fields are ignored by the legacy fixed-epoch callers.
    epoch_history: List[Dict[str, Any]] = field(default_factory=list)
    epochs_used: int = 0
    stop_reason: str = "cap"
    best_val: float = float("nan")
    best_epoch: int = -1


# --------------------------------------------------------------------------
# Per-σ residual losses (arctan and lshape)
# --------------------------------------------------------------------------


def loss_p2_single_sigma(
    params: PDN2DParams,
    sigma: Array,
    denom_sq: Array,
    *,
    p: int,
    n_elem: int,
    q_K: int,
    q_F: int,
    q_est: int,
    epsilon: float = 0.0,
    solver: str = "dense",
) -> Array:
    """Residual loss for one σ-tuple in arctan:

        L(σ) = η²(θ_φ(σ); σ) / (denom_sq(σ) + ε).

    Per decision 7-H, ``denom_sq`` for arctan is the **analytic H^1 seminorm
    squared** of the manufactured solution u^σ (precomputed once per σ
    before training; arctan has a closed-form exact solution). Per decision
    7-A, ``ε`` is a small safety regulariser against division by tiny
    numbers; callers pass ``epsilon = EPSILON_DENOM`` from
    ``src.config``.

    Dimensionless, ≥ 0; minimisation drives the parametric mesh toward
    the residual-minimising r-adapted layout.
    """
    knots_x = knots_p2_from_network_axis(params, sigma, n_elem, p, axis_id=0.0)
    knots_y = knots_p2_from_network_axis(params, sigma, n_elem, p, axis_id=1.0)
    # solver="dense" (default): the validated dense Cholesky path (byte-identical
    # to before). "fdm": the exact O(N³) Kronecker solver (same u_h/gradient to
    # machine precision; see solver_2d_fdm.py / REPORT_optB_fdm_arctan.md).
    solve_fn = galerkin_solve_arctan_fdm if solver == "fdm" else galerkin_solve_arctan
    res = solve_fn(
        knots_x, knots_y, p, n_elem, n_elem,
        sigma[0], sigma[1], sigma[2],
        q_K=q_K, q_F=q_F,
    )
    eta_sq = eta_squared_arctan(
        res.u_h, knots_x, knots_y, p, n_elem, n_elem,
        sigma[0], sigma[1], sigma[2], q_est=q_est,
    )
    return eta_sq / (
        jnp.asarray(denom_sq, dtype=DEFAULT_DTYPE)
        + jnp.asarray(epsilon, dtype=DEFAULT_DTYPE)
    )


def loss_p3_single_sigma(
    params: PDN2DParams,
    sigma: Array,
    denom_sq: Array,
    *,
    p: int,
    n_elem: int,
    q_K: int,
    q_F: int,
    q_est: int,
    epsilon: float = 0.0,
) -> Array:
    """Residual loss for one σ-tuple in lshape (L-shape).

    Per decision 7-H, ``denom_sq`` for lshape stays as the **uniform-mesh
    estimator** ``η²(θ_unif; σ)`` (no closed-form solution available);
    callers pre-compute it once per σ before training. Per decision 7-A,
    ``ε`` is a small safety regulariser; pass ``epsilon = EPSILON_DENOM``
    from ``src.config``.

    The knot builder produces a C^0 basis at x/y = 0.5 (multiplicity p);
    the solver / estimator are called with the augmented element count
    ``n_eff = n_elem + (p - 1)`` to match.
    """
    n_eff = p3_effective_n_elem(int(n_elem), int(p))
    knots_x = knots_p3_from_network_axis(params, sigma, n_elem, p, axis_id=0.0)
    knots_y = knots_p3_from_network_axis(params, sigma, n_elem, p, axis_id=1.0)
    res = galerkin_solve_lshape(
        knots_x, knots_y, p, n_eff, n_eff,
        sigma[0], sigma[1],
        q_K=q_K, q_F=q_F,
    )
    eta_sq = eta_squared_lshape(
        res.u_h, knots_x, knots_y, p, n_eff, n_eff,
        sigma[0], sigma[1], q_est=q_est,
    )
    return eta_sq / (
        jnp.asarray(denom_sq, dtype=DEFAULT_DTYPE)
        + jnp.asarray(epsilon, dtype=DEFAULT_DTYPE)
    )


# --------------------------------------------------------------------------
# Batched losses (mean over σ in the mini-batch)
# --------------------------------------------------------------------------


def make_batch_loss_arctan(
    *,
    p: int,
    n_elem: int,
    q_K: int = None,
    q_F: int = 50,
    q_est: int = 50,
    epsilon: float = 0.0,
    solver: str = "dense",
):
    """Build a batched arctan loss implementing manuscript eq. (26) literally:

        J_res = (1/2n) Σ_i η²(θ_φ(σ_i); σ_i) / (denom(σ_i) + ε),

    where ``denoms`` MUST be ``η²(θ_unif^{(N)}; σ)`` on the uniform mesh at
    THIS level N (config.LOSS_NORM == "uniform_same_level"). ``epsilon`` is
    the denominator regulariser (decision 7-A); pass ``EPSILON_DENOM``."""
    if q_K is None:
        q_K = p + 1

    def batch_loss(params: PDN2DParams, sigmas: Array, denoms: Array) -> Array:
        per = jax.vmap(
            lambda s, d: loss_p2_single_sigma(
                params, s, d,
                p=p, n_elem=n_elem, q_K=q_K, q_F=q_F, q_est=q_est,
                epsilon=float(epsilon), solver=solver,
            )
        )(sigmas, denoms)
        return 0.5 * jnp.mean(per)          # the 1/(2n) of eq. (26)

    return batch_loss


def make_batch_loss_lshape(
    *,
    p: int,
    n_elem: int,
    q_K: int = None,
    q_F: int = 2,
    q_est: int = 4,
    epsilon: float = 0.0,
):
    """Build a batched lshape loss implementing manuscript eq. (26) literally
    (see ``make_batch_loss_arctan``): ``denoms`` must be ``η²(θ_unif^{(N)}; σ)``
    at THIS level N — recomputed per level, NOT frozen at the coarsest one."""
    if q_K is None:
        q_K = p + 1

    def batch_loss(params: PDN2DParams, sigmas: Array, denoms: Array) -> Array:
        per = jax.vmap(
            lambda s, d: loss_p3_single_sigma(
                params, s, d,
                p=p, n_elem=n_elem, q_K=q_K, q_F=q_F, q_est=q_est,
                epsilon=float(epsilon),
            )
        )(sigmas, denoms)
        return 0.5 * jnp.mean(per)          # the 1/(2n) of eq. (26)

    return batch_loss


# --------------------------------------------------------------------------
# Combined VALIDATION forward (one solve per ν): per-sample eq-26 loss term
# PLUS the discrete solution + knots, so the H¹-error monitor can be computed
# from the SAME solve the val loss already pays for (no duplicate solves).
# --------------------------------------------------------------------------


def make_val_forward_arctan(*, p: int, n_elem: int, q_K: int = None, q_F: int = 50,
                        q_est: int = 50, epsilon: float = 0.0, solver: str = "dense"):
    """Return ``val_forward(params, sigmas, denoms) -> (per, u_hs, kxs, kys)``
    where ``per`` are the per-σ eq-26 terms (η²/(denom+ε); the caller applies
    the ½·mean) and ``u_hs``/``kxs``/``kys`` are the stacked solves/knots.

    ``solver="dense"`` (default) uses the validated dense path; ``"fdm"`` uses the
    exact O(N³) Kronecker solver (machine-precision-identical u_h/gradient)."""
    if q_K is None:
        q_K = p + 1
    solve_fn = galerkin_solve_arctan_fdm if solver == "fdm" else galerkin_solve_arctan

    def one(params, sigma, denom):
        knots_x = knots_p2_from_network_axis(params, sigma, n_elem, p, axis_id=0.0)
        knots_y = knots_p2_from_network_axis(params, sigma, n_elem, p, axis_id=1.0)
        res = solve_fn(
            knots_x, knots_y, p, n_elem, n_elem,
            sigma[0], sigma[1], sigma[2], q_K=q_K, q_F=q_F,
        )
        eta_sq = eta_squared_arctan(
            res.u_h, knots_x, knots_y, p, n_elem, n_elem,
            sigma[0], sigma[1], sigma[2], q_est=q_est,
        )
        per = eta_sq / (jnp.asarray(denom, dtype=DEFAULT_DTYPE)
                        + jnp.asarray(epsilon, dtype=DEFAULT_DTYPE))
        return per, res.u_h, knots_x, knots_y

    def val_forward(params: PDN2DParams, sigmas: Array, denoms: Array):
        return jax.vmap(lambda s, d: one(params, s, d))(sigmas, denoms)

    return val_forward


def make_val_forward_lshape(*, p: int, n_elem: int, q_K: int = None, q_F: int = 2,
                        q_est: int = 4, epsilon: float = 0.0):
    """lshape (L-shape) analogue of ``make_val_forward_arctan`` (C⁰ basis, n_eff)."""
    if q_K is None:
        q_K = p + 1
    n_eff = p3_effective_n_elem(int(n_elem), int(p))

    def one(params, sigma, denom):
        knots_x = knots_p3_from_network_axis(params, sigma, n_elem, p, axis_id=0.0)
        knots_y = knots_p3_from_network_axis(params, sigma, n_elem, p, axis_id=1.0)
        res = galerkin_solve_lshape(
            knots_x, knots_y, p, n_eff, n_eff,
            sigma[0], sigma[1], q_K=q_K, q_F=q_F,
        )
        eta_sq = eta_squared_lshape(
            res.u_h, knots_x, knots_y, p, n_eff, n_eff,
            sigma[0], sigma[1], q_est=q_est,
        )
        per = eta_sq / (jnp.asarray(denom, dtype=DEFAULT_DTYPE)
                        + jnp.asarray(epsilon, dtype=DEFAULT_DTYPE))
        return per, res.u_h, knots_x, knots_y

    def val_forward(params: PDN2DParams, sigmas: Array, denoms: Array):
        return jax.vmap(lambda s, d: one(params, s, d))(sigmas, denoms)

    return val_forward


# --------------------------------------------------------------------------
# Per-level early-stopping decision (pure helpers — unit-testable without JAX).
# --------------------------------------------------------------------------


def _es_is_improvement(loss: float, best: float, tol: float) -> bool:
    """An epoch counts as an improvement only if ``loss`` drops by more than
    ``tol`` (relative) versus the best-so-far ``best`` (decision: monitor the
    per-epoch mean training loss). With ``best = inf`` the first finite loss is
    always an improvement."""
    return float(loss) < float(best) * (1.0 - float(tol))


def _es_should_stop(epoch: int, best_epoch: int, *, min_epochs: int, patience: int) -> bool:
    """Stop iff we are past the ``min_epochs`` floor AND ``patience`` epochs
    have elapsed since the last improvement (``best_epoch``). ``epoch`` is
    0-based, so ``epoch + 1`` is the number of completed epochs."""
    return (int(epoch) + 1) >= int(min_epochs) and (int(epoch) - int(best_epoch)) >= int(patience)


def _es_should_stop_windowed(
    epoch: int,
    best_history: List[float],
    *,
    min_epochs: int,
    patience: int,
    rel_tol: float,
) -> bool:
    """Scale-FREE plateau test. ``best_history[k]`` is the best monitored value at
    the end of epoch ``k`` (monotone non-increasing). Stop iff, past the
    ``min_epochs`` floor and once a full ``patience`` window exists, the best value
    has improved by LESS than ``rel_tol`` *relative* over the last ``patience``
    epochs::

        (best[epoch - patience] - best[epoch]) / best[epoch - patience] < rel_tol

    Because the bar is a fraction of the current best, it auto-scales to the
    level's val-η² magnitude (works identically at val~3.5 and val~6e-6), which a
    fixed absolute floor cannot. ``best_history`` must already include the current
    epoch's best (index ``epoch``)."""
    if (int(epoch) + 1) < int(min_epochs) or int(epoch) < int(patience):
        return False
    ref = float(best_history[int(epoch) - int(patience)])
    cur = float(best_history[int(epoch)])
    if not np.isfinite(ref) or ref <= 0.0:
        return False
    rel_gain = (ref - cur) / ref
    return rel_gain < float(rel_tol)


# --------------------------------------------------------------------------
# Memory bookkeeping (the val protocol runs up to 400 epochs/level; the old
# fixed path ran 100, so the shared trainer must release per-level JAX state).
# --------------------------------------------------------------------------


def mem_probe(tag: str) -> None:
    """Env-gated, numerically INERT memory probe (set ``PDN_MEM_PROBE=1``).

    Prints the live JAX array count and the current process RSS (MB). Pure
    observation — no effect on the RNG, the math, or control flow — so a
    production run (env unset) is byte-identical and silent. Used for the
    leak recon and the post-fix memory-stability proof.
    """
    if not os.environ.get("PDN_MEM_PROBE"):
        return
    try:
        n_live = len(jax.live_arrays())
    except Exception:
        n_live = -1
    rss_mb = -1.0
    try:
        import subprocess
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True)
        rss_mb = float(out.stdout.strip()) / 1024.0   # ps RSS is KB
    except Exception:
        pass
    print(f"  [mem-probe {tag}] live_arrays={n_live} rss_mb={rss_mb:.1f}", flush=True)


def _params_to_host(params: PDN2DParams) -> PDN2DParams:
    """Copy a param pytree to host (numpy) — a single best-val snapshot that
    holds no device buffer (so it survives a level-boundary ``clear_caches``).
    Float64 values are preserved exactly."""
    return PDN2DParams(layers=[(np.asarray(W), np.asarray(b)) for (W, b) in params.layers])


def _params_from_host(host: PDN2DParams) -> PDN2DParams:
    """Rebuild a device param pytree from a host snapshot (exact float64)."""
    return PDN2DParams(layers=[(jnp.asarray(W, dtype=DEFAULT_DTYPE), jnp.asarray(b, dtype=DEFAULT_DTYPE))
                               for (W, b) in host.layers])


# --------------------------------------------------------------------------
# Single-level Adam loop (signature matches the bootstrap continuation.py).
# --------------------------------------------------------------------------


def train_one_level(
    params0: PDN2DParams,
    train_sigmas: np.ndarray,
    train_denoms: np.ndarray,
    *,
    batch_loss_fn: Callable[[PDN2DParams, Array, Array], Array],
    cfg: TrainConfig,
    val_sigmas: np.ndarray | None = None,
    val_denoms: np.ndarray | None = None,
    val_forward_fn: Callable | None = None,
    h1_metric_fn: Callable | None = None,
) -> LevelTrainResult:
    n_train = int(train_sigmas.shape[0])
    bs = int(cfg.batch_size)
    if bs > n_train:
        bs = n_train

    # --- Optimizer (eq. 26 release): warmup -> exponential decay + global-norm
    # clipping. Reproduces the previous lr_init -> lr_end exponential decay over
    # the same horizon, prefixed by a linear 0 -> lr_init ramp of
    # ``cfg.warmup_epochs`` epochs. Re-initialised at each level (as before).
    steps_per_epoch = max(n_train // bs, 1)
    warmup_steps = max(int(cfg.warmup_epochs) * steps_per_epoch, 0)
    schedule = optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=float(cfg.lr_init),
        warmup_steps=warmup_steps,
        transition_steps=max(int(cfg.n_epochs) * steps_per_epoch, 1),
        decay_rate=float(cfg.lr_end) / float(cfg.lr_init),
        end_value=float(cfg.lr_end),
    )
    opt = optax.chain(
        optax.clip_by_global_norm(float(cfg.clip_norm)),
        optax.adam(learning_rate=schedule),
    )
    print(f"  [optim] clip_norm={float(cfg.clip_norm):g} "
          f"warmup={int(cfg.warmup_epochs)}ep ({warmup_steps} steps): "
          f"lr(0)={float(schedule(0)):.2e} -> peak={float(cfg.lr_init):g} "
          f"-> end={float(cfg.lr_end):g}", flush=True)

    params = params0
    opt_state = opt.init(params)
    loss_and_grad = jax.value_and_grad(batch_loss_fn)

    @jax.jit
    def step(params, opt_state, sigmas, denoms):
        loss, grads = loss_and_grad(params, sigmas, denoms)
        gnorm = optax.global_norm(grads)          # pre-clip global grad norm
        updates, opt_state_out = opt.update(grads, opt_state, params)
        params_out = optax.apply_updates(params, updates)
        return params_out, opt_state_out, loss, gnorm

    # Env-gated clip diagnostics (FI_DEBUG_CLIP=1): log the pre/post-clip
    # global gradient norm for the first few iterations of the level.
    debug_clip = os.environ.get("FI_DEBUG_CLIP", "0") == "1"

    # --- VALIDATION monitor (unified val protocol). The val η² is the SAME
    # eq-26 batch loss evaluated forward-only on the held-out val set at this
    # level's N. When ``val_forward_fn`` is given, the SAME solve also yields
    # u_h + knots for the H¹-error monitor (no duplicate solves):
    # val_eta2 = ½·mean(per) is numerically identical to batch_loss_fn. ---
    have_val = (val_sigmas is not None) and (val_denoms is not None)
    if have_val:
        if val_forward_fn is not None:
            val_fwd = jax.jit(val_forward_fn)
        else:
            val_loss_fn = jax.jit(batch_loss_fn)
        val_sigmas_j = jnp.asarray(val_sigmas, dtype=DEFAULT_DTYPE)
        val_denoms_j = jnp.asarray(val_denoms, dtype=DEFAULT_DTYPE)

    rng = np.random.default_rng(int(cfg.seed))
    monitor_val = (str(cfg.monitor) == "val") and have_val
    history: list[Dict[str, Any]] = []
    epoch_history: List[Dict[str, Any]] = []
    last_loss = float("nan")
    # Early-stop bookkeeping (only consulted when cfg.early_stopping is True).
    # ``best_mon`` tracks the best MONITORED value (val η² if monitor=="val",
    # else the per-epoch mean training loss); ``best_params`` is the snapshot at
    # that epoch (functional JAX: each step returns fresh params, so a plain
    # reference is a valid checkpoint).
    best_mon = float("inf")
    best_epoch = -1
    # Separate patience anchor for the STOP decision (additive). It coincides with
    # best_epoch when es_abs_tol == 0.0 (default -> byte-identical to before); when
    # es_abs_tol > 0 it only advances on a "significant" improvement (relative AND
    # absolute), so a sub-noise downward drift can no longer keep the level alive.
    stop_anchor_epoch = -1
    # best-so-far value at the END of each completed epoch (monotone non-increasing).
    # Used ONLY by the windowed-relative plateau test (es_plateau_rel_tol > 0); a
    # cheap list of floats, length == epochs run. Empty/unused on the legacy path.
    best_mon_history: List[float] = []
    # best-val snapshot kept as a SINGLE host (numpy) copy — not a growing list,
    # and no pinned device buffer (so a level-boundary clear_caches reclaims
    # everything). Only maintained when best_checkpoint is requested.
    best_params_host = _params_to_host(params) if bool(cfg.best_checkpoint) else None
    stop_reason = "cap"
    epochs_used = 0
    t0 = time.perf_counter()
    it = 0
    for ep in range(int(cfg.n_epochs)):
        perm = rng.permutation(n_train)
        ep_loss_acc = 0.0
        ep_nb = 0
        for b in range(0, n_train, bs):
            idx = perm[b : b + bs]
            sigmas_b = jnp.asarray(train_sigmas[idx], dtype=DEFAULT_DTYPE)
            denoms_b = jnp.asarray(train_denoms[idx], dtype=DEFAULT_DTYPE)
            params, opt_state, loss, gnorm = step(params, opt_state, sigmas_b, denoms_b)
            it += 1
            if debug_clip and it <= 5:
                g = float(gnorm)
                print(f"  [clip] it={it} |g|_pre={g:.4e} "
                      f"|g|_post={min(g, float(cfg.clip_norm)):.4e} "
                      f"lr={float(schedule(it - 1)):.3e}", flush=True)
            if it % int(cfg.log_every) == 0 or it == 1:
                jax.block_until_ready(loss)
                rec = {"iter": it, "epoch": ep, "loss": float(loss)}
                history.append(rec)
                print(f"  it={it:>5d}  ep={ep}  loss={float(loss):.4e}", flush=True)
            last_loss = float(loss)            # already a per-batch host sync
            ep_loss_acc += last_loss
            ep_nb += 1

        # Per-epoch mean training loss. Computed from the already-synced
        # per-batch losses — ADDITIVE, no numerical effect on training.
        epoch_loss = ep_loss_acc / max(ep_nb, 1)
        # Held-out mean val η² (forward-only) when a val set is given; with a
        # val_forward_fn the same solve also feeds the H¹-error monitor.
        h1_rel_val_median = float("nan")
        if have_val:
            if val_forward_fn is not None:
                per_v, u_hs, kxs, kys = val_fwd(params, val_sigmas_j, val_denoms_j)
                val_eta2 = 0.5 * float(jnp.mean(per_v))   # == batch_loss_fn (eq. 26)
                if h1_metric_fn is not None:
                    h1_arr = np.asarray(
                        h1_metric_fn(np.asarray(u_hs), np.asarray(kxs),
                                     np.asarray(kys), np.asarray(val_sigmas)),
                        dtype=np.float64,
                    )
                    finite = h1_arr[np.isfinite(h1_arr)]
                    h1_rel_val_median = (float(np.median(finite))
                                         if finite.size else float("nan"))
            else:
                val_eta2 = float(val_loss_fn(params, val_sigmas_j, val_denoms_j))
        else:
            val_eta2 = float("nan")
        # Monitored quantity for early stopping / best checkpoint.
        monitored = val_eta2 if monitor_val else epoch_loss
        epochs_used = ep + 1
        epoch_history.append(
            {"epoch": ep, "loss": float(epoch_loss), "val_eta2": float(val_eta2),
             "h1_rel_val_median": float(h1_rel_val_median),
             "iter": it, "time_sec": float(time.perf_counter() - t0)}
        )
        mem_probe(f"ep={ep} it={it}")          # env-gated, inert

        if _es_is_improvement(monitored, best_mon, float(cfg.es_tol)):
            # SELECTION (best_mon / best_epoch / best-val checkpoint): UNCHANGED.
            prev_best = best_mon
            # "Significant" for the STOP anchor: with es_abs_tol == 0.0 this is the
            # plain relative test (anchor == best_epoch -> identical to before);
            # with es_abs_tol > 0 it ALSO requires an absolute drop > es_abs_tol.
            significant = (not np.isfinite(prev_best)) or (
                (prev_best - float(monitored)) > float(cfg.es_abs_tol)
            )
            best_mon = float(monitored)
            best_epoch = ep
            if significant:
                stop_anchor_epoch = ep
            if bool(cfg.best_checkpoint):
                best_params_host = _params_to_host(params)   # single host snapshot

        # Record the best-so-far at the end of THIS epoch (monotone; used only by
        # the windowed-relative plateau test). Cheap append; inert on the legacy path.
        best_mon_history.append(float(best_mon))

        if bool(cfg.early_stopping):
            use_windowed = float(cfg.es_plateau_rel_tol) > 0.0
            if use_windowed:
                fired = _es_should_stop_windowed(
                    ep, best_mon_history,
                    min_epochs=int(cfg.es_min_epochs),
                    patience=int(cfg.es_patience),
                    rel_tol=float(cfg.es_plateau_rel_tol),
                )
            else:
                fired = _es_should_stop(
                    ep, stop_anchor_epoch,
                    min_epochs=int(cfg.es_min_epochs),
                    patience=int(cfg.es_patience),
                )
            if fired:
                stop_reason = "plateau"
                mname = "val_eta2" if monitor_val else "train_loss"
                crit = (f"windowed rel<{float(cfg.es_plateau_rel_tol):.3g} over {int(cfg.es_patience)}ep"
                        if use_windowed else f"per-event tol={float(cfg.es_tol):.3g}")
                print(f"  [early-stop] level plateaued: epochs_used={epochs_used} "
                      f"best_{mname}={best_mon:.4e}@ep{best_epoch} "
                      f"(criterion={crit}, min={int(cfg.es_min_epochs)}, "
                      f"cap={int(cfg.n_epochs)})", flush=True)
                break

    # best-val checkpoint: rebuild the best-monitored host snapshot (val
    # protocol); otherwise the last params (legacy / std behaviour, unchanged).
    params_out = (_params_from_host(best_params_host)
                  if (bool(cfg.best_checkpoint) and best_params_host is not None)
                  else params)

    if bool(cfg.early_stopping):
        mname = "val_eta2" if monitor_val else "train_loss"
        print(f"  [level done] epochs_used={epochs_used}/{int(cfg.n_epochs)} "
              f"stop_reason={stop_reason} monitor={mname} "
              f"best={best_mon:.4e}@ep{best_epoch} "
              f"ckpt={'best' if cfg.best_checkpoint else 'last'}", flush=True)

    jax.block_until_ready(params_out.layers[0][0])
    elapsed = time.perf_counter() - t0
    return LevelTrainResult(
        params_final=params_out,
        history=history,
        elapsed_sec=elapsed,
        final_loss=last_loss,
        epoch_history=epoch_history,
        epochs_used=int(epochs_used),
        stop_reason=str(stop_reason),
        best_val=float(best_mon),
        best_epoch=int(best_epoch),
    )


__all__ = [
    "TrainConfig",
    "LevelTrainResult",
    "loss_p2_single_sigma",
    "loss_p3_single_sigma",
    "make_batch_loss_arctan",
    "make_batch_loss_lshape",
    "make_val_forward_arctan",
    "make_val_forward_lshape",
    "train_one_level",
    "_es_should_stop_windowed",
]
