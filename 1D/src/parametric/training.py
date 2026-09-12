"""Train the positional density network at one (p, level) with AdamW.

Loss
----
We train rho_phi to minimize the *normalized residual loss* averaged over
training betas. For one beta:

    L(theta(beta), p, N) = reduced_loss(theta, p, q_order, h_min, beta) / H1_norm(beta)^2

where reduced_loss = 0.5 * eta^2 from the analytic estimator and the
denominator removes beta-dependent scaling so that batches mix gracefully.

Training
--------
Single AdamW pass over `max_epochs` epochs with mini-batches of size 64.
Early stopping with patience 80 on validation loss (same definition).
Up to `max_optimizer_restarts` warm restarts of AdamW with the best params
when validation stagnates.

Output
------
A tuple (best_params, history_rows) — the best params seen on validation,
plus the per-epoch log used to write `history.csv`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.config import (EPSILON_DENOM, H_MIN_SCHEDULE, T_SCHEDULE, TRAIN, VAL,
                        CLIP_NORM, WARMUP_EPOCHS)
from src.nonparametric.discretization import (
    solve_state,
    stiffness_quadrature_order,
)
from src.nonparametric.quadrature_analytic import estimator_loss_analytic_power
from src.parametric.positional_density_network import (
    PDNParams,
    cell_xi,
    forward_scalar,
    gauge_fix,
    saturate,
)

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Per-sample loss
# -----------------------------------------------------------------------------


def _make_per_beta_loss(p: int, N: int, h_min: float):
    """Return a JIT-compiled eq-26 per-sample loss(params, beta, denom).

    ``estimator_loss_analytic_power`` returns L = ½·η², and ``denom`` must be
    the uniform-mesh estimator η²(θ_unif^{(N)}; β) at THIS level
    (config.LOSS_NORM == "uniform_same_level"), so

        L / (denom + ε)  =  ½ · η² / (η²_unif + ε)

    and the batch mean is exactly the (1/2n)Σ of manuscript eq. (26).
    The legacy analytic ‖u_β‖²_{H¹} denominator is DEPRECATED (it remains
    available in the evaluation metrics, not in the training loss)."""
    q_order = stiffness_quadrature_order(int(p))
    T = float(T_SCHEDULE[int(p)][int(N)])
    xi = cell_xi(int(N))

    def per_beta_loss(params: PDNParams, beta: Array, denom: Array) -> Array:
        beta_t = jnp.asarray(beta, dtype=DEFAULT_DTYPE)
        z = forward_scalar(params, beta_t, xi, int(N))
        z = gauge_fix(z)
        theta = saturate(z, T)
        # Tracer-friendly residual loss path: solve_state and
        # estimator_loss_analytic_power treat beta as a value (vmap-able).
        u_full, cache = solve_state(theta, int(p), int(q_order), float(h_min), beta_t)
        l = estimator_loss_analytic_power(cache.knots, int(p), u_full, beta=beta_t)
        return l / (jnp.asarray(denom, dtype=beta_t.dtype)
                    + jnp.asarray(EPSILON_DENOM, dtype=beta_t.dtype))

    return per_beta_loss


def make_uniform_eta2_fn(p: int, N: int, h_min: float):
    """Vmapped η²(θ_unif^{(N)}; β) — the eq-26 denominator at level N.

    θ = 0 logits ⇒ the uniform mesh; the estimator helper returns ½·η², so
    the denominator is 2·L_unif."""
    q_order = stiffness_quadrature_order(int(p))
    xi = cell_xi(int(N))

    def one(beta: Array) -> Array:
        beta_t = jnp.asarray(beta, dtype=DEFAULT_DTYPE)
        theta0 = jnp.zeros_like(xi)
        u_full, cache = solve_state(theta0, int(p), int(q_order), float(h_min), beta_t)
        l_unif = estimator_loss_analytic_power(cache.knots, int(p), u_full, beta=beta_t)
        return 2.0 * l_unif

    return jax.jit(jax.vmap(one))


def make_val_forward(p: int, N: int, h_min: float, *, n_sub: int = 10):
    """Combined VAL forward: one solve per β returning the eq-26 per-sample
    term AND the relative H¹-seminorm error vs the exact x^β (in-graph via
    the traceable ``_h1_error_squared`` kernel + the analytic energy norm
    β²/(2β−1); ``n_sub=10`` is the documented 1e-7-accurate setting for
    network meshes — sufficient for a history monitor)."""
    from src.h1_metric import _h1_error_squared

    q_order = stiffness_quadrature_order(int(p))
    T = float(T_SCHEDULE[int(p)][int(N)])
    xi = cell_xi(int(N))

    def one(params: PDNParams, beta: Array, denom: Array):
        beta_t = jnp.asarray(beta, dtype=DEFAULT_DTYPE)
        z = forward_scalar(params, beta_t, xi, int(N))
        z = gauge_fix(z)
        theta = saturate(z, T)
        u_full, cache = solve_state(theta, int(p), int(q_order), float(h_min), beta_t)
        l = estimator_loss_analytic_power(cache.knots, int(p), u_full, beta=beta_t)
        per = l / (jnp.asarray(denom, dtype=beta_t.dtype)
                   + jnp.asarray(EPSILON_DENOM, dtype=beta_t.dtype))
        _, en_sq = _h1_error_squared(u_full, cache, int(p), beta_t,
                                     n_sub=int(n_sub))
        en_norm_sq = (beta_t * beta_t) / (2.0 * beta_t - 1.0)
        h1_rel = jnp.sqrt(jnp.maximum(en_sq, 0.0) / en_norm_sq)
        return per, h1_rel

    def val_forward(params: PDNParams, betas: Array, denoms: Array):
        return jax.vmap(lambda b, d: one(params, b, d))(betas, denoms)

    return jax.jit(val_forward)


def _batched_loss(params: PDNParams, betas: Array, denoms: Array,
                  per_beta_loss: Callable):
    losses = jax.vmap(lambda b, d: per_beta_loss(params, b, d))(betas, denoms)
    return jnp.mean(losses)


# -----------------------------------------------------------------------------
# Train at one level
# -----------------------------------------------------------------------------


@dataclass
class LevelTrainResult:
    params_best: PDNParams
    best_val_loss: float
    history: List[dict]
    optimizer_restarts: int
    elapsed_sec: float
    # --- unified val protocol bookkeeping (additive; defaults keep the legacy
    # callers unchanged) ---
    stop_reason: str = "cap"
    best_epoch: int = -1
    epochs_used: int = 0


def _flatten_params(params: PDNParams) -> List[Array]:
    return [t for layer in params.layers for t in layer]


def _replace_params(params: PDNParams, flat: List[Array]) -> PDNParams:
    layers: list[tuple[Array, Array]] = []
    for i in range(0, len(flat), 2):
        layers.append((flat[i], flat[i + 1]))
    return PDNParams(layers=layers)


def train_at_level(
    params_init: PDNParams,
    *,
    p: int,
    N: int,
    train_betas: np.ndarray,
    val_betas: np.ndarray,
    seed: int,
    protocol: str = "fixed",
    epochs_override: Optional[int] = None,
) -> LevelTrainResult:
    """One continuation pass at level (p, N), monitoring VALIDATION η².

    protocol="fixed" (default): the legacy recipe — AdamW(lr=TRAIN['lr'], wd),
    batch=TRAIN['batch_size'], patience=TRAIN['patience'] with absolute-improvement
    early stopping and up to TRAIN['max_optimizer_restarts'] warm restarts. This
    path is byte-identical to before.

    protocol="val" (unified protocol, config.VAL): Adam with a two-stage
    exponential LR schedule 1e-2 -> 1e-4 over the max_epochs horizon, batch 16,
    and relative-1% / patience-40 / min-40 / cap-400 early stopping on the
    val η² (no warm restarts). Both paths checkpoint the best-val params.
    """
    h_min = float(H_MIN_SCHEDULE[int(p)][int(N)])
    per_beta_loss = _make_per_beta_loss(int(p), int(N), h_min)
    batched_loss = jax.jit(lambda params, betas, denoms: _batched_loss(
        params, betas, denoms, per_beta_loss))
    loss_and_grad = jax.jit(jax.value_and_grad(batched_loss))

    train_betas = np.asarray(train_betas, dtype=np.float64).reshape(-1)
    val_betas = np.asarray(val_betas, dtype=np.float64).reshape(-1)
    n_train = train_betas.shape[0]

    # Eq-26 denominators at THIS level: η²(θ_unif^{(N)}; β) for train + val,
    # recomputed at every level (config.LOSS_NORM == "uniform_same_level").
    unif_fn = make_uniform_eta2_fn(int(p), int(N), h_min)
    train_denoms = np.asarray(unif_fn(jnp.asarray(train_betas, dtype=DEFAULT_DTYPE)))
    val_denoms = np.asarray(unif_fn(jnp.asarray(val_betas, dtype=DEFAULT_DTYPE)))
    print(f"  [singular eq26] η²_unif at N={int(N)}: train median="
          f"{float(np.median(train_denoms)):.4e}, val median="
          f"{float(np.median(val_denoms)):.4e} (recomputed per level)", flush=True)
    # Combined val forward: one solve per β → (eq-26 term, rel H¹-seminorm
    # error vs the exact x^β) for the h1_rel_val_median history column.
    val_forward = make_val_forward(int(p), int(N), h_min)
    val_betas_j = jnp.asarray(val_betas, dtype=DEFAULT_DTYPE)
    val_denoms_j = jnp.asarray(val_denoms, dtype=DEFAULT_DTYPE)

    val_protocol = (str(protocol) == "val")
    if val_protocol:
        bs = int(VAL["batch_size"])                # type: ignore[index]
        if bs > n_train:
            bs = n_train
        max_epochs = int(VAL["max_epochs"])        # type: ignore[index]
        patience = int(VAL["patience"])            # type: ignore[index]
        min_epochs = int(VAL["min_epochs"])        # type: ignore[index]
        tol = float(VAL["tol"])                    # type: ignore[index]
        max_restarts = 0
        nb = max(n_train // bs, 1)
        lr_init = float(VAL["lr_init"]); lr_end = float(VAL["lr_end"])   # type: ignore[index]
        # Eq-26 release optimizer: linear warmup 0 -> lr_init over
        # WARMUP_EPOCHS epochs + the same exponential decay, with global-norm
        # gradient clipping (config.CLIP_NORM).
        warmup_steps = max(int(WARMUP_EPOCHS) * nb, 0)
        schedule = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=lr_init,
            warmup_steps=warmup_steps,
            transition_steps=max(int(max_epochs) * nb, 1),
            decay_rate=lr_end / lr_init,
            end_value=lr_end,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(CLIP_NORM)),
            optax.adam(learning_rate=schedule),
        )
        print(f"  [optim] clip_norm={float(CLIP_NORM):g} warmup={int(WARMUP_EPOCHS)}ep "
              f"({warmup_steps} steps): lr(0)={float(schedule(0)):.2e} -> "
              f"peak={lr_init:g} -> end={lr_end:g}", flush=True)
        lr_log = lr_init
    else:
        lr = float(TRAIN["lr"])             # type: ignore[index]
        wd = float(TRAIN["weight_decay"])   # type: ignore[index]
        bs = int(TRAIN["batch_size"])       # type: ignore[index]
        max_epochs = int(TRAIN["max_epochs"])      # type: ignore[index]
        patience = int(TRAIN["patience"])          # type: ignore[index]
        min_epochs = 0
        tol = 0.0
        max_restarts = int(TRAIN["max_optimizer_restarts"])  # type: ignore[index]
        optimizer = optax.adamw(lr, weight_decay=wd)
        lr_log = lr

    # --epochs-per-level override (None -> config value, byte-identical default):
    # caps every level's training horizon at the requested epoch count.
    if epochs_override is not None:
        max_epochs = int(epochs_override)

    params = params_init
    opt_state = optimizer.init(params)

    rng = np.random.default_rng(int(seed))

    best_params = params
    best_val = float("inf")
    best_epoch = -1
    restarts = 0
    stop_reason = "cap"
    epochs_used = 0
    history: List[dict] = []
    t0 = time.perf_counter()

    epoch = 0
    while epoch < max_epochs:
        idx = rng.permutation(n_train)
        train_loss_acc = 0.0
        n_batches = 0
        for b0 in range(0, n_train, bs):
            sel = idx[b0 : b0 + bs]
            betas_b = jnp.asarray(train_betas[sel], dtype=DEFAULT_DTYPE)
            denoms_b = jnp.asarray(train_denoms[sel], dtype=DEFAULT_DTYPE)
            loss_val, grads = loss_and_grad(params, betas_b, denoms_b)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            train_loss_acc += float(loss_val)
            n_batches += 1
        train_loss = train_loss_acc / max(n_batches, 1)

        # Combined val pass: ONE solve per β yields the eq-26 val loss AND the
        # H¹-seminorm relative error (h1_rel_val_median history column).
        per_v, h1_v = val_forward(params, val_betas_j, val_denoms_j)
        val_loss = float(jnp.mean(per_v))
        h1_rel_val_median = float(jnp.median(h1_v))
        elapsed = time.perf_counter() - t0
        # Improvement: relative > tol (val protocol) or absolute (legacy).
        is_best = (val_loss < best_val * (1.0 - tol)) if val_protocol else (val_loss < best_val)
        if is_best:
            best_val = val_loss
            best_params = jax.tree_util.tree_map(lambda x: x, params)
            best_epoch = epoch
        history.append(
            {
                "level_N": int(N),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "h1_rel_val_median": float(h1_rel_val_median),
                "lr": float(lr_log),
                "time_sec": float(elapsed),
                "optimizer_restarts": int(restarts),
                "is_best": int(bool(is_best)),
            }
        )
        epochs_used = epoch + 1

        if val_protocol:
            # relative-tol / patience / min_epochs / cap; no warm restarts.
            if (epoch + 1) >= min_epochs and (epoch - best_epoch) >= patience:
                stop_reason = "plateau"
                break
        else:
            if epoch - best_epoch > patience:
                if restarts < max_restarts:
                    restarts += 1
                    params = best_params
                    opt_state = optimizer.init(params)
                    best_epoch = epoch  # reset patience window
                else:
                    stop_reason = "plateau"
                    break
        epoch += 1

    if val_protocol:
        print(f"  [singular val] N={int(N)} stop_reason={stop_reason} epochs_used={epochs_used}/"
              f"{max_epochs} best_val_eta2={best_val:.4e}@ep{best_epoch}", flush=True)

    return LevelTrainResult(
        params_best=best_params,
        best_val_loss=float(best_val),
        history=history,
        optimizer_restarts=int(restarts),
        elapsed_sec=float(time.perf_counter() - t0),
        stop_reason=str(stop_reason),
        best_epoch=int(best_epoch),
        epochs_used=int(epochs_used),
    )


__all__ = ["LevelTrainResult", "train_at_level"]
