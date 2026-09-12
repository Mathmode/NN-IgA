from __future__ import annotations

"""Reusable optimization/training utilities for the repo (JAX/Optax)."""

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

Array = jnp.ndarray


class Eta2Terms(NamedTuple):
    """Standard decomposition of eta^2 used across 1D experiments."""

    elem: Array
    jump: Array
    bc: Array
    osc: Array


def eta2_total(
    terms: Eta2Terms,
    *,
    w_elem: float = 1.0,
    w_jump: float = 1.0,
    w_bc: float = 1.0,
    w_osc: float = 1.0,
) -> Array:
    """Combine eta^2 terms with optional weights."""
    return (
        jnp.asarray(w_elem, dtype=terms.elem.dtype) * terms.elem
        + jnp.asarray(w_jump, dtype=terms.elem.dtype) * terms.jump
        + jnp.asarray(w_bc, dtype=terms.elem.dtype) * terms.bc
        + jnp.asarray(w_osc, dtype=terms.elem.dtype) * terms.osc
    )


def residual_loss_from_eta2(eta2: Array) -> Array:
    """Return L = 0.5 * eta^2."""
    return jnp.asarray(0.5, dtype=eta2.dtype) * eta2


def residual_loss_from_terms(terms: Eta2Terms, **weights) -> Array:
    """Return L = 0.5 * eta^2 from a terms decomposition."""
    return residual_loss_from_eta2(eta2_total(terms, **weights))


@dataclass(frozen=True)
class AdamTrainConfig:
    """Generic Adam training configuration (two/three-phase schedule)."""

    iters: int = 10_000
    lr1: float = 1e-3
    lr2: float = 1e-4
    lr_switch: float = 0.5  # fraction in [0,1] or iteration index if >=1
    lr3: Optional[float] = None
    lr_switch2: Optional[float] = None  # second boundary for three-phase schedule

    log_every: int = 200

    # Optional init noise for symmetry breaking / robustness.
    noise_scale: float = 1e-3


def _lr_switch_iter(iters: int, lr_switch: float) -> int:
    iters = int(iters)
    s = float(lr_switch)
    if s < 0:
        return 0
    if s <= 1.0:
        # Match common practice in the repo: `int(frac * iters)` (floor).
        return int(float(s) * float(iters))
    return int(round(s))


def make_exponential_schedule(*, iters: int, lr_init: float, lr_end: float) -> optax.Schedule:
    """Smooth exponential decay from ``lr_init`` to ``lr_end`` over ``iters`` steps.

    lr(t) = lr_init * (lr_end / lr_init)^{t / iters}.
    """
    iters = max(int(iters), 1)
    lr_init = float(lr_init)
    lr_end = float(lr_end)
    if lr_init <= 0.0:
        raise ValueError(f"lr_init must be > 0; got {lr_init}")
    if lr_end <= 0.0:
        raise ValueError(f"lr_end must be > 0; got {lr_end}")
    decay_rate = lr_end / lr_init
    return optax.exponential_decay(
        init_value=lr_init,
        transition_steps=iters,
        decay_rate=decay_rate,
        end_value=lr_end,
    )


def make_two_phase_schedule(*, iters: int, lr1: float, lr2: float, lr_switch: float) -> optax.Schedule:
    """Piecewise-constant schedule: lr1 for [0,switch), then lr2.

    .. deprecated:: Use :func:`make_exponential_schedule` instead.
    """
    switch_it = _lr_switch_iter(int(iters), float(lr_switch))
    scale = float(lr2) / float(lr1) if float(lr1) != 0.0 else 0.0
    return optax.piecewise_constant_schedule(init_value=float(lr1), boundaries_and_scales={int(switch_it): scale})


def make_three_phase_schedule(
    *,
    iters: int,
    lr1: float,
    lr2: float,
    lr3: float,
    lr_switch1: float,
    lr_switch2: float,
) -> optax.Schedule:
    """Piecewise-constant schedule: lr1 -> lr2 -> lr3."""
    s1 = _lr_switch_iter(int(iters), float(lr_switch1))
    s2 = _lr_switch_iter(int(iters), float(lr_switch2))
    if int(s2) <= int(s1):
        raise ValueError(f"lr_switch2 must be greater than lr_switch1, got {s1=} {s2=}")
    scale12 = float(lr2) / float(lr1) if float(lr1) != 0.0 else 0.0
    scale23 = float(lr3) / float(lr2) if float(lr2) != 0.0 else 0.0
    return optax.piecewise_constant_schedule(
        init_value=float(lr1),
        boundaries_and_scales={int(s1): scale12, int(s2): scale23},
    )


def train_adam(
    theta0: Array,
    loss_and_aux_fn: Callable[[Array], Tuple[Array, Any]],
    *,
    cfg: AdamTrainConfig,
    key: Optional[jax.random.PRNGKey] = None,
    log_fn: Optional[
        Callable[[int, Array, Array, Any, Array, float], Dict[str, Any]]
    ] = None,
    early_stop_fn: Optional[Callable[[int, Dict[str, Any], list[Dict[str, Any]], float], Optional[str]]] = None,
) -> tuple[Array, Any, list[Dict[str, Any]], Dict[str, float]]:
    """Generic Adam training loop with a jitted step function.

    Args:
        theta0: Initial parameters (1D array).
        loss_and_aux_fn: Pure JAX function returning (loss, aux).
        cfg: AdamTrainConfig.
        key: Optional PRNGKey used only for init noise.
        log_fn: Optional callback invoked on logging iterations:
            log_fn(it, theta, loss, aux, grad_norm, lr) -> dict to append to history.

    Returns:
        theta, aux, history, summary
        where theta is the FINAL iterate (last Adam step).  best_loss / best_iter
        are recorded in *summary* for diagnostics only.
    """
    theta = jnp.asarray(theta0).reshape(-1)
    dtype = theta.dtype

    if key is not None and float(cfg.noise_scale) != 0.0:
        theta = theta + jnp.asarray(float(cfg.noise_scale), dtype=dtype) * jax.random.normal(
            key, theta.shape, dtype=dtype
        )

    schedule = make_exponential_schedule(
        iters=int(cfg.iters),
        lr_init=float(cfg.lr1),
        lr_end=float(cfg.lr2),
    )

    opt = optax.adam(schedule)
    opt_state = opt.init(theta)

    loss_and_grad = jax.value_and_grad(loss_and_aux_fn, has_aux=True)

    @jax.jit
    def step(theta_in: Array, opt_state_in):
        (loss, aux), g = loss_and_grad(theta_in)
        updates, opt_state_out = opt.update(g, opt_state_in, theta_in)
        theta_out = optax.apply_updates(theta_in, updates)
        gnorm = jnp.linalg.norm(g)
        return theta_out, opt_state_out, loss, aux, gnorm

    history: list[Dict[str, Any]] = []
    last_aux: Any = None
    stop_reason: Optional[str] = None
    stop_iter = -1

    # Best-iterate tracking (Algorithm 1, lines 13-15)
    best_theta = theta
    best_aux: Any = None
    best_loss = float('inf')

    t0 = time.perf_counter()

    n_steps = int(cfg.iters)
    if n_steps <= 0:
        # Degenerate "no training" case: just evaluate loss/aux once.
        loss0, aux0 = loss_and_aux_fn(theta)
        jax.block_until_ready(loss0)
        last_aux = aux0
        best_theta = theta
        best_aux = aux0
        best_loss = float(loss0)
    else:
        for it in range(n_steps):
            theta, opt_state, loss, aux, gnorm = step(theta, opt_state)
            last_aux = aux
            stop_iter = int(it)

            # Track the best iterate
            current_loss = float(loss)
            if current_loss < best_loss:
                best_loss = current_loss
                best_theta = jnp.array(theta)
                best_aux = aux

            if log_fn is not None and ((it % int(cfg.log_every) == 0) or (it == n_steps - 1)):
                # Sync at log points so timings are meaningful.
                jax.block_until_ready(loss)
                lr_now = float(schedule(int(it)))
                rec = log_fn(int(it), theta, loss, aux, gnorm, lr_now)
                history.append(rec)
                if early_stop_fn is not None:
                    reason = early_stop_fn(int(it), rec, history, lr_now)
                    if reason:
                        stop_reason = str(reason)
                        break

        # Final sync before timing.
        jax.block_until_ready(theta)

    elapsed = time.perf_counter() - t0

    summary = {
        "elapsed_sec": float(elapsed),
        "iters_requested": int(n_steps),
        "iters_executed": int(max(stop_iter + 1, 0)),
        "stopped_early": int(stop_reason is not None),
        "stop_iter": int(stop_iter),
        "stop_reason": str(stop_reason) if stop_reason is not None else "",
        "best_loss": best_loss,
    }
    return theta, last_aux, history, summary


__all__ = [
    "Array",
    "Eta2Terms",
    "eta2_total",
    "residual_loss_from_eta2",
    "residual_loss_from_terms",
    "AdamTrainConfig",
    "make_exponential_schedule",
    "make_two_phase_schedule",
    "make_three_phase_schedule",
    "train_adam",
]
