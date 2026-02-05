from __future__ import annotations

"""r-adapt optimization loop for Experiment 3 (2D square reaction--diffusion, JAX)."""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import optax

from mesh_param_jax import build_breakpoints_softmax
from square_reactdiff_jax import solve_square_reactdiff_coeffs
from exp3_estimator_jax import estimator_eta2_exp3
from exp3_metrics_jax import compute_error_metrics_exp3

Array = jnp.ndarray


@dataclass
class Exp3TrainConfig:
    p: int = 2
    iters: int = 10_000
    lr1: float = 1e-3
    lr2: float = 1e-4
    lr_switch: float = 0.5
    q_order: int = 12
    q_order_val: int = 20
    q_order_err: int = 20
    q_order_bc: int = 40
    log_every: int = 200
    seed: int = 42
    eps: float = 1e-2
    sigma: float = 1.0
    h_min: float = 1e-6


def train_exp3_estimator_adam(
    *,
    N: int,
    cfg: Exp3TrainConfig,
    init_theta: Optional[Dict[str, Array]] = None,
) -> Dict[str, Any]:
    """Train (theta_x, theta_y) via Adam to minimise ``0.5 * eta^2``."""
    N = int(N)
    p = int(cfg.p)
    dtype = jnp.float64

    def _init_theta(key: str) -> Array:
        if init_theta is not None and key in init_theta:
            th0 = jnp.asarray(init_theta[key], dtype=dtype).reshape(-1)
            if int(th0.shape[0]) != N:
                raise ValueError(f"init_theta[{key}] must have length {N}; got {int(th0.shape[0])}")
            return th0
        return jnp.zeros((N,), dtype=dtype)

    params = {"x": _init_theta("x"), "y": _init_theta("y")}

    # Small random init (optional symmetry breaking)
    key = jax.random.PRNGKey(int(cfg.seed))
    key_x, key_y = jax.random.split(key, 2)
    params = {
        "x": params["x"] + jnp.asarray(1e-3, dtype=dtype) * jax.random.normal(key_x, (N,), dtype=dtype),
        "y": params["y"] + jnp.asarray(1e-3, dtype=dtype) * jax.random.normal(key_y, (N,), dtype=dtype),
    }

    lr_switch = int(float(cfg.lr_switch) * float(cfg.iters))
    schedule = optax.piecewise_constant_schedule(
        init_value=float(cfg.lr1),
        boundaries_and_scales={int(lr_switch): float(cfg.lr2) / float(cfg.lr1)},
    )
    opt = optax.adam(schedule)
    opt_state = opt.init(params)

    def loss_and_aux(params_in):
        bp_x = build_breakpoints_softmax(params_in["x"], 0.0, 1.0, h_min=float(cfg.h_min))
        bp_y = build_breakpoints_softmax(params_in["y"], 0.0, 1.0, h_min=float(cfg.h_min))

        u, knots_x, knots_y = solve_square_reactdiff_coeffs(
            bp_x,
            bp_y,
            p,
            eps=float(cfg.eps),
            sigma=float(cfg.sigma),
            q_order=int(cfg.q_order),
            q_order_bc=int(cfg.q_order_bc),
        )

        eta2, (eta2_vol, _) = estimator_eta2_exp3(
            u,
            bp_x,
            bp_y,
            knots_x,
            knots_y,
            p,
            eps=float(cfg.eps),
            sigma=float(cfg.sigma),
            q_order=int(cfg.q_order),
        )
        loss = jnp.asarray(0.5, dtype=eta2.dtype) * eta2
        return loss, (bp_x, bp_y, knots_x, knots_y, u, eta2, eta2_vol)

    loss_and_grad = jax.value_and_grad(loss_and_aux, has_aux=True)

    @jax.jit
    def step(params_in, opt_state_in):
        (loss, aux), grads = loss_and_grad(params_in)
        updates, opt_state_out = opt.update(grads, opt_state_in, params_in)
        params_out = optax.apply_updates(params_in, updates)
        gnorm = jnp.sqrt(jnp.sum(grads["x"] * grads["x"]) + jnp.sum(grads["y"] * grads["y"]))
        return params_out, opt_state_out, loss, aux, gnorm

    history: list[Dict[str, Any]] = []
    t0 = time.perf_counter()

    last_bp_x = None
    last_bp_y = None
    last_knots_x = None
    last_knots_y = None
    last_u = None

    for it in range(int(cfg.iters) + 1):
        params, opt_state, loss, aux, gnorm = step(params, opt_state)
        bp_x, bp_y, knots_x, knots_y, u, eta2, eta2_vol = aux
        last_bp_x, last_bp_y = bp_x, bp_y
        last_knots_x, last_knots_y = knots_x, knots_y
        last_u = u

        if (it % int(cfg.log_every) == 0) or (it == int(cfg.iters)):
            eta2_val, _ = estimator_eta2_exp3(
                u,
                bp_x,
                bp_y,
                knots_x,
                knots_y,
                p,
                eps=float(cfg.eps),
                sigma=float(cfg.sigma),
                q_order=int(cfg.q_order_val),
            )

            metrics = compute_error_metrics_exp3(
                u,
                bp_x,
                bp_y,
                knots_x,
                knots_y,
                p,
                q_order=int(cfg.q_order_err),
                eps=float(cfg.eps),
                sigma=float(cfg.sigma),
            )

            bp_x_np = jax.device_get(bp_x)
            bp_y_np = jax.device_get(bp_y)
            min_hx = float((bp_x_np[1:] - bp_x_np[:-1]).min())
            min_hy = float((bp_y_np[1:] - bp_y_np[:-1]).min())

            lr_k = float(schedule(it))
            rec = {
                "iter": int(it),
                "loss": float(jax.device_get(loss)),
                "loss_val": float(0.5 * float(jax.device_get(eta2_val))),
                "eta": float(jnp.sqrt(float(jax.device_get(eta2)))),
                "eta_val": float(jnp.sqrt(float(jax.device_get(eta2_val)))),
                "eta2_vol": float(jax.device_get(eta2_vol)),
                "lr": float(lr_k),
                "min_hx": float(min_hx),
                "min_hy": float(min_hy),
                "h_min": float(min(min_hx, min_hy)),
                "gnorm": float(jax.device_get(gnorm)),
                **metrics,
            }
            history.append(rec)
            print(
                f"[N={N:3d} it {it:5d}] loss={rec['loss']:.3e} val={rec['loss_val']:.3e} | "
                f"H1_rel={rec['H1_rel']:.3e} | L2_rel={rec['L2_rel']:.3e} | h_min={rec['h_min']:.2e}"
            )

    elapsed = time.perf_counter() - t0

    if last_bp_x is None or last_bp_y is None or last_knots_x is None or last_knots_y is None or last_u is None:
        bp_x = build_breakpoints_softmax(params["x"], 0.0, 1.0, h_min=float(cfg.h_min))
        bp_y = build_breakpoints_softmax(params["y"], 0.0, 1.0, h_min=float(cfg.h_min))
        last_u, last_knots_x, last_knots_y = solve_square_reactdiff_coeffs(
            bp_x,
            bp_y,
            p,
            eps=float(cfg.eps),
            sigma=float(cfg.sigma),
            q_order=int(cfg.q_order),
            q_order_bc=int(cfg.q_order_bc),
        )
        last_bp_x, last_bp_y = bp_x, bp_y

    out = {
        "breakpoints": {"x": jax.device_get(last_bp_x), "y": jax.device_get(last_bp_y)},
        "knots": {"x": jax.device_get(last_knots_x), "y": jax.device_get(last_knots_y)},
        "u": jax.device_get(last_u),
        "history": history,
        "elapsed_sec": float(elapsed),
        "dofs": int(jnp.asarray(last_u).size),
        "cfg": cfg.__dict__,
        "N": int(N),
    }
    return out


__all__ = [
    "Exp3TrainConfig",
    "train_exp3_estimator_adam",
]
