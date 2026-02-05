from __future__ import annotations

"""JAX/Optax trainer for the 1D Helmholtz interface experiment (fixed split).

Key differences vs the PyTorch trainer:
  - Uses a fixed interface index ``n_left`` for JIT-friendly static shapes.
  - Full pipeline is jitted: breakpoints -> assemble -> solve -> estimator.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from helmholtz_mesh_param_jax import build_breakpoints_fixed_split, theta_from_breakpoints_fixed_split
from helmholtz_iga_jax import solve_helmholtz
from helmholtz_estimator_jax import estimator_eta2
from helmholtz_metrics_jax import compute_error_metrics
from optimization import AdamTrainConfig, residual_loss_from_eta2, train_adam

Array = jnp.ndarray


@dataclass
class TrainConfig:
    """Training configuration for r-adaptivity (JAX)."""

    p: int = 2
    iters: int = 10_000

    # Adam schedule
    lr1: float = 1e-3
    lr2: float = 1e-4
    lr_switch: float = 0.5

    # Quadrature
    q_order: int = 20
    q_order_val: int = 40
    q_order_err: int = 60

    # Mesh parametrization
    h_min: float = 1e-6

    # PDE / estimator
    bc_right: str = "neumann"
    include_jump: bool = True

    # Logging
    log_every: int = 200
    seed: int = 42


def train_estimator_adam(
    problem,
    *,
    n_left: int,
    n_right: int,
    cfg: TrainConfig,
    init_breakpoints: Optional[jnp.ndarray] = None,
    init_theta: Optional[Array] = None,
) -> Dict[str, Any]:
    """Train fixed-split r-adaptation parameters via Adam to minimise ``0.5 * eta^2``."""

    n_left = int(n_left)
    n_right = int(n_right)
    n_elem = int(n_left + n_right)
    if n_elem < 2:
        raise ValueError("need at least 2 elements")
    if not (1 <= n_left <= n_elem - 1):
        raise ValueError("n_left must be in 1..n_elem-1")

    p = int(cfg.p)
    bc_neumann = str(cfg.bc_right).lower().startswith("neu")

    # Problem constants (floats) for JIT-friendly closures
    xI = float(problem.xI)
    sigma_L = float(problem.sigma_L)
    sigma_R = float(problem.sigma_R)
    alpha_L = float(problem.alpha_L)
    alpha_R = float(problem.alpha_R)
    gN = float(problem.gN_right())

    k_L = float(problem.k_L)
    k_R = float(problem.k_R)
    A_L = float(problem.A_L)
    A_R = float(problem.A_R)
    B_R = float(problem.B_R)

    u0 = float(problem.u0())
    u1 = float(problem.u1())

    dtype = jnp.float64

    theta = jnp.zeros((n_elem,), dtype=dtype)
    if init_theta is not None:
        th0 = jnp.asarray(init_theta, dtype=dtype).reshape(-1)
        if int(th0.shape[0]) != n_elem:
            raise ValueError(f"init_theta must have length {n_elem}; got {int(th0.shape[0])}")
        theta = th0
    elif init_breakpoints is not None:
        bp0 = jnp.asarray(init_breakpoints, dtype=dtype).reshape(-1)
        if int(bp0.shape[0]) != n_elem + 1:
            raise ValueError(f"init_breakpoints must have length {n_elem+1}; got {int(bp0.shape[0])}")
        theta = theta_from_breakpoints_fixed_split(
            bp0,
            xI=xI,
            n_left=n_left,
            a=0.0,
            b=1.0,
            h_min=float(cfg.h_min),
        )

    # Small random init (optional) for symmetry breaking
    key = jax.random.PRNGKey(int(cfg.seed))
    theta = theta + jnp.asarray(1e-3, dtype=dtype) * jax.random.normal(key, theta.shape, dtype=dtype)

    def loss_and_aux(theta_in: Array):
        bp = build_breakpoints_fixed_split(
            theta_in,
            xI=xI,
            n_left=n_left,
            a=0.0,
            b=1.0,
            h_min=float(cfg.h_min),
        )
        u, knots = solve_helmholtz(
            bp,
            p,
            xI=xI,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
            alpha_L=alpha_L,
            alpha_R=alpha_R,
            n_left=n_left,
            q_order=int(cfg.q_order),
            bc_neumann=bool(bc_neumann),
            u0=u0,
            u1=u1,
            gN=gN,
        )

        eta2, (eta2_elem, eta2_jump, eta2_neu, eta2_osc) = estimator_eta2(
            u,
            bp,
            knots,
            p,
            xI=xI,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
            alpha_L=alpha_L,
            alpha_R=alpha_R,
            gN=gN,
            n_left=n_left,
            q_order=int(cfg.q_order),
            bc_neumann=bool(bc_neumann),
            include_jump=bool(cfg.include_jump),
        )
        loss = residual_loss_from_eta2(eta2)
        return loss, (bp, u, knots, eta2, eta2_elem, eta2_jump, eta2_neu, eta2_osc)

    cfg_adam = AdamTrainConfig(
        iters=int(cfg.iters),
        lr1=float(cfg.lr1),
        lr2=float(cfg.lr2),
        lr_switch=float(cfg.lr_switch),
        log_every=int(cfg.log_every),
        noise_scale=1e-3,
    )

    def log_fn(it: int, _theta: Array, loss: Array, aux, gnorm: Array, lr: float) -> Dict[str, Any]:
        bp, u, knots, eta2, eta2_elem, eta2_jump, eta2_neu, eta2_osc = aux

        eta2_val, _ = estimator_eta2(
            u,
            bp,
            knots,
            p,
            xI=xI,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
            alpha_L=alpha_L,
            alpha_R=alpha_R,
            gN=gN,
            n_left=n_left,
            q_order=int(cfg.q_order_val),
            bc_neumann=bool(bc_neumann),
            include_jump=bool(cfg.include_jump),
        )

        metrics = compute_error_metrics(
            u_coeffs=u,
            breakpoints=bp,
            knots=knots,
            p=p,
            xI=xI,
            sigma_L=sigma_L,
            sigma_R=sigma_R,
            alpha_L=alpha_L,
            alpha_R=alpha_R,
            k_L=k_L,
            k_R=k_R,
            A_L=A_L,
            A_R=A_R,
            B_R=B_R,
            gN=gN,
            n_left=n_left,
            q_order=int(cfg.q_order_err),
        )

        rec = {
            "iter": int(it),
            "loss": float(loss),
            "loss_val": float(0.5 * eta2_val),
            "eta": float(jnp.sqrt(eta2)),
            "eta_val": float(jnp.sqrt(eta2_val)),
            "eta2_total": float(eta2),
            "eta2_elem": float(eta2_elem),
            "eta2_osc": float(eta2_osc),
            "eta2_jump": float(eta2_jump),
            "eta2_neu": float(eta2_neu),
            "n_left": int(n_left),
            **metrics,
            "lr": float(lr),
            "gnorm": float(gnorm),
        }

        print(
            f"[N={n_elem:3d} it {it:5d}] loss={rec['loss']:.3e} val={rec['loss_val']:.3e} | "
            f"H1_rel={rec['H1_rel']:.3e} | L2_rel={rec['L2_rel']:.3e} | h_min={rec['h_min']:.2e}"
        )
        return rec

    theta, aux_final, history, summary = train_adam(
        theta,
        loss_and_aux,
        cfg=cfg_adam,
        key=jax.random.PRNGKey(int(cfg.seed)),
        log_fn=log_fn,
    )

    bp, u, knots, _eta2, _eta2_elem, _eta2_jump, _eta2_neu, _eta2_osc = aux_final

    return {
        "breakpoints": jnp.asarray(bp).astype(jnp.float64),
        "u_coeffs": jnp.asarray(u).astype(jnp.float64),
        "knots": jnp.asarray(knots).astype(jnp.float64),
        "history": history,
        "elapsed_sec": float(summary["elapsed_sec"]),
        "n_left_final": int(n_left),
        "n_right_final": int(n_right),
        "n_elem": int(n_elem),
        "cfg": cfg.__dict__,
    }


__all__ = ["TrainConfig", "train_estimator_adam"]
