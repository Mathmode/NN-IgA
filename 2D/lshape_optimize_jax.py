from __future__ import annotations

"""r-adapt optimization loop for the L-shape Laplace experiment (JAX)."""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from lshape_dof import build_global_dof_maps, dirichlet_dofs_lshape
from mesh_param_jax import build_breakpoints_softmax
from lshape_poisson_jax import solve_lshape_coeffs
from lshape_estimator_jax import estimator_eta2
from lshape_metrics_jax import compute_error_metrics

Array = jnp.ndarray


@dataclass
class TrainConfig:
    p: int = 2
    iters: int = 2000

    # Adam schedule
    lr1: float = 1e-2
    lr2: float = 1e-3
    lr_switch: float = 0.5

    # Quadrature
    q_order: int = 8
    q_order_val: int = 12
    q_order_err: int = 20

    # Mesh parametrization
    h_min: float = 1e-6

    # Estimator
    include_jump: bool = True
    eps_side: float = 1e-6

    # CG solver
    solver: str = "cg"
    dense_max_dofs: int = 1500
    cg_tol: float = 1e-4
    cg_maxiter: int = 4000
    precond: str = "jacobi"

    # Logging
    log_every: int = 100
    seed: int = 42
    noise_scale: float = 0.0

    # Mesh regularization (stability on graded meshes)
    mesh_reg: float = 0.0
    mesh_reg_hmin: float = 0.0

    # Gradient clipping (0 disables)
    grad_clip: float = 0.0


def _free_dofs(n_dofs: int, dirichlet: np.ndarray) -> np.ndarray:
    all_idx = np.arange(int(n_dofs), dtype=np.int32)
    return np.setdiff1d(all_idx, np.asarray(dirichlet, dtype=np.int32), assume_unique=False)


def train_estimator_adam(
    *,
    n_xL: int,
    n_xR: int,
    n_yB: int,
    n_yT: int,
    cfg: TrainConfig,
    init_theta: Optional[Dict[str, Array]] = None,
) -> Dict[str, Any]:
    """Train breakpoint parameters via Adam to minimize 0.5*η^2."""
    n_xL = int(n_xL)
    n_xR = int(n_xR)
    n_yB = int(n_yB)
    n_yT = int(n_yT)
    p = int(cfg.p)
    dtype = jnp.float64

    # Static mesh topology (depends only on counts)
    nBxL = n_xL + p
    nBxR = n_xR + p
    nByB = n_yB + p
    nByT = n_yT + p
    g0_np, g1_np, g2_np, n_dofs = build_global_dof_maps(nBxL, nBxR, nByB, nByT)
    dir_np = dirichlet_dofs_lshape(g0_np, g1_np, g2_np)
    free_np = _free_dofs(n_dofs, dir_np)

    g0 = jnp.asarray(g0_np, dtype=jnp.int32)
    g1 = jnp.asarray(g1_np, dtype=jnp.int32)
    g2 = jnp.asarray(g2_np, dtype=jnp.int32)
    dirichlet = jnp.asarray(dir_np, dtype=jnp.int32)
    free_idx = jnp.asarray(free_np, dtype=jnp.int32)
    n_free = int(free_idx.shape[0])
    x0_free = jnp.zeros((n_free,), dtype=dtype)

    def _init_vec(key: str, n: int) -> Array:
        if init_theta is not None and key in init_theta:
            th0 = jnp.asarray(init_theta[key], dtype=dtype).reshape(-1)
            if int(th0.shape[0]) != int(n):
                raise ValueError(f"init_theta[{key}] must have length {int(n)}; got {int(th0.shape[0])}")
            return th0
        return jnp.zeros((int(n),), dtype=dtype)

    params = {
        "xL": _init_vec("xL", n_xL),
        "xR": _init_vec("xR", n_xR),
        "yB": _init_vec("yB", n_yB),
        "yT": _init_vec("yT", n_yT),
    }

    if float(cfg.noise_scale) != 0.0:
        # Optional symmetry breaking / robustness. Use 0.0 for strictly-uniform
        # initialisations (exercise-mode, reproducibility).
        key = jax.random.PRNGKey(int(cfg.seed))
        keys = jax.random.split(key, 4)
        scale = jnp.asarray(float(cfg.noise_scale), dtype=dtype)
        params = {
            "xL": params["xL"] + scale * jax.random.normal(keys[0], params["xL"].shape, dtype=dtype),
            "xR": params["xR"] + scale * jax.random.normal(keys[1], params["xR"].shape, dtype=dtype),
            "yB": params["yB"] + scale * jax.random.normal(keys[2], params["yB"].shape, dtype=dtype),
            "yT": params["yT"] + scale * jax.random.normal(keys[3], params["yT"].shape, dtype=dtype),
        }

    lr_switch = int(float(cfg.lr_switch) * float(cfg.iters))
    schedule = optax.piecewise_constant_schedule(
        init_value=float(cfg.lr1),
        boundaries_and_scales={int(lr_switch): float(cfg.lr2) / float(cfg.lr1)},
    )
    opt = optax.adam(schedule)
    opt_state = opt.init(params)

    precond = str(cfg.precond)
    include_jump = bool(cfg.include_jump)
    solver = str(cfg.solver)
    dense_max_dofs = int(cfg.dense_max_dofs)
    mesh_reg = jnp.asarray(float(cfg.mesh_reg), dtype=dtype)
    mesh_reg_hmin = jnp.asarray(float(cfg.mesh_reg_hmin), dtype=dtype)
    grad_clip = float(cfg.grad_clip)

    def _mesh_reg(bp: Array) -> tuple[Array, Array, Array]:
        """Return (reg_total, reg_smooth, reg_hmin) for one breakpoint vector."""
        h = bp[1:] - bp[:-1]
        if h.shape[0] <= 1:
            z = jnp.asarray(0.0, dtype=dtype)
            return z, z, z
        logh = jnp.log(jnp.maximum(h, jnp.asarray(1e-30, dtype=dtype)))
        smooth = jnp.mean((logh[1:] - logh[:-1]) ** 2)

        h_ref = (bp[-1] - bp[0]) / jnp.asarray(h.shape[0], dtype=dtype)
        hmin_pen = jnp.mean((h_ref / jnp.maximum(h, jnp.asarray(1e-30, dtype=dtype))) ** 2)

        reg_smooth = mesh_reg * smooth
        reg_hmin = mesh_reg_hmin * hmin_pen
        return reg_smooth + reg_hmin, reg_smooth, reg_hmin

    def loss_and_aux(params_in, x0_free_in):
        bp_xL = build_breakpoints_softmax(params_in["xL"], -1.0, 0.0, h_min=float(cfg.h_min))
        bp_xR = build_breakpoints_softmax(params_in["xR"], 0.0, 1.0, h_min=float(cfg.h_min))
        bp_yB = build_breakpoints_softmax(params_in["yB"], -1.0, 0.0, h_min=float(cfg.h_min))
        bp_yT = build_breakpoints_softmax(params_in["yT"], 0.0, 1.0, h_min=float(cfg.h_min))

        u, kxL, kxR, kyB, kyT = solve_lshape_coeffs(
            bp_xL,
            bp_xR,
            bp_yB,
            bp_yT,
            g0,
            g1,
            g2,
            dirichlet,
            free_idx,
            x0_free_in,
            p=p,
            q_order=int(cfg.q_order),
            solver=solver,
            dense_max_dofs=dense_max_dofs,
            cg_tol=float(cfg.cg_tol),
            cg_maxiter=int(cfg.cg_maxiter),
            precond=precond,
            n_dofs=int(n_dofs),
        )

        eta2, (eta2_vol, eta2_jump, eta2_jump01, eta2_jump12) = estimator_eta2(
            u,
            bp_xL=bp_xL,
            bp_xR=bp_xR,
            bp_yB=bp_yB,
            bp_yT=bp_yT,
            knots_xL=kxL,
            knots_xR=kxR,
            knots_yB=kyB,
            knots_yT=kyT,
            g0=g0,
            g1=g1,
            g2=g2,
            p=p,
            q_order=int(cfg.q_order),
            include_jump=include_jump,
            eps_side=float(cfg.eps_side),
        )
        loss_eta = jnp.asarray(0.5, dtype=eta2.dtype) * eta2

        reg_xL, reg_xL_s, reg_xL_h = _mesh_reg(bp_xL)
        reg_xR, reg_xR_s, reg_xR_h = _mesh_reg(bp_xR)
        reg_yB, reg_yB_s, reg_yB_h = _mesh_reg(bp_yB)
        reg_yT, reg_yT_s, reg_yT_h = _mesh_reg(bp_yT)
        reg_smooth = reg_xL_s + reg_xR_s + reg_yB_s + reg_yT_s
        reg_hmin = reg_xL_h + reg_xR_h + reg_yB_h + reg_yT_h
        reg_total = reg_xL + reg_xR + reg_yB + reg_yT

        loss = loss_eta + reg_total
        return loss, (
            bp_xL,
            bp_xR,
            bp_yB,
            bp_yT,
            u,
            kxL,
            kxR,
            kyB,
            kyT,
            eta2,
            eta2_vol,
            eta2_jump,
            eta2_jump01,
            eta2_jump12,
            loss_eta,
            reg_smooth,
            reg_hmin,
        )

    loss_and_grad = jax.value_and_grad(loss_and_aux, argnums=0, has_aux=True)

    @jax.jit
    def step(params_in, opt_state_in, x0_free_in):
        (loss, aux), grads = loss_and_grad(params_in, x0_free_in)
        gnorm = jnp.sqrt(
            jnp.sum(grads["xL"] * grads["xL"])
            + jnp.sum(grads["xR"] * grads["xR"])
            + jnp.sum(grads["yB"] * grads["yB"])
            + jnp.sum(grads["yT"] * grads["yT"])
        )
        if grad_clip > 0.0:
            scale = jnp.minimum(1.0, jnp.asarray(grad_clip, dtype=gnorm.dtype) / (gnorm + jnp.asarray(1e-30, dtype=gnorm.dtype)))
            grads = {k: v * scale for k, v in grads.items()}
            gnorm = gnorm * scale
        updates, opt_state_out = opt.update(grads, opt_state_in, params_in)
        params_out = optax.apply_updates(params_in, updates)
        u = aux[4]
        x0_free_out = u[free_idx]
        return params_out, opt_state_out, x0_free_out, loss, aux, gnorm

    history: list[Dict[str, Any]] = []
    t0 = time.perf_counter()

    last_bp = None
    last_u = None
    last_knots = None

    for it in range(int(cfg.iters) + 1):
        params, opt_state, x0_free, loss, aux, gnorm = step(params, opt_state, x0_free)
        bp_xL, bp_xR, bp_yB, bp_yT, u, kxL, kxR, kyB, kyT, eta2, eta2_vol, eta2_jump, eta2_jump01, eta2_jump12, loss_eta, reg_smooth, reg_hmin = aux

        last_bp = (bp_xL, bp_xR, bp_yB, bp_yT)
        last_u = u
        last_knots = (kxL, kxR, kyB, kyT)

        if (it % int(cfg.log_every) == 0) or (it == int(cfg.iters)):
            # Validation loss (higher quadrature)
            eta2_val, _ = estimator_eta2(
                u,
                bp_xL=bp_xL,
                bp_xR=bp_xR,
                bp_yB=bp_yB,
                bp_yT=bp_yT,
                knots_xL=kxL,
                knots_xR=kxR,
                knots_yB=kyB,
                knots_yT=kyT,
                g0=g0,
                g1=g1,
                g2=g2,
                p=p,
                q_order=int(cfg.q_order_val),
                include_jump=include_jump,
                eps_side=float(cfg.eps_side),
            )

            metrics = compute_error_metrics(
                u,
                bp_xL=bp_xL,
                bp_xR=bp_xR,
                bp_yB=bp_yB,
                bp_yT=bp_yT,
                g0=g0,
                g1=g1,
                g2=g2,
                p=p,
                q_order=int(cfg.q_order_val),
            )

            bp_xL_np = jax.device_get(bp_xL)
            bp_xR_np = jax.device_get(bp_xR)
            bp_yB_np = jax.device_get(bp_yB)
            bp_yT_np = jax.device_get(bp_yT)
            h_min = float(
                min(
                    float((bp_xL_np[1:] - bp_xL_np[:-1]).min()),
                    float((bp_xR_np[1:] - bp_xR_np[:-1]).min()),
                    float((bp_yB_np[1:] - bp_yB_np[:-1]).min()),
                    float((bp_yT_np[1:] - bp_yT_np[:-1]).min()),
                )
            )

            rec = {
                "iter": int(it),
                "loss": float(jax.device_get(loss)),
                "loss_eta": float(jax.device_get(loss_eta)),
                "loss_reg_smooth": float(jax.device_get(reg_smooth)),
                "loss_reg_hmin": float(jax.device_get(reg_hmin)),
                "loss_val": float(0.5 * float(jax.device_get(eta2_val))),
                "eta": float(np.sqrt(float(jax.device_get(eta2)))),
                "eta_val": float(np.sqrt(float(jax.device_get(eta2_val)))),
                "eta2_total": float(jax.device_get(eta2)),
                "eta2_vol": float(jax.device_get(eta2_vol)),
                "eta2_jump": float(jax.device_get(eta2_jump)),
                "eta2_jump01": float(jax.device_get(eta2_jump01)),
                "eta2_jump12": float(jax.device_get(eta2_jump12)),
                "H1_rel": float(metrics["H1_rel"]),
                "L2_rel": float(metrics["L2_rel"]),
                "dG_err": float(metrics["dG_err"]),
                "h_min": float(h_min),
                "dofs": int(metrics["dofs"]),
                "lr": float(schedule(it)),
                "gnorm": float(jax.device_get(gnorm)),
                "time_total_sec": float(time.perf_counter() - t0),
            }
            history.append(rec)
            N_disp = int(max(n_xL, n_xR, n_yB, n_yT))
            print(
                f"[N={N_disp:3d} it {it:5d}] loss={rec['loss']:.3e} val={rec['loss_val']:.3e} | "
                f"H1_rel={rec['H1_rel']:.3e} | L2_rel={rec['L2_rel']:.3e} | h_min={rec['h_min']:.2e}"
            )

    elapsed = time.perf_counter() - t0

    if last_bp is None or last_u is None or last_knots is None:
        raise RuntimeError("Training produced no iterates.")

    bp_xL, bp_xR, bp_yB, bp_yT = last_bp
    out = {
        "breakpoints": {
            "xL": jax.device_get(bp_xL),
            "xR": jax.device_get(bp_xR),
            "yB": jax.device_get(bp_yB),
            "yT": jax.device_get(bp_yT),
        },
        "knots": {
            "xL": jax.device_get(last_knots[0]),
            "xR": jax.device_get(last_knots[1]),
            "yB": jax.device_get(last_knots[2]),
            "yT": jax.device_get(last_knots[3]),
        },
        "u": jax.device_get(last_u),
        "history": history,
        "elapsed_sec": float(elapsed),
        "dofs": int(jnp.asarray(last_u).size),
        "cfg": cfg.__dict__,
        "n_dofs": int(n_dofs),
    }
    return out


__all__ = [
    "TrainConfig",
    "train_estimator_adam",
]
