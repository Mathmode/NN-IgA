from __future__ import annotations

"""Training and evaluation API for the 1D power-solution experiment."""

import functools
import math
import time
from typing import Any

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from src.nonparametric.discretization import solve_state
from src.nonparametric.effectivity import effectivity_diagnostics_from_state
from src.nonparametric.knots import H_MIN_DEFAULT
from src.nonparametric.metrics import diagnostic_metrics_from_state
from src.nonparametric.pde import BETA, exact_h1_norm_power
from src.nonparametric.quadrature_analytic import compute_validation_loss, reduced_loss, residual_loss_terms_from_state
from common.profiling import get_profiler

Array = jnp.ndarray

DEFAULT_P_LIST = (2, 3)
DEFAULT_N_LIST = (2, 4, 8, 16, 32, 64, 128, 256)
RATIO_DENOM_THRESHOLD = 1.0e-14


def iterations_for_level(n_elem: int) -> int:
    """Default iteration budget per refinement level.

    With warmstart the optimizer starts near the minimum, so coarse levels
    need more exploration while fine levels need only minor polishing.
    The exponential LR decay from lr1 to lr2 distributes the budget
    smoothly; there is no abrupt schedule switch to calibrate.
    """
    if n_elem <= 8:
        return 10_000
    if n_elem <= 64:
        return 15_000
    if n_elem <= 128:
        return 20_000
    return 25_000


def build_optimizer(
    iters: int,
    lr1: float = 1e-2,
    lr2: float = 1e-3,
    eps: float = 1e-15,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Adam with exponential LR decay from ``lr1`` to ``lr2``."""
    iters = max(int(iters), 1)
    decay_rate = float(lr2) / float(lr1) if float(lr1) > 0.0 else 1.0
    schedule = optax.exponential_decay(
        init_value=float(lr1),
        transition_steps=iters,
        decay_rate=decay_rate,
        end_value=float(lr2),
    )
    # eps=1e-15: prevents Adam denominator from clipping O(1e-12) gradient
    # components at convergence (see docs/hyperparameter_audit.md §2).
    return optax.adam(schedule, eps=float(eps)), schedule


def normalized_residual_estimator(eta_residual: Array, beta: float = BETA) -> Array:
    return eta_residual / jnp.asarray(exact_h1_norm_power(float(beta)), dtype=eta_residual.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("degree", "quadrature_order", "beta"),
)
def solve_state_and_metrics(
    theta: Array,
    degree: int,
    quadrature_order: int,
    h_min: float,
    beta: float = BETA,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    u_full, cache_train = solve_state(theta, degree, quadrature_order, h_min, beta)
    loss_terms = residual_loss_terms_from_state(u_full, cache_train, degree, beta)
    diagnostics = diagnostic_metrics_from_state(u_full, cache_train, degree, beta)

    eta_residual = jnp.sqrt(
        jnp.maximum(2.0 * loss_terms.total_loss, jnp.asarray(0.0, dtype=theta.dtype))
    )
    eta_rel = normalized_residual_estimator(eta_residual, beta)
    energy_threshold = jnp.asarray(RATIO_DENOM_THRESHOLD, dtype=theta.dtype)
    osc_aware = jnp.sqrt(
        jnp.maximum(
            diagnostics.err_energy_abs * diagnostics.err_energy_abs + diagnostics.osc_sq,
            jnp.asarray(0.0, dtype=theta.dtype),
        )
    )
    i_eff_energy_raw = jnp.where(
        diagnostics.err_energy_abs > energy_threshold,
        eta_residual / diagnostics.err_energy_abs,
        jnp.asarray(jnp.nan, dtype=theta.dtype),
    )
    i_eff_energy_plus_osc = jnp.where(
        osc_aware > energy_threshold,
        eta_residual / osc_aware,
        jnp.asarray(jnp.nan, dtype=theta.dtype),
    )

    return (
        u_full,
        cache_train.knots,
        cache_train.sizes,
        loss_terms.total_loss,
        loss_terms.volume_loss,
        loss_terms.neumann_loss,
        diagnostics.err_l2_abs,
        diagnostics.err_l2_rel,
        diagnostics.err_energy_abs,
        diagnostics.err_energy_rel,
        diagnostics.err_full_h1_abs,
        diagnostics.err_full_h1_rel,
        diagnostics.osc_sq,
        diagnostics.osc_abs,
        eta_residual,
        eta_rel,
        i_eff_energy_raw,
        i_eff_energy_plus_osc,
    )


def metric_dict_from_state(
    *,
    u_full: Array,
    knots: Array,
    sizes: Array,
    total_loss: Array,
    volume_loss: Array,
    neumann_loss: Array,
    err_l2_abs: Array,
    err_l2_rel: Array,
    err_energy_abs: Array,
    err_energy_rel: Array,
    err_full_h1_abs: Array,
    err_full_h1_rel: Array,
    osc_sq: Array,
    osc_abs: Array,
    eta_residual: Array,
    eta_rel: Array,
    i_eff_energy_raw: Array,
    i_eff_energy_plus_osc: Array,
) -> dict[str, Any]:
    (
        u_full_np,
        knots_np,
        sizes_np,
        total_loss_np,
        volume_loss_np,
        neumann_loss_np,
        err_l2_abs_np,
        err_l2_rel_np,
        err_energy_abs_np,
        err_energy_rel_np,
        err_full_h1_abs_np,
        err_full_h1_rel_np,
        osc_sq_np,
        osc_abs_np,
        eta_residual_np,
        eta_rel_np,
        i_eff_energy_raw_np,
        i_eff_energy_plus_osc_np,
    ) = jax.device_get(
        (
            u_full,
            knots,
            sizes,
            total_loss,
            volume_loss,
            neumann_loss,
            err_l2_abs,
            err_l2_rel,
            err_energy_abs,
            err_energy_rel,
            err_full_h1_abs,
            err_full_h1_rel,
            osc_sq,
            osc_abs,
            eta_residual,
            eta_rel,
            i_eff_energy_raw,
            i_eff_energy_plus_osc,
        )
    )

    sizes_arr = np.asarray(sizes_np, dtype=np.float64)
    return {
        "u_full": u_full_np,
        "knots": knots_np,
        "sizes": sizes_np,
        "total_loss": float(total_loss_np),
        "volume_loss": float(volume_loss_np),
        "neumann_loss": float(neumann_loss_np),
        "err_l2_abs": float(err_l2_abs_np),
        "err_l2_rel": float(err_l2_rel_np),
        "err_energy_abs": float(err_energy_abs_np),
        "err_energy_rel": float(err_energy_rel_np),
        "err_full_h1_abs": float(err_full_h1_abs_np),
        "err_full_h1_rel": float(err_full_h1_rel_np),
        "osc_sq": float(osc_sq_np),
        "osc_abs": float(osc_abs_np),
        "eta_residual": float(eta_residual_np),
        "eta_rel": float(eta_rel_np),
        "I_eff_energy_raw": float(i_eff_energy_raw_np),
        "I_eff_energy_plus_osc": float(i_eff_energy_plus_osc_np),
        "mesh_hmin": float(np.min(sizes_arr)),
    }


def evaluate_case(
    theta: Array,
    degree: int,
    quadrature_order: int,
    h_min: float,
    beta: float = BETA,
) -> dict[str, Any]:
    (
        u_full,
        knots,
        sizes,
        total_loss,
        volume_loss,
        neumann_loss,
        err_l2_abs,
        err_l2_rel,
        err_energy_abs,
        err_energy_rel,
        err_full_h1_abs,
        err_full_h1_rel,
        osc_sq,
        osc_abs,
        eta_residual,
        eta_rel,
        i_eff_energy_raw,
        i_eff_energy_plus_osc,
    ) = solve_state_and_metrics(theta, degree, quadrature_order, h_min, beta)
    return metric_dict_from_state(
        u_full=u_full,
        knots=knots,
        sizes=sizes,
        total_loss=total_loss,
        volume_loss=volume_loss,
        neumann_loss=neumann_loss,
        err_l2_abs=err_l2_abs,
        err_l2_rel=err_l2_rel,
        err_energy_abs=err_energy_abs,
        err_energy_rel=err_energy_rel,
        err_full_h1_abs=err_full_h1_abs,
        err_full_h1_rel=err_full_h1_rel,
        osc_sq=osc_sq,
        osc_abs=osc_abs,
        eta_residual=eta_residual,
        eta_rel=eta_rel,
        i_eff_energy_raw=i_eff_energy_raw,
        i_eff_energy_plus_osc=i_eff_energy_plus_osc,
    )


def evaluate_case_detailed(
    theta: Array,
    degree: int,
    quadrature_order: int,
    h_min: float,
    beta: float = BETA,
) -> dict[str, Any]:
    u_full, cache = solve_state(theta, degree, quadrature_order, h_min, beta)
    loss_terms = residual_loss_terms_from_state(u_full, cache, degree, beta)
    diagnostics = diagnostic_metrics_from_state(u_full, cache, degree, beta)

    eta_residual = jnp.sqrt(
        jnp.maximum(2.0 * loss_terms.total_loss, jnp.asarray(0.0, dtype=u_full.dtype))
    )
    eta_rel = normalized_residual_estimator(eta_residual, beta)
    energy_threshold = jnp.asarray(RATIO_DENOM_THRESHOLD, dtype=u_full.dtype)
    osc_aware = jnp.sqrt(
        jnp.maximum(
            diagnostics.err_energy_abs * diagnostics.err_energy_abs + diagnostics.osc_sq,
            jnp.asarray(0.0, dtype=u_full.dtype),
        )
    )
    i_eff_energy_raw = jnp.where(
        diagnostics.err_energy_abs > energy_threshold,
        eta_residual / diagnostics.err_energy_abs,
        jnp.asarray(jnp.nan, dtype=u_full.dtype),
    )
    i_eff_energy_plus_osc = jnp.where(
        osc_aware > energy_threshold,
        eta_residual / osc_aware,
        jnp.asarray(jnp.nan, dtype=u_full.dtype),
    )

    metric_dict = metric_dict_from_state(
        u_full=u_full,
        knots=cache.knots,
        sizes=cache.sizes,
        total_loss=loss_terms.total_loss,
        volume_loss=loss_terms.volume_loss,
        neumann_loss=loss_terms.neumann_loss,
        err_l2_abs=diagnostics.err_l2_abs,
        err_l2_rel=diagnostics.err_l2_rel,
        err_energy_abs=diagnostics.err_energy_abs,
        err_energy_rel=diagnostics.err_energy_rel,
        err_full_h1_abs=diagnostics.err_full_h1_abs,
        err_full_h1_rel=diagnostics.err_full_h1_rel,
        osc_sq=diagnostics.osc_sq,
        osc_abs=diagnostics.osc_abs,
        eta_residual=eta_residual,
        eta_rel=eta_rel,
        i_eff_energy_raw=i_eff_energy_raw,
        i_eff_energy_plus_osc=i_eff_energy_plus_osc,
    )
    metric_dict.update(
        effectivity_diagnostics_from_state(
            u_full,
            cache,
            degree,
            metric_dict["err_energy_abs"],
            beta=beta,
            loss_terms=loss_terms,
        )
    )
    return metric_dict


def history_row(
    iteration: int,
    eval_data: dict[str, Any],
    *,
    loss_val: float,
    grad_norm: float,
    lr: float,
    theta_span: float = float("nan"),
) -> dict[str, float]:
    return {
        "iter": float(iteration),
        "total_loss": float(eval_data["total_loss"]),
        "loss_val": float(loss_val),
        "volume_loss": float(eval_data["volume_loss"]),
        "neumann_loss": float(eval_data["neumann_loss"]),
        "eta_residual": float(eval_data["eta_residual"]),
        "eta_rel": float(eval_data["eta_rel"]),
        "err_l2_abs": float(eval_data["err_l2_abs"]),
        "err_l2_rel": float(eval_data["err_l2_rel"]),
        "err_energy_abs": float(eval_data["err_energy_abs"]),
        "err_energy_rel": float(eval_data["err_energy_rel"]),
        "err_full_h1_abs": float(eval_data["err_full_h1_abs"]),
        "err_full_h1_rel": float(eval_data["err_full_h1_rel"]),
        "osc_abs": float(eval_data["osc_abs"]),
        "I_eff_energy_raw": float(eval_data["I_eff_energy_raw"]),
        "I_eff_energy_plus_osc": float(eval_data["I_eff_energy_plus_osc"]),
        "grad_norm": float(grad_norm),
        "lr": float(lr),
        "mesh_hmin": float(eval_data["mesh_hmin"]),
        "theta_span": float(theta_span),
    }


def final_metric_dict(
    *,
    eval_data: dict[str, Any],
    iters: float,
    loss_val: float,
    grad_norm_final: float,
    elapsed_sec: float,
) -> dict[str, float]:
    keys = (
        "total_loss",
        "volume_loss",
        "neumann_loss",
        "eta_residual",
        "eta_rel",
        "eta_exact",
        "eta2_exact",
        "eta_proj",
        "eta2_proj",
        "osc",
        "osc_abs",
        "osc_sq",
        "eta_N",
        "eta2_N",
        "eta_J",
        "eta2_J",
        "err_l2_abs",
        "err_l2_rel",
        "err_energy_abs",
        "err_energy_rel",
        "err_h1_semi_abs",
        "err_full_h1_abs",
        "err_full_h1_rel",
        "I_eff_energy_raw",
        "I_eff_energy_plus_osc",
        "I_eff_exact",
        "I_eff_proj",
        "I_osc",
        "frac_proj",
        "frac_osc",
        "frac_N",
        "frac_J",
        "solve_residual_rel",
        "cond_K",
        "identity_eta2_abs",
        "identity_eta2_rel",
        "identity_elem_max_abs",
        "identity_elem_l1",
        "mesh_hmin",
        "used_gl64_error_fallback",
    )
    metrics = {key: float(eval_data[key]) for key in keys}
    metrics.update(
        {
            "iters": float(iters),
            "loss_val": float(loss_val),
            "grad_norm_final": float(grad_norm_final),
            "elapsed_sec": float(elapsed_sec),
        }
    )
    return metrics


def build_training_kernels(
    degree: int,
    quadrature_order: int,
    h_min: float,
    iters: int,
    history_every: int,
    beta: float = BETA,
    lr1: float = 1e-2,
    lr2: float = 1e-3,
) -> tuple[optax.GradientTransformation, optax.Schedule, Any, Any, Any]:
    loss_fn = lambda th: reduced_loss(th, degree, quadrature_order, h_min, beta)
    value_and_grad = jax.value_and_grad(loss_fn)
    optimizer, schedule = build_optimizer(iters, lr1=lr1, lr2=lr2)

    @jax.jit
    def loss_and_grad(theta: Array):
        loss, grad = value_and_grad(theta)
        grad_norm = jnp.linalg.norm(grad)
        return loss, grad, grad_norm

    @jax.jit
    def advance_n_steps(
        theta: Array,
        opt_state,
        best_theta: Array,
        best_loss: Array,
        n_steps: Array,
    ):
        zero = jnp.asarray(0.0, dtype=theta.dtype)
        n_steps = jnp.asarray(n_steps, dtype=jnp.int32)

        def cond_fn(carry):
            i, *_rest = carry
            return i < n_steps

        def body_fn(carry):
            i, theta_i, opt_state_i, best_theta_i, best_loss_i, _loss_i, _grad_norm_i = carry
            loss_i, grad_i = value_and_grad(theta_i)
            grad_norm_i = jnp.linalg.norm(grad_i)
            better = loss_i < best_loss_i
            best_theta_i = jnp.where(better, theta_i, best_theta_i)
            best_loss_i = jnp.where(better, loss_i, best_loss_i)
            updates, opt_state_o = optimizer.update(grad_i, opt_state_i, theta_i)
            theta_o = optax.apply_updates(theta_i, updates)
            return (
                i + 1,
                theta_o,
                opt_state_o,
                best_theta_i,
                best_loss_i,
                loss_i,
                grad_norm_i,
            )

        carry0 = (
            jnp.asarray(0, dtype=jnp.int32),
            theta,
            opt_state,
            best_theta,
            best_loss,
            zero,
            zero,
        )
        carryf = lax.while_loop(cond_fn, body_fn, carry0)
        _i, theta_f, opt_state_f, best_theta_f, best_loss_f, _loss_last, _grad_norm_last = carryf

        final_loss, _grad, final_grad_norm = loss_and_grad(theta_f)
        better_f = final_loss < best_loss_f
        best_theta_f = jnp.where(better_f, theta_f, best_theta_f)
        best_loss_f = jnp.where(better_f, final_loss, best_loss_f)
        return theta_f, opt_state_f, best_theta_f, best_loss_f, final_loss, final_grad_norm

    @jax.jit
    def advance_chunk(theta: Array, opt_state, best_theta: Array, best_loss: Array):
        return advance_n_steps(
            theta,
            opt_state,
            best_theta,
            best_loss,
            jnp.asarray(history_every, dtype=jnp.int32),
        )

    @jax.jit
    def advance_remainder(
        theta: Array,
        opt_state,
        best_theta: Array,
        best_loss: Array,
        n_steps: int,
    ):
        return advance_n_steps(theta, opt_state, best_theta, best_loss, jnp.asarray(n_steps, dtype=jnp.int32))

    return optimizer, schedule, loss_and_grad, advance_chunk, advance_remainder


def run_uniform_case(
    n_elem: int,
    degree: int,
    quadrature_order: int,
    h_min: float,
    beta: float = BETA,
) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, float]]:
    theta = jnp.zeros((n_elem,), dtype=DEFAULT_DTYPE)

    start = time.perf_counter()
    eval_data = evaluate_case_detailed(theta, degree, quadrature_order, h_min, beta)
    loss_val = compute_validation_loss(
        theta,
        degree,
        h_min,
        beta=beta,
        train_loss=float(eval_data["total_loss"]),
    )
    elapsed = time.perf_counter() - start

    history = [history_row(0, eval_data, loss_val=loss_val, grad_norm=math.nan, lr=0.0)]
    metrics = final_metric_dict(
        eval_data=eval_data,
        iters=0.0,
        loss_val=loss_val,
        grad_norm_final=math.nan,
        elapsed_sec=elapsed,
    )
    return eval_data, history, metrics


def run_r_adapt_case(
    theta0: Array,
    n_elem: int,
    degree: int,
    quadrature_order: int,
    h_min: float,
    *,
    history_every: int,
    iters_override: int | None,
    beta: float = BETA,
    lr1: float = 1e-2,
    lr2: float = 1e-3,
) -> tuple[Array, dict[str, Any], list[dict[str, float]], dict[str, float], Array]:
    """Run r-adaptive optimization.

    Returns
    -------
    theta, eval_data, history, metrics, theta
        ``theta`` is the FINAL iterate — used for reporting AND warmstarting.
        ``best_loss`` is recorded in metrics for diagnostics only.
    """
    iters = int(iters_override) if iters_override is not None else iterations_for_level(n_elem)
    history_every = max(1, min(int(history_every), max(int(iters), 1)))

    optimizer, schedule, loss_and_grad, advance_chunk, advance_remainder = build_training_kernels(
        degree,
        quadrature_order,
        h_min,
        iters,
        history_every,
        beta=beta,
        lr1=lr1,
        lr2=lr2,
    )

    theta = jnp.asarray(theta0, dtype=DEFAULT_DTYPE)
    opt_state = optimizer.init(theta)

    initial_loss, _initial_grad, initial_grad_norm = loss_and_grad(theta)
    best_theta = theta
    best_loss = initial_loss

    initial_eval = evaluate_case(theta, degree, quadrature_order, h_min, beta)
    initial_loss_val = compute_validation_loss(
        theta,
        degree,
        h_min,
        beta=beta,
        train_loss=float(initial_eval["total_loss"]),
    )
    history: list[dict[str, float]] = [
        history_row(
            0,
            initial_eval,
            loss_val=initial_loss_val,
            grad_norm=float(jax.device_get(initial_grad_norm)),
            lr=float(schedule(0)),
        )
    ]

    prof = get_profiler()
    prof.snapshot_memory("before_training_1d")

    start = time.perf_counter()
    completed = 0

    n_chunks, remainder = divmod(iters, history_every)
    for _ in range(n_chunks):
        prof.begin_step()
        with prof.jax_region("advance_chunk"):
            theta, opt_state, best_theta, best_loss, _loss_f, _grad_norm_f = advance_chunk(
                theta,
                opt_state,
                best_theta,
                best_loss,
            )
        completed += history_every

        with prof.jax_region("evaluate_case"):
            eval_data = evaluate_case(theta, degree, quadrature_order, h_min, beta)
        with prof.jax_region("validation_loss"):
            loss_val = compute_validation_loss(
                theta,
                degree,
                h_min,
                beta=beta,
                train_loss=float(eval_data["total_loss"]),
            )
        with prof.jax_region("grad_norm_eval"):
            current_grad_norm = float(jax.device_get(loss_and_grad(theta)[2]))
        prof.end_step()

        _tspan = float(jax.device_get(jnp.max(theta) - jnp.min(theta)))
        history.append(
            history_row(
                completed,
                eval_data,
                loss_val=loss_val,
                grad_norm=current_grad_norm,
                lr=float(schedule(completed)),
                theta_span=_tspan,
            )
        )

    if remainder:
        prof.begin_step()
        with prof.jax_region("advance_chunk"):
            theta, opt_state, best_theta, best_loss, _loss_f, _grad_norm_f = advance_remainder(
                theta,
                opt_state,
                best_theta,
                best_loss,
                int(remainder),
            )
        completed += remainder

        with prof.jax_region("evaluate_case"):
            eval_data = evaluate_case(theta, degree, quadrature_order, h_min, beta)
        with prof.jax_region("validation_loss"):
            loss_val = compute_validation_loss(
                theta,
                degree,
                h_min,
                beta=beta,
                train_loss=float(eval_data["total_loss"]),
            )
        with prof.jax_region("grad_norm_eval"):
            current_grad_norm = float(jax.device_get(loss_and_grad(theta)[2]))
        prof.end_step()

        if int(history[-1]["iter"]) != completed:
            _tspan = float(jax.device_get(jnp.max(theta) - jnp.min(theta)))
            history.append(
                history_row(
                    completed,
                    eval_data,
                    loss_val=loss_val,
                    grad_norm=current_grad_norm,
                    lr=float(schedule(completed)),
                    theta_span=_tspan,
                )
            )

    elapsed = time.perf_counter() - start
    prof.snapshot_memory("after_training_1d")

    final_eval = evaluate_case_detailed(theta, degree, quadrature_order, h_min, beta)
    final_loss_val = compute_validation_loss(
        theta,
        degree,
        h_min,
        beta=beta,
        train_loss=float(final_eval["total_loss"]),
    )
    grad_norm_final = float(jax.device_get(loss_and_grad(theta)[2]))

    metrics = final_metric_dict(
        eval_data=final_eval,
        iters=float(iters),
        loss_val=final_loss_val,
        grad_norm_final=grad_norm_final,
        elapsed_sec=elapsed,
    )
    return theta, final_eval, history, metrics, theta


def summary_row(
    *,
    degree: int,
    n_elem: int,
    method: str,
    metrics: dict[str, float],
    quadrature_order: int | str,
    quadrature_order_val: int | str,
    h_min: float,
    warmstart: bool,
) -> dict[str, object]:
    return {
        "degree": int(degree),
        "N": int(n_elem),
        "mesh_type": str(method),
        "iters": int(metrics["iters"]),
        "quadrature_order": quadrature_order,
        "quadrature_order_val": quadrature_order_val,
        "h_min_input": float(h_min),
        "mesh_hmin": float(metrics["mesh_hmin"]),
        "total_loss": float(metrics["total_loss"]),
        "volume_loss": float(metrics["volume_loss"]),
        "neumann_loss": float(metrics["neumann_loss"]),
        "eta_residual": float(metrics["eta_residual"]),
        "eta_rel": float(metrics["eta_rel"]),
        "eta_exact": float(metrics["eta_exact"]),
        "eta2_exact": float(metrics["eta2_exact"]),
        "eta_proj": float(metrics["eta_proj"]),
        "eta2_proj": float(metrics["eta2_proj"]),
        "err_l2_abs": float(metrics["err_l2_abs"]),
        "err_l2_rel": float(metrics["err_l2_rel"]),
        "err_energy_abs": float(metrics["err_energy_abs"]),
        "err_energy_rel": float(metrics["err_energy_rel"]),
        "err_h1_semi_abs": float(metrics["err_h1_semi_abs"]),
        "err_full_h1_abs": float(metrics["err_full_h1_abs"]),
        "err_full_h1_rel": float(metrics["err_full_h1_rel"]),
        "osc": float(metrics["osc"]),
        "osc_abs": float(metrics["osc_abs"]),
        "osc_sq": float(metrics["osc_sq"]),
        "eta_N": float(metrics["eta_N"]),
        "eta2_N": float(metrics["eta2_N"]),
        "eta_J": float(metrics["eta_J"]),
        "eta2_J": float(metrics["eta2_J"]),
        "I_eff_energy_raw": float(metrics["I_eff_energy_raw"]),
        "I_eff_energy_plus_osc": float(metrics["I_eff_energy_plus_osc"]),
        "I_eff_exact": float(metrics["I_eff_exact"]),
        "I_eff_proj": float(metrics["I_eff_proj"]),
        "I_osc": float(metrics["I_osc"]),
        "frac_proj": float(metrics["frac_proj"]),
        "frac_osc": float(metrics["frac_osc"]),
        "frac_N": float(metrics["frac_N"]),
        "frac_J": float(metrics["frac_J"]),
        "solve_residual_rel": float(metrics["solve_residual_rel"]),
        "cond_K": float(metrics["cond_K"]),
        "identity_eta2_abs": float(metrics["identity_eta2_abs"]),
        "identity_eta2_rel": float(metrics["identity_eta2_rel"]),
        "identity_elem_max_abs": float(metrics["identity_elem_max_abs"]),
        "identity_elem_l1": float(metrics["identity_elem_l1"]),
        "used_gl64_error_fallback": int(metrics["used_gl64_error_fallback"]),
        "grad_norm_final": float(metrics["grad_norm_final"]),
        "elapsed_sec": float(metrics["elapsed_sec"]),
        "warmstart": int(bool(warmstart)),
    }


def validate_n_list(n_list: list[int]) -> None:
    if not n_list:
        raise ValueError("n-list cannot be empty.")
    if n_list[0] != 2:
        raise ValueError("singular warmstart assumes the first level is N=2.")
    for i in range(len(n_list) - 1):
        if n_list[i + 1] != 2 * n_list[i]:
            raise ValueError("singular warmstart expects dyadic refinement N -> 2N.")


__all__ = [
    "Array",
    "DEFAULT_P_LIST",
    "DEFAULT_N_LIST",
    "H_MIN_DEFAULT",
    "iterations_for_level",
    "build_optimizer",
    "normalized_residual_estimator",
    "solve_state_and_metrics",
    "evaluate_case_detailed",
    "evaluate_case",
    "history_row",
    "build_training_kernels",
    "run_uniform_case",
    "run_r_adapt_case",
    "summary_row",
    "validate_n_list",
]
