"""Online L-BFGS-B corrector for ADVDIFF (advection–diffusion).

Mirrors ``corrector_v2.corrector_arctan`` in spirit: take the network's mesh
prediction as a warm-start and refine it locally with a few L-BFGS-B
iterations that minimise the SAME residual loss used in training,

    L_p4(m; ν) = η²(m; ν) / (denom(ν) + ε),

over the per-axis mesh logits ``m = (z_x, z_y)``. ``denom(ν) + ε`` is a
constant per ν, so the argmin is the same as minimising η²; we keep the
normalisation so ``loss_init`` / ``loss_final`` are directly comparable
to the training-loss values.

REUSE vs DUPLICATE
------------------
* REUSED: ``radapt_local`` + ``RAdaptResult`` (the L-BFGS-B wrapper from
  ``r_adapt``), and ``policy_step`` / ``forward`` / ``cell_midpoints``
  from the positional network module. The forward solve + estimator are
  the existing ``galerkin_solve_advdiff`` / ``eta_squared_advdiff``.
* DUPLICATED (deliberately): the ``loss_fn`` / ``obj`` scaffolding. We
  cannot reuse ``training_advdiff.loss_p4_single_nu`` because that function
  differentiates w.r.t. the *network parameters* and rebuilds the mesh
  from ``params`` via the policy — the corrector instead differentiates
  w.r.t. the *mesh logits* directly. This is the same reason
  ``corrector_v2`` does not reuse ``loss_p2_single_sigma``.

KNOT BUILDER — matches the ADVDIFF policy exactly
--------------------------------------------
``corrector_v2`` rebuilds knots inside its loss via ``theta_to_knots_arctan``
(plain ``softmax → cumsum``). The ADVDIFF network policy
(``knots_p2_from_network_axis``) instead uses ``policy_step`` (gauge-fix
+ ``T·tanh`` + softmax + ``h_min`` floor). To make the warm-start
reproduce the *positional* mesh exactly — so ``loss_init`` equals the
positional η²/denom — this corrector uses the **policy** map
(``_knots_from_logits_advdiff`` = ``policy_step → cumsum``), not the plain
softmax. The warm-start logits are the raw network output
``z = forward(params, ν, ξ, axis_id)``; ``_knots_from_logits_advdiff(z)`` then
equals ``knots_p2_from_network_axis(params, ν, …)`` bit-for-bit.

Box constraint: logits are constrained to ``[-T, T]`` (T defaults to
``T_ARCTAN``; ADVDIFF reuses ARCTAN's unified mesh-policy constants), mirroring
``corrector_v2``.
"""
from __future__ import annotations

from dataclasses import dataclass

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from src.nonparametric.r_adapt import RAdaptResult, radapt_local
from src.parametric.positional_density_network_2d import policy_step
from src.nonparametric.advdiff.solver_advdiff import galerkin_solve_advdiff
from src.nonparametric.advdiff.eta_estimator_advdiff import eta_squared_advdiff

Array = jnp.ndarray


@dataclass
class CorrectorP4Result:
    """Output of ``corrector_advdiff``.

    Attributes
    ----------
    knots_x, knots_y : Array
        The corrected knot vectors (ready for the solver / metrics).
    theta_x, theta_y : Array
        The corrected per-axis logits (the optimisation variables).
    n_iter : int
        L-BFGS-B iterations actually taken (<= max_iter).
    converged : bool
        Whether L-BFGS-B reported convergence.
    loss_init : float
        L_p4 at the warm-start (== positional η²/denom).
    loss_final : float
        L_p4 at the corrected mesh.
    message : str
        L-BFGS-B termination message.
    """
    knots_x: Array
    knots_y: Array
    theta_x: Array
    theta_y: Array
    n_iter: int
    converged: bool
    loss_init: float
    loss_final: float
    message: str


def _knots_from_logits_advdiff(z: Array, p: int, n_elem: int, *, T: float, h_min: float) -> Array:
    """Logits -> open-uniform knot vector via the ADVDIFF mesh policy.

    Identical to ``positional_density_network_2d.knots_p2_from_network_axis``
    except it takes raw logits ``z`` directly (instead of running the
    network forward internally), so the corrector can optimise over ``z``:

        sizes  = policy_step(z, T, h_min, budget=1.0)   # gauge + T·tanh + softmax + h_min
        interior = cumsum(sizes)[:-1]
        knots  = [0]*(p+1) ++ interior ++ [1]*(p+1).
    """
    sizes = policy_step(z, T=float(T), h_min=float(h_min), budget=1.0)
    interior = jnp.cumsum(sizes)[:-1]
    zeros = jnp.zeros((p + 1,), dtype=sizes.dtype)
    ones = jnp.ones((p + 1,), dtype=sizes.dtype)
    return jnp.concatenate([zeros, interior, ones])


def corrector_advdiff(
    z_x_init: Array,
    z_y_init: Array,
    nu: Array,                         # (2,) = (logeps, b)
    denom: float,
    *,
    p: int,
    n_elem: int,
    q_K: int = 4,
    q_F: int = 50,
    q_est: int = 50,
    max_iter: int = 30,
    T: float | None = None,
    h_min: float | None = None,
    ftol: float | None = None,
    gtol: float | None = None,
    epsilon: float | None = None,
) -> CorrectorP4Result:
    """Local L-BFGS-B on ``(z_x, z_y)`` to minimise the ADVDIFF residual loss
    ``η²(m; ν) / (denom + ε)`` for one ν-tuple, warm-started from the
    network's logits.

    Parameters
    ----------
    z_x_init, z_y_init : raw network logits ``forward(params, ν, ξ, axis)``.
    nu : (logeps, b); ``eps = 10**logeps``, ``b = nu[1]``.
    denom : per-ν loss denominator (analytic ``|u*_ν|²_{H¹}``).
    max_iter : L-BFGS-B iteration cap (default 30, matching LSHAPE eval).
    T, h_min : mesh-policy constants (default ``T_ARCTAN`` / ``H_MIN_ARCTAN``).
    ftol, gtol : L-BFGS-B tolerances (default ``CORRECTOR['ftol'/'gtol']``).
    epsilon : denominator regulariser (default ``EPSILON_DENOM``).
    """
    from src.config import CORRECTOR, EPSILON_DENOM, T_ARCTAN, H_MIN_ARCTAN

    T_val = float(T_ARCTAN) if T is None else float(T)
    h_min_val = float(H_MIN_ARCTAN) if h_min is None else float(h_min)
    ftol_val = float(CORRECTOR["ftol"]) if ftol is None else float(ftol)
    gtol_val = float(CORRECTOR["gtol"]) if gtol is None else float(gtol)
    eps_reg = float(EPSILON_DENOM) if epsilon is None else float(epsilon)

    nu = jnp.asarray(nu, dtype=DEFAULT_DTYPE).reshape(-1)
    eps = jnp.power(jnp.asarray(10.0, dtype=DEFAULT_DTYPE), nu[0])
    b = nu[1]
    denom_t = jnp.asarray(float(denom) + eps_reg, dtype=DEFAULT_DTYPE)

    z_x_init = np.asarray(z_x_init, dtype=np.float64).reshape(-1)
    z_y_init = np.asarray(z_y_init, dtype=np.float64).reshape(-1)
    nx = int(z_x_init.shape[0])
    ny = int(z_y_init.shape[0])
    theta_flat0 = np.concatenate([z_x_init, z_y_init])

    def loss_fn(theta_flat):
        zx = theta_flat[:nx]
        zy = theta_flat[nx:]
        knots_x = _knots_from_logits_advdiff(zx, p, n_elem, T=T_val, h_min=h_min_val)
        knots_y = _knots_from_logits_advdiff(zy, p, n_elem, T=T_val, h_min=h_min_val)
        res = galerkin_solve_advdiff(
            knots_x, knots_y, p, n_elem, n_elem, eps, b, q_K=q_K, q_F=q_F,
        )
        eta_sq = eta_squared_advdiff(
            res.u_h, knots_x, knots_y, p, n_elem, n_elem, eps, b, q_est=q_est,
        )
        return eta_sq / denom_t

    loss_and_grad = jax.value_and_grad(loss_fn)

    def obj(theta_np):
        theta_jax = jnp.asarray(theta_np, dtype=DEFAULT_DTYPE)
        loss, grad = loss_and_grad(theta_jax)
        return float(loss), np.asarray(grad, dtype=np.float64)

    # Warm-start loss (== positional η²/denom, since the builder matches
    # the policy and z is the raw network output).
    loss_init, _ = obj(theta_flat0)

    bounds = [(-T_val, T_val)] * (nx + ny)
    result: RAdaptResult = radapt_local(
        obj, theta_flat0,
        max_iter=int(max_iter), ftol=ftol_val, gtol=gtol_val, bounds=bounds,
    )

    theta_x_opt = jnp.asarray(result.theta_final[:nx], dtype=DEFAULT_DTYPE)
    theta_y_opt = jnp.asarray(result.theta_final[nx:], dtype=DEFAULT_DTYPE)
    knots_x_opt = _knots_from_logits_advdiff(theta_x_opt, p, n_elem, T=T_val, h_min=h_min_val)
    knots_y_opt = _knots_from_logits_advdiff(theta_y_opt, p, n_elem, T=T_val, h_min=h_min_val)

    return CorrectorP4Result(
        knots_x=knots_x_opt,
        knots_y=knots_y_opt,
        theta_x=theta_x_opt,
        theta_y=theta_y_opt,
        n_iter=int(result.n_iter),
        converged=bool(result.converged),
        loss_init=float(loss_init),
        loss_final=float(result.energy_final),
        message=str(result.message),
    )


__all__ = [
    "CorrectorP4Result",
    "corrector_advdiff",
    "_knots_from_logits_advdiff",
]
