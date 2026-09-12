"""H1-seminorm error on a tensor-product B-spline mesh in 2D.

Computes ||grad(u_h - u_analytic)||_L2(Omega)^2 element-by-element with
Gauss-Legendre quadrature. The DIFFERENCE is integrated inside the
quadrature integrand (Aballay 4.3.1; mirrors the v3 fix from singular: never
subtract two large numbers).

This module is independent of the solver: it consumes a coefficient
array on a tensor-product B-spline basis and an analytic gradient
callable.
"""
from __future__ import annotations

import functools

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from common.bspline_basis import basis_batch_for_degree
from common.quadrature import rule_gl_on_elements

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Knot vectors and basis indexing for open-uniform tensor-product B-splines
# --------------------------------------------------------------------------


def open_uniform_knots(n_elem: int, p: int) -> np.ndarray:
    """Open-uniform knot vector on [0, 1] with ``n_elem`` elements and degree ``p``.

    Length = ``n_elem + 2 p + 1``. Multiplicity (p+1) at both endpoints.
    Returns ``np.ndarray`` with float64 dtype.
    """
    n_elem = int(n_elem)
    p = int(p)
    if n_elem < 1 or p < 1:
        raise ValueError(f"need n_elem>=1 and p>=1, got n_elem={n_elem}, p={p}")
    h = 1.0 / float(n_elem)
    interior = (np.arange(1, n_elem, dtype=np.float64) * h)
    return np.concatenate(
        [np.zeros(p + 1, dtype=np.float64), interior, np.ones(p + 1, dtype=np.float64)]
    )


def n_basis_open_uniform(n_elem: int, p: int) -> int:
    return int(n_elem) + int(p)


def element_endpoints(knots: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """``a[e] = knots[p + e]``, ``b[e] = knots[p + e + 1]`` for ``e = 0..n_elem - 1``."""
    p = int(p)
    n_elem = int(knots.shape[0] - 2 * p - 1)
    a = np.asarray(knots[p : p + n_elem], dtype=np.float64)
    b = np.asarray(knots[p + 1 : p + 1 + n_elem], dtype=np.float64)
    return a, b


def local_knot_windows(knots: np.ndarray, p: int) -> np.ndarray:
    """For each element ``e`` build the local window of length ``2 p + 2``
    centred on the element's span. Shape ``(n_elem, 2 p + 2)``."""
    p = int(p)
    n_elem = int(knots.shape[0] - 2 * p - 1)
    out = np.zeros((n_elem, 2 * p + 2), dtype=np.float64)
    for e in range(n_elem):
        out[e] = knots[e : e + 2 * p + 2]
    return out


def element_local_indices(n_elem: int, p: int) -> np.ndarray:
    """For element e the (p+1) active basis functions have global indices
    ``[e, e + 1, ..., e + p]`` (open-uniform convention).

    Returns shape ``(n_elem, p + 1)``.
    """
    n_elem = int(n_elem)
    p = int(p)
    idx = np.arange(p + 1, dtype=np.int64)[None, :] + np.arange(n_elem, dtype=np.int64)[:, None]
    return idx


# --------------------------------------------------------------------------
# Core JAX kernel: seminorm-squared of (grad u_h - grad u_analytic).
# --------------------------------------------------------------------------


def _gather_local_coefficients(C: Array, idx_x: Array, idx_y: Array) -> Array:
    """Gather ``C[idx_x[ex, i], idx_y[ey, j]]`` -> shape ``(nex, ney, p+1, p+1)``.

    ``C`` is the global coefficient matrix of shape ``(n_x, n_y)``.
    All inputs must be JAX-traceable arrays.
    """
    # (nex, p+1) and (ney, p+1) -> broadcasted (nex, ney, p+1, p+1)
    Cx = C[idx_x[:, :, None, None], idx_y[None, None, :, :]]
    return Cx.transpose(0, 2, 1, 3)  # (nex, ney, p+1, p+1)


@functools.partial(jax.jit, static_argnames=("p", "q_per_dim"))
def _h1_seminorm_sq_kernel(
    C: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    grad_x_at_quad: Array,
    grad_y_at_quad: Array,
    a_x: Array,
    b_x: Array,
    a_y: Array,
    b_y: Array,
    U_local_x: Array,
    U_local_y: Array,
    idx_x: Array,
    idx_y: Array,
    q_per_dim: int,
) -> Array:
    """Pure-JAX kernel. ``grad_x/y_at_quad`` are pre-evaluated analytic
    gradient components at the tensor-product GL grid (shape
    ``(n_elem_x, n_elem_y, q, q)``). All other arrays are derived from
    the knots.
    """
    p_i = int(p)
    q = int(q_per_dim)

    # Map GL rules to element intervals.
    xq, wq_x = rule_gl_on_elements(a_x, b_x, q)   # (n_elem_x, q)
    yq, wq_y = rule_gl_on_elements(a_y, b_y, q)   # (n_elem_y, q)

    # Evaluate the basis on each element axis-by-axis.
    Nx, dNx, _ = basis_batch_for_degree(p_i, xq, U_local_x)
    Ny, dNy, _ = basis_batch_for_degree(p_i, yq, U_local_y)
    # Shapes: (n_elem_*, q, p + 1)

    # Gather local coefficients per (ex, ey).
    C_loc = _gather_local_coefficients(C, idx_x, idx_y)
    # Shape: (n_elem_x, n_elem_y, p + 1, p + 1)

    # Compute partial derivatives of u_h at the tensor-product GL grid.
    # u_h_x = sum_{i,j} c_{ij} N'_i(xq) N_j(yq).
    # einsum indices:
    #   c: (ex, ey, i, j)
    #   dNx: (ex, qx, i)
    #   Ny:  (ey, qy, j)
    # -> dux: (ex, ey, qx, qy)
    dux = jnp.einsum("xyij,xqi,ypj->xyqp", C_loc, dNx, Ny)
    duy = jnp.einsum("xyij,xqi,ypj->xyqp", C_loc, Nx, dNy)
    # NB: dummy axis labels "qp" in the output reuse 'q' and 'p'; this is
    # just a label choice and does not refer to polynomial degree.

    # Integrand (gradient difference squared).
    diff_x = dux - grad_x_at_quad
    diff_y = duy - grad_y_at_quad
    integrand = diff_x * diff_x + diff_y * diff_y

    # Tensor product weights.
    # wq_x: (n_elem_x, q), wq_y: (n_elem_y, q).
    w_2d = wq_x[:, None, :, None] * wq_y[None, :, None, :]
    # (n_elem_x, n_elem_y, q, q)

    return jnp.sum(integrand * w_2d)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def h1_seminorm_sq_2d(
    u_h_coeffs: Array,
    knots_x: np.ndarray,
    knots_y: np.ndarray,
    p: int,
    grad_u_analytic_callable,
    *,
    quad_points_per_dim: int = 50,
) -> float:
    """Return ``||grad u_h - grad u_analytic||_L2(Omega)^2`` on a tensor-product
    B-spline mesh.

    Parameters
    ----------
    u_h_coeffs
        Global coefficient matrix of shape ``(n_x, n_y)`` with
        ``n_x = n_basis_open_uniform(n_elem_x, p)`` and similarly for y.
    knots_x, knots_y
        Open-uniform 1D knot vectors.
    p
        Polynomial degree (same in both axes).
    grad_u_analytic_callable
        Callable ``(x, y) -> (g_x, g_y)`` where ``x, y`` are arrays of the
        same shape and the callable returns two arrays of that shape.
    quad_points_per_dim
        GL points per dimension per element. Default 50 (Aballay's choice).

    Returns
    -------
    seminorm_sq : float
        ``||grad u_h - grad u_analytic||_L2(Omega)^2`` as a python float.
    """
    p = int(p)
    q = int(quad_points_per_dim)
    knots_x = np.asarray(knots_x, dtype=np.float64)
    knots_y = np.asarray(knots_y, dtype=np.float64)

    a_x, b_x = element_endpoints(knots_x, p)
    a_y, b_y = element_endpoints(knots_y, p)
    U_local_x = local_knot_windows(knots_x, p)
    U_local_y = local_knot_windows(knots_y, p)
    idx_x = element_local_indices(int(a_x.shape[0]), p)
    idx_y = element_local_indices(int(a_y.shape[0]), p)

    # Pre-evaluate the analytic gradient at the full tensor-product GL grid.
    # We do this OUTSIDE the JIT because the callable may be a Python lambda.
    xq, _ = rule_gl_on_elements(jnp.asarray(a_x), jnp.asarray(b_x), q)
    yq, _ = rule_gl_on_elements(jnp.asarray(a_y), jnp.asarray(b_y), q)
    # xq: (n_elem_x, q), yq: (n_elem_y, q).
    XX = jnp.broadcast_to(xq[:, None, :, None], (xq.shape[0], yq.shape[0], q, q))
    YY = jnp.broadcast_to(yq[None, :, None, :], (xq.shape[0], yq.shape[0], q, q))
    g_x, g_y = grad_u_analytic_callable(XX, YY)

    val = _h1_seminorm_sq_kernel(
        jnp.asarray(u_h_coeffs, dtype=DEFAULT_DTYPE),
        jnp.asarray(knots_x, dtype=DEFAULT_DTYPE),
        jnp.asarray(knots_y, dtype=DEFAULT_DTYPE),
        p,
        jnp.asarray(g_x, dtype=DEFAULT_DTYPE),
        jnp.asarray(g_y, dtype=DEFAULT_DTYPE),
        jnp.asarray(a_x, dtype=DEFAULT_DTYPE),
        jnp.asarray(b_x, dtype=DEFAULT_DTYPE),
        jnp.asarray(a_y, dtype=DEFAULT_DTYPE),
        jnp.asarray(b_y, dtype=DEFAULT_DTYPE),
        jnp.asarray(U_local_x, dtype=DEFAULT_DTYPE),
        jnp.asarray(U_local_y, dtype=DEFAULT_DTYPE),
        jnp.asarray(idx_x, dtype=jnp.int32),
        jnp.asarray(idx_y, dtype=jnp.int32),
        q,
    )
    return float(val)


def h1_seminorm_sq_analytic(
    grad_u_analytic_callable,
    *,
    domain_bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
    quad_points_per_dim: int = 50,
    n_subdiv_per_dim: int = 32,
) -> float:
    """Reference value ``||grad u_analytic||_L2(Omega)^2`` on a uniform GL tessellation.

    The domain is split into ``n_subdiv_per_dim x n_subdiv_per_dim`` equal cells
    and each cell uses GL ``quad_points_per_dim^2`` points. With the default
    32x32 sub-cells and q=50 we comfortably resolve arctangent fields with
    alpha up to ~20 (the Aballay regime) to ~1e-13 relative precision.
    """
    (ax, bx), (ay, by) = domain_bounds
    q = int(quad_points_per_dim)
    n_sub = int(n_subdiv_per_dim)
    if n_sub < 1:
        raise ValueError(f"n_subdiv_per_dim must be >= 1; got {n_sub}")

    a_x = jnp.linspace(float(ax), float(bx), n_sub + 1)[:-1]
    b_x = jnp.linspace(float(ax), float(bx), n_sub + 1)[1:]
    a_y = jnp.linspace(float(ay), float(by), n_sub + 1)[:-1]
    b_y = jnp.linspace(float(ay), float(by), n_sub + 1)[1:]
    xq, wq_x = rule_gl_on_elements(a_x, b_x, q)   # (n_sub, q)
    yq, wq_y = rule_gl_on_elements(a_y, b_y, q)
    XX = jnp.broadcast_to(xq[:, None, :, None], (n_sub, n_sub, q, q))
    YY = jnp.broadcast_to(yq[None, :, None, :], (n_sub, n_sub, q, q))
    g_x, g_y = grad_u_analytic_callable(XX, YY)
    integrand = g_x * g_x + g_y * g_y
    w_2d = wq_x[:, None, :, None] * wq_y[None, :, None, :]
    return float(jnp.sum(integrand * w_2d))


__all__ = [
    "Array",
    "open_uniform_knots",
    "n_basis_open_uniform",
    "element_endpoints",
    "local_knot_windows",
    "element_local_indices",
    "h1_seminorm_sq_2d",
    "h1_seminorm_sq_analytic",
]
