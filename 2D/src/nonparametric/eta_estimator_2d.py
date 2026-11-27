"""Residual a-posteriori estimator η² for the 2D experiments arctan and lshape.

Ported from ``src/nonparametric/exp2_cpu/estimator.py`` (top-level reference
implementation, see audit_2d_estimator_implementation.md). Re-written to
consume the ``<repo>/2D`` solver's primitives:

  - knot vectors ``knots_x, knots_y`` (open-uniform, possibly with a fixed
    node at 0.5 for the lshape L-shape),
  - degree ``p``, element counts ``n_elem_x, n_elem_y`` (static),
  - coefficient matrix ``u_coeffs`` of shape ``(n_x, n_y)`` (x-major),
  - parameter tuple (alpha, s1, s2) for arctan or (sigma1, sigma2) for lshape,
  - GL quadrature order ``q_est`` (default 50 for arctan, 2 for lshape).

Formula (both arctan and lshape, for IGA basis with ``C^{p-1}`` interelement
continuity the internal flux jumps vanish):

    η²(θ; σ) = Σ_E h_E² ‖R_E‖²_{L²(E)} (+ Neumann boundary terms for arctan)

where for arctan ``R_E = −Δu_h − f^σ`` (sigma=1, α=0)
and for lshape ``R_E = −σ(x)·Δu_h − 1`` (piecewise σ inside each element).

The Neumann edge contributions for arctan are
``Σ_F h_F · ‖∂_n u_h − g^σ‖²_{L²(F)}`` over the right (x=1) and top (y=1)
edges, with h_F the edge-element width along the boundary.

JIT-compatible and ``jax.vmap``-friendly over batches of σ.
"""
from __future__ import annotations

import functools

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from common.bspline_basis import basis_batch_for_degree
from common.quadrature import rule_gl_on_elements

from src.nonparametric.arctan.pde import (
    f_sigma,
    g_sigma_neumann_right,
    g_sigma_neumann_top,
)
from src.nonparametric.lshape.integration import volume, flux_face
from src.nonparametric.solver_2d import _element_data, galerkin_solve_arctan, galerkin_solve_lshape

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Helper: gather local coefficients
# --------------------------------------------------------------------------


def _gather_local_coeffs(C: Array, idx_x: Array, idx_y: Array) -> Array:
    """Return (n_elem_x, n_elem_y, p+1, p+1).

    Layout convention:
        C: (n_x, n_y)  — x-major flatten, i*n_y + j.
        idx_x: (n_elem_x, p+1)  — global x-indices for the active basis on each element.
        idx_y: (n_elem_y, p+1)  — same for y.
    """
    Cx = C[idx_x[:, :, None, None], idx_y[None, None, :, :]]
    # Currently (n_elem_x, p+1, n_elem_y, p+1) — swap middle dims.
    return Cx.transpose(0, 2, 1, 3)


# --------------------------------------------------------------------------
# Endpoint basis for evaluating ∂_n u_h on the right / top edge.
# --------------------------------------------------------------------------


def _endpoint_basis_right(knots: Array, p: int, n_elem: int):
    """Return (g, dN_at_boundary) for the basis functions active in the
    rightmost element, evaluated at the boundary x = 1.

    Output:
        g   : (p+1,) int32  — global indices of active basis on last element.
        dN  : (p+1,) — derivative ∂x of those functions evaluated at x = 1.
    """
    a, b, U_local, idx_local = _element_data(knots, p, n_elem)
    last = n_elem - 1
    g = idx_local[last]
    U_loc_last = U_local[last:last + 1]               # (1, 2p+2)
    x_eval = jnp.asarray([[1.0]], dtype=DEFAULT_DTYPE)
    N, dN, _ = basis_batch_for_degree(p, x_eval, U_loc_last)
    # N, dN shape (1, 1, p+1)
    return g, N[0, 0], dN[0, 0]


# --------------------------------------------------------------------------
# Volume term arctan (alpha=0, sigma=1)
# --------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_est"))
def _eta_volume_arctan(
    C: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    alpha: Array, s1: Array, s2: Array,
    q_est: int,
) -> Array:
    a_x, b_x, U_local_x, idx_local_x = _element_data(knots_x, p, n_elem_x)
    a_y, b_y, U_local_y, idx_local_y = _element_data(knots_y, p, n_elem_y)

    xq, wq_x = rule_gl_on_elements(a_x, b_x, q_est)
    yq, wq_y = rule_gl_on_elements(a_y, b_y, q_est)
    Nx, _, d2Nx = basis_batch_for_degree(p, xq, U_local_x)
    Ny, _, d2Ny = basis_batch_for_degree(p, yq, U_local_y)

    C_loc = _gather_local_coeffs(C, idx_local_x, idx_local_y)
    # einsum indices:
    #   x = element index along x-axis
    #   y = element index along y-axis
    #   i = local basis along x (p+1)
    #   j = local basis along y (p+1)
    #   q = quad index along x (q_est)
    #   r = quad index along y (q_est)
    uh = jnp.einsum("xyij,xqi,yrj->xyqr", C_loc, Nx, Ny)
    uhxx = jnp.einsum("xyij,xqi,yrj->xyqr", C_loc, d2Nx, Ny)
    uhyy = jnp.einsum("xyij,xqi,yrj->xyqr", C_loc, Nx, d2Ny)

    XX = jnp.broadcast_to(xq[:, None, :, None], uh.shape)
    YY = jnp.broadcast_to(yq[None, :, None, :], uh.shape)
    f_vals = f_sigma(XX, YY, alpha, s1, s2)
    R = -uhxx - uhyy - f_vals

    w_2d = wq_x[:, None, :, None] * wq_y[None, :, None, :]
    R2 = jnp.sum(R * R * w_2d, axis=(2, 3))   # (n_elem_x, n_elem_y)
    hx = b_x - a_x
    hy = b_y - a_y
    hE_sq = hx[:, None] ** 2 + hy[None, :] ** 2
    return jnp.maximum(jnp.sum(hE_sq * R2), 0.0)


# --------------------------------------------------------------------------
# Neumann term arctan (right + top edges)
# --------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_est"))
def _eta_neumann_arctan(
    C: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    alpha: Array, s1: Array, s2: Array,
    q_est: int,
) -> Array:
    a_x, b_x, U_local_x, idx_local_x = _element_data(knots_x, p, n_elem_x)
    a_y, b_y, U_local_y, idx_local_y = _element_data(knots_y, p, n_elem_y)
    xq, wq_x = rule_gl_on_elements(a_x, b_x, q_est)
    yq, wq_y = rule_gl_on_elements(a_y, b_y, q_est)
    Nx, _, _ = basis_batch_for_degree(p, xq, U_local_x)
    Ny, _, _ = basis_batch_for_degree(p, yq, U_local_y)

    # ---- x = 1 edge (right) ----
    gx_r, _, dNx_r = _endpoint_basis_right(knots_x, p, n_elem_x)
    # ∂x u_h(1, y) = sum_i sum_j c_{ij} N'_i(1) N_j(y) = sum_j coeff_x[j] N_j(y)
    coeff_x = dNx_r @ C[gx_r, :]                                  # (n_y,)
    coeff_local_y = coeff_x[idx_local_y]                           # (n_elem_y, p+1)
    uhx_right = jnp.einsum("eqi,ei->eq", Ny, coeff_local_y)        # (n_elem_y, q_est)
    rx = uhx_right - g_sigma_neumann_right(yq, alpha, s1, s2)
    hy = b_y - a_y
    eta_x_sq = jnp.sum(hy * jnp.sum(wq_y * rx * rx, axis=1))

    # ---- y = 1 edge (top) ----
    gy_t, _, dNy_t = _endpoint_basis_right(knots_y, p, n_elem_y)
    coeff_y = C[:, gy_t] @ dNy_t                                   # (n_x,)
    coeff_local_x = coeff_y[idx_local_x]                           # (n_elem_x, p+1)
    uhy_top = jnp.einsum("eqi,ei->eq", Nx, coeff_local_x)          # (n_elem_x, q_est)
    ry = uhy_top - g_sigma_neumann_top(xq, alpha, s1, s2)
    hx = b_x - a_x
    eta_y_sq = jnp.sum(hx * jnp.sum(wq_x * ry * ry, axis=1))

    return jnp.maximum(eta_x_sq + eta_y_sq, 0.0)


# --------------------------------------------------------------------------
# Public API: η²(θ; σ) for arctan (arctangent 2D, Aballay 4.3.1)
# --------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_est"))
def eta_squared_arctan(
    u_coeffs: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    alpha: Array,
    s1: Array,
    s2: Array,
    *,
    q_est: int = 50,
) -> Array:
    """Residual a-posteriori estimator η² for arctan (Aballay 4.3.1).

    Returns a non-negative scalar. JIT-compatible. Differentiable through
    ``alpha, s1, s2`` and the knot vectors. ``vmap``-friendly when
    broadcasting σ-tuples.
    """
    return (
        _eta_volume_arctan(u_coeffs, knots_x, knots_y, p, n_elem_x, n_elem_y, alpha, s1, s2, q_est)
        + _eta_neumann_arctan(u_coeffs, knots_x, knots_y, p, n_elem_x, n_elem_y, alpha, s1, s2, q_est)
    )


# --------------------------------------------------------------------------
# Volume term lshape (piecewise σ, Dirichlet everywhere)
# --------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_est"))
def _eta_volume_lshape(
    C: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    sigma1: Array,
    sigma2: Array,
    q_est: int,
) -> Array:
    return volume(C, knots_x, knots_y, p, sigma1, sigma2, q_est)


# --------------------------------------------------------------------------
# Flux-jump term lshape (across the two MATERIAL interfaces x=0.5 and y=0.5)
# --------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_est"))
def _eta_jump_lshape(
    C: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    sigma1: Array,
    sigma2: Array,
    q_est: int,
) -> Array:
    """Residual estimator's flux-jump contribution on the two material
    interfaces of the L-shape.

    For an interface ``F`` separating element ``E⁻`` (coefficient ``σ⁻``)
    from element ``E⁺`` (coefficient ``σ⁺``), with unit normal ``n``
    pointing from ``−`` to ``+``, the discrete flux jump of the FE
    solution is

        J_F(s) = σ⁺ · (∂_n u_h)|_{E⁺}(s)  −  σ⁻ · (∂_n u_h)|_{E⁻}(s).

    The two normal derivatives are computed from the basis functions of
    the ``+`` and ``−`` elements *independently*, evaluated at the same
    physical points ``s`` on the interface. They are NOT equal: the lshape
    knot vector inserts the 0.5 node with multiplicity ``p``, making the
    basis only ``C⁰`` at the interface (i.e. ``∂_n u_h`` jumps there).

    Earlier versions of this function read ``∂_n u_h`` from a single
    side and scaled by ``(σ⁺ − σ⁻)``, justified by a (false) claim of
    ``C^{p-1}`` continuity at the interface. That collapsed the diagonal
    symmetry of η² under ``σ1 ↔ σ2`` (the two interfaces were no longer
    diagonal reflections of each other) and produced ratios of 10×-94×
    on a uniform mesh. See audit_p3_jump_term_fix.md for the trace.

    The two interfaces:

        (a) ``x = 0.5``, ``y ∈ [0.5, 1]``: separates top-left (σ⁻ = 1,
            ``x < 0.5`` side) from top-right (σ⁺ = σ2, ``x > 0.5`` side).
            Normal in +x.  Need ∂_x u_h from the LEFT element (last real
            element of the left half, index half_x - 1) AND from the
            RIGHT element (first real right-half element, index right_x).

        (b) ``y = 0.5``, ``x ∈ [0, 0.5]``: separates bot-left (σ⁻ = σ1,
            ``y < 0.5`` side) from top-left (σ⁺ = 1, ``y > 0.5`` side).
            Normal in +y.  Need ∂_y u_h from the BELOW element (index
            half_y - 1) AND from the ABOVE element (index right_y).

    Each contributes  ``(h_F / σ_F) · ‖J_F‖²_{L²(F)}``  with
    ``σ_F = max(σ⁺, σ⁻)`` (Bernardi-Verfürth / Petzoldt contrast-robust
    scaling; unchanged from the previous version).

    The implementation is written so that the code for interface (a) and
    the code for interface (b) are *structural mirrors* of each other
    under x↔y swap and (sigma1, sigma2) ↔ (sigma2, sigma1). That code-
    level symmetry is what enforces the η² symmetry numerically.

    The other two segments touching x = 0.5 / y = 0.5 — namely
    ``{x=0.5} × [0, 0.5]`` and ``{y=0.5} × [0.5, 1]`` — are pieces of
    the L-shape's external Dirichlet boundary, where u = 0; no flux-jump
    contribution there.
    """
    return (flux_face(C, knots_x, knots_y, p, 1.0, sigma2, True, q_est)
            + flux_face(C.T, knots_y, knots_x, p, sigma1, 1.0, False, q_est))


# --------------------------------------------------------------------------
# Public API: η²(θ; σ) for lshape (L-shape, Aballay 4.3.2)
# --------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_est"))
def eta_squared_lshape(
    u_coeffs: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    sigma1: Array,
    sigma2: Array,
    *,
    q_est: int = 4,
) -> Array:
    """Residual a-posteriori estimator η² for lshape (Aballay 4.3.2).

    σ-weighted formula (Bernardi-Verfürth scaling for piecewise diffusion):

        η²(θ; σ) =  Σ_E (h_E² / σ_E) ‖−∇·(σ ∇u_h) − f‖²_{L²(E)}
                  + Σ_F (h_F  / σ_F) ‖[σ ∂_n u_h]‖²_{L²(F)}

    where
        σ_E = value of σ in element E (constant since element edges align
              with the material interfaces),
        σ_F = max(σ_+, σ_−) across the interface F (contrast-robust).

    The second sum runs over the two L-shape material interfaces:
    ``{x = 0.5} × [0.5, 1]`` (top-left / top-right, σ jumps 1 ↔ σ2) and
    ``{y = 0.5} × [0, 0.5]``  (bot-left / top-left, σ jumps σ1 ↔ 1).
    Internal element interfaces away from these two segments have zero
    flux jump because the IGA basis is ``C^{p-1}`` continuous and σ is
    constant within each L-shape sub-region.

    Elements whose centroid lies in the closed removed quadrant
    ``{x ≥ 0.5} ∩ {y ≤ 0.5}`` are excluded from the volume sum (their
    DOFs are zeroed by the solver's Dirichlet mask).

    For σ = (1, 1) the weights are trivial (σ_E = σ_F = 1) and the
    estimator reduces to the unweighted form used in arctan.
    """
    return (
        _eta_volume_lshape(
            u_coeffs, knots_x, knots_y, p, n_elem_x, n_elem_y, sigma1, sigma2, q_est,
        )
        + _eta_jump_lshape(
            u_coeffs, knots_x, knots_y, p, n_elem_x, n_elem_y, sigma1, sigma2, q_est,
        )
    )


# --------------------------------------------------------------------------
# Convenience: η² on the UNIFORM mesh (used as σ-balancing denominator).
# --------------------------------------------------------------------------


def eta_uniform_squared_arctan(
    sigma: Array,
    N: int,
    p: int,
    *,
    q_est: int = 50,
    q_K: int = None,
    q_F: int = 50,
) -> Array:
    """Compute η²(θ_uniform; σ) for arctan — used as denominator of the residual loss.

    ``sigma`` is the tuple ``(alpha, s1, s2)``. Mesh = uniform ``N×N`` on the
    unit square (open-uniform B-spline).
    """
    from common.h1_seminorm_2d import open_uniform_knots

    if q_K is None:
        q_K = p + 1
    knots = open_uniform_knots(int(N), int(p))
    knots_j = jnp.asarray(knots, dtype=DEFAULT_DTYPE)
    res = galerkin_solve_arctan(
        knots_j, knots_j, int(p), int(N), int(N),
        sigma[0], sigma[1], sigma[2],
        q_K=int(q_K), q_F=int(q_F),
    )
    return eta_squared_arctan(
        res.u_h, knots_j, knots_j, int(p), int(N), int(N),
        sigma[0], sigma[1], sigma[2], q_est=int(q_est),
    )


def eta_uniform_squared_lshape(
    sigma: Array,
    N: int,
    p: int,
    *,
    q_est: int = 4,
    q_K: int = None,
    q_F: int = 2,
) -> Array:
    """Compute η²(θ_uniform; σ) for lshape (uniform mesh with multiplicity-p
    interface knot at 0.5)."""
    from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape

    if q_K is None:
        q_K = p + 1
    half = int(N) // 2
    if half * 2 != int(N):
        raise ValueError(f"lshape needs even N; got {N}")
    knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), int(p))
    n_eff = p3_effective_n_elem(int(N), int(p))
    res = galerkin_solve_lshape(
        knots, knots, int(p), n_eff, n_eff,
        sigma[0], sigma[1],
        q_K=int(q_K), q_F=int(q_F),
    )
    return eta_squared_lshape(
        res.u_h, knots, knots, int(p), n_eff, n_eff,
        sigma[0], sigma[1], q_est=int(q_est),
    )


__all__ = [
    "eta_squared_arctan",
    "eta_squared_lshape",
    "eta_uniform_squared_arctan",
    "eta_uniform_squared_lshape",
]
