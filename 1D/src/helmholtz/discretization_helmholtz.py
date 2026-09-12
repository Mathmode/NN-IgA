from __future__ import annotations

"""IGA assembly + indefinite solve for the helmholtz experiment (1D Helmholtz transmission).

Operator (symmetric, INDEFINITE):
    B(omega) = K(sigma) - omega^2 M(rho),
    K_ij = int sigma(x) N_i' N_j' dx,     M_ij = int rho(x) N_i N_j dx,
with sigma, rho piecewise-constant about the interface x_I = 1/2. Because x_I is a
knot (element boundary) every element lies in ONE region, so per-element sigma/rho
are constants and all element integrals are polynomial (exact Gauss).

Load: source-free + Neumann sigma(1) u'(1) = g_N. The only basis nonzero at x=1 is
the last one (value 1), so l = g_N * e_last. Dirichlet u(0)=0 drops DOF 0.

The C^0 interface (knot multiplicity p at x_I) follows the 2D-lshape convention: the
standard span count is n_elem_eff = N + (p-1), i.e. N real cells plus (p-1)
zero-length spans at x_I that carry zero quadrature weight and contribute nothing
(the contiguous local->global map e..e+p stays valid).

Direct LU solve via ``solver_indef`` (NEVER CG: indefinite, resonance-near-singular).
"""

import functools

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from common.bspline_basis import bspline_basis_local
from common.quadrature import rule_gl_on_elements
from src.helmholtz.mesh_helmholtz import X_I, n_elem_from_knots
from src.helmholtz.pde_helmholtz import SIGMA1, SIGMA2, RHO1, RHO2, G_N
from src.helmholtz.solver_indef import solve_indef

Array = jnp.ndarray


def quad_order(degree: int) -> int:
    """GL points/element: exact for the mass integrand N_i N_j (degree 2p)."""
    return int(degree) + 2


def _spans(knots: Array, degree: int):
    p = int(degree)
    n_eff = n_elem_from_knots(knots, p)
    a = knots[p : p + n_eff]
    b = knots[p + 1 : p + n_eff + 1]
    return n_eff, a, b


@functools.partial(jax.jit, static_argnames=("degree", "nq"))
def assemble_B_l(knots: Array, degree: int, omega: Array, nq: int) -> tuple[Array, Array]:
    """Assemble the reduced (Dirichlet-eliminated) system (B_ff, l_f)."""
    p = int(degree)
    dtype = knots.dtype
    omega_t = jnp.asarray(omega, dtype=dtype)
    n_eff, a, b = _spans(knots, p)
    n_ctrl = knots.shape[0] - p - 1

    xq, wq = rule_gl_on_elements(a, b, int(nq))            # (n_eff, nq)
    N_flat, dN_flat, _d2, _spans_ = bspline_basis_local(xq.reshape(-1), knots, p)
    Nb = N_flat.reshape(n_eff, int(nq), p + 1)
    dNb = dN_flat.reshape(n_eff, int(nq), p + 1)

    # Per-span material constants (interface spans are zero-weight => value moot).
    xm = 0.5 * (a + b)
    sig = jnp.where(xm < jnp.asarray(X_I, dtype), jnp.asarray(SIGMA1, dtype), jnp.asarray(SIGMA2, dtype))
    rho = jnp.where(xm < jnp.asarray(X_I, dtype), jnp.asarray(RHO1, dtype), jnp.asarray(RHO2, dtype))

    k_loc = jnp.einsum("eq,eqi,eq,eqj->eij", sig[:, None] * jnp.ones_like(wq), dNb, wq, dNb)
    m_loc = jnp.einsum("eq,eqi,eq,eqj->eij", rho[:, None] * jnp.ones_like(wq), Nb, wq, Nb)
    b_loc = k_loc - (omega_t * omega_t) * m_loc

    cols = jnp.arange(n_eff, dtype=jnp.int32)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    B = jnp.zeros((n_ctrl, n_ctrl), dtype=dtype)
    B = B.at[cols[:, :, None], cols[:, None, :]].add(b_loc)

    l = jnp.zeros((n_ctrl,), dtype=dtype)
    l = l.at[-1].add(jnp.asarray(G_N, dtype=dtype))       # Neumann at x=1 (last basis = 1 there)

    return B[1:, 1:], l[1:]                                # drop DOF 0 (u(0)=0)


@functools.partial(jax.jit, static_argnames=("degree", "nq"))
def solve_state_helmholtz(knots: Array, degree: int, omega: Array, nq: int) -> Array:
    """Return the full coefficient vector u (with the Dirichlet 0 prepended)."""
    B_ff, l_f = assemble_B_l(knots, degree, omega, nq)
    c = solve_indef(B_ff, l_f)
    return jnp.concatenate([jnp.zeros((1,), dtype=c.dtype), c], axis=0)


@functools.partial(jax.jit, static_argnames=("degree",))
def uh_deriv_at(knots: Array, degree: int, u_full: Array, x: Array) -> Array:
    """Evaluate u_h'(x) at arbitrary points (for the H1 metric)."""
    p = int(degree)
    n_eff = n_elem_from_knots(knots, p)
    N_flat, dN_flat, _d2, spans = bspline_basis_local(jnp.asarray(x), knots, p)
    # local active dofs for each x: span-p .. span  (contiguous open-knot map)
    base = spans - p
    idx = base[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    u_loc = u_full[idx]
    return jnp.sum(dN_flat * u_loc, axis=1)


def h1_seminorm_error(knots, degree, u_full, exact, *, nq: int = 24):
    """Host-side relative H1-seminorm error |u_h - u*|_{H1} / |u*|_{H1}.

    Integrates element-by-element on the candidate mesh (each element lies in one
    region, so u*' is a single trig function there) with an nq-point GL rule.
    Returns (abs_err, rel_err, ref_seminorm).
    """
    p = int(degree)
    kn = np.asarray(knots, dtype=np.float64)
    n_eff = int(kn.shape[0] - 2 * p - 1)
    a = kn[p : p + n_eff]
    b = kn[p + 1 : p + n_eff + 1]
    xg, wg = np.polynomial.legendre.leggauss(int(nq))
    err2 = 0.0
    for e in range(n_eff):
        lo, hi = float(a[e]), float(b[e])
        if hi - lo <= 0.0:
            continue
        xm = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        wm = 0.5 * (hi - lo) * wg
        duh = np.asarray(uh_deriv_at(jnp.asarray(kn), p, u_full, jnp.asarray(xm)))
        due = exact.du(xm)
        d = duh - due
        err2 += float(np.sum(wm * d * d))
    abs_err = np.sqrt(max(err2, 0.0))
    ref = exact.h1_seminorm()
    return float(abs_err), float(abs_err / ref if ref > 0 else np.nan), float(ref)


__all__ = [
    "quad_order", "assemble_B_l", "solve_state_helmholtz",
    "uh_deriv_at", "h1_seminorm_error",
]
