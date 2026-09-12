from __future__ import annotations

"""Shared JAX B-spline basis routines for the repo (1D/2D).

This module provides:
  - A generic (degree p) local active basis evaluator on a *global* knot vector
    (NURBS Book style algorithms, with safe divisions).
  - Specialized, fully-vectorized batch kernels for p=2 and p=3 on *local*
    knot windows (used by the Helmholtz interface experiment).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


def safe_div(num: Array, den: Array, *, eps: float = 1e-30) -> Array:
    """Gradient-safe division: returns ``num / den`` where ``|den| >= eps``,
    else 0.

    Uses the *double-where* pattern so that ``jax.grad`` does not produce
    NaN on the discarded branch (a known JAX/autograd-of-where pitfall —
    see the JAX FAQ entry on "Gradients contain NaN where using where").

    For ``|den| >= eps`` the value and gradient are identical to
    ``num / den``; for ``|den| < eps`` both the value and the gradient
    contribution are zero. This makes the function safe under repeated
    knots in B-spline assemblies where some Cox-de Boor denominators are
    structurally zero.
    """
    den = jnp.asarray(den)
    num = jnp.asarray(num, dtype=den.dtype)
    eps_t = jnp.asarray(eps, dtype=den.dtype)
    safe = jnp.abs(den) >= eps_t
    # Replace ``den`` with 1 on the unsafe branch BEFORE dividing, so the
    # forward pass never produces inf/NaN that would corrupt the
    # gradient through the unselected branch of ``jnp.where``.
    safe_den = jnp.where(safe, den, jnp.ones_like(den))
    return jnp.where(safe, num / safe_den, jnp.zeros_like(num))


def _knot_at(knots: Array, idx: Array) -> Array:
    return lax.dynamic_index_in_dim(knots, idx, axis=0, keepdims=False)


def _ders_basis_funs_single(span: Array, x: Array, knots: Array, p: int, n_ders: int) -> Array:
    """Derivatives of B-spline basis functions at one point.

    Returns ders with shape (n_ders+1, p+1) where ders[k, i] is k-th derivative
    of i-th active function.

    Notes: this follows the standard algorithm but uses safe divisions so
    repeated knots contribute zero in those branches.
    """
    p = int(p)
    n_ders = int(min(max(int(n_ders), 0), p))

    dtype = knots.dtype
    x = jnp.asarray(x, dtype=dtype)
    span = jnp.asarray(span, dtype=jnp.int32)

    ndu = jnp.zeros((p + 1, p + 1), dtype=dtype)
    left = jnp.zeros((p + 1,), dtype=dtype)
    right = jnp.zeros((p + 1,), dtype=dtype)
    ndu = ndu.at[0, 0].set(1.0)

    for j in range(1, p + 1):
        left = left.at[j].set(x - _knot_at(knots, span + (1 - j)))
        right = right.at[j].set(_knot_at(knots, span + j) - x)
        saved = jnp.asarray(0.0, dtype=dtype)
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            ndu = ndu.at[j, r].set(denom)
            temp = safe_div(ndu[r, j - 1], denom)
            ndu = ndu.at[r, j].set(saved + right[r + 1] * temp)
            saved = left[j - r] * temp
        ndu = ndu.at[j, j].set(saved)

    ders = jnp.zeros((n_ders + 1, p + 1), dtype=dtype)
    ders = ders.at[0, :].set(ndu[:, p])

    a = jnp.zeros((2, p + 1), dtype=dtype)

    for r in range(p + 1):
        a = a.at[0, 0].set(1.0)
        s1, s2 = 0, 1
        for k in range(1, n_ders + 1):
            a = a.at[s2, :].set(0.0)
            d = jnp.asarray(0.0, dtype=dtype)
            rk = r - k
            pk = p - k
            if r >= k:
                denom = ndu[pk + 1, rk]
                val = safe_div(a[s1, 0], denom)
                a = a.at[s2, 0].set(val)
                d = d + val * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = (k - 1) if (r - 1) <= pk else (p - r)
            for j in range(j1, j2 + 1):
                denom = ndu[pk + 1, rk + j]
                val = safe_div(a[s1, j] - a[s1, j - 1], denom)
                a = a.at[s2, j].set(val)
                d = d + val * ndu[rk + j, pk]

            if r <= pk:
                denom = ndu[pk + 1, r]
                val = -safe_div(a[s1, k - 1], denom)
                a = a.at[s2, k].set(val)
                d = d + val * ndu[r, pk]

            ders = ders.at[k, r].set(d)

            # Swap buffer roles; do not swap row contents.
            s1, s2 = s2, s1

    # Multiply by factorial factors
    for k in range(1, n_ders + 1):
        factor = 1.0
        for j in range(p - k + 1, p + 1):
            factor *= j
        ders = ders.at[k, :].set(ders[k, :] * factor)

    return ders


@functools.partial(jax.jit, static_argnames=("p",))
def bspline_basis_local(x: Array, knots: Array, p: int) -> tuple[Array, Array, Array, Array]:
    """Evaluate active B-spline basis functions and derivatives at points ``x``.

    Returns
    -------
    N, dN, d2N, spans
        N[i,:] contains the (p+1) active basis functions at x[i],
        and spans[i] is the global span index such that x[i] in [U[span], U[span+1]).
    """
    x = jnp.asarray(x, dtype=knots.dtype)
    p = int(p)

    last = knots.shape[0] - p - 2  # last interior span

    def _span_single(xi):
        s = jnp.searchsorted(knots, xi, side="right") - 1
        s = jnp.clip(s, p, last)
        s = jnp.where(jnp.isclose(xi, knots[-1], atol=1e-14), last, s)
        return s.astype(jnp.int32)

    spans = jax.vmap(_span_single)(x)

    def _eval_one(xi, sp):
        ders = _ders_basis_funs_single(sp, xi, knots, p, 2)
        N = ders[0]
        dN = ders[1] if p >= 1 else jnp.zeros_like(N)
        d2N = ders[2] if p >= 2 else jnp.zeros_like(N)
        return N, dN, d2N

    N, dN, d2N = jax.vmap(_eval_one)(x, spans)
    return N, dN, d2N, spans


def basis_p2_batch(u: Array, U_local: Array) -> tuple[Array, Array, Array]:
    """Quadratic basis + first/second derivatives in batch.

    Args:
        u:       (n_elem, n_q) or (n_elem,)
        U_local: (n_elem, 6) local knot windows

    Returns:
        N, dN, d2N: each (n_elem, n_q, 3)
    """
    u = jnp.asarray(u)
    if u.ndim == 1:
        u = u[:, None]

    Ua = U_local[:, 1][:, None]
    Us = U_local[:, 2][:, None]
    Ub = U_local[:, 3][:, None]
    Uc = U_local[:, 4][:, None]

    A = Ub - Us
    D0 = Ub - Ua
    D2 = Uc - Us

    t = Ub - u
    s = u - Us

    denom0 = A * D0
    denom2 = A * D2

    N0 = safe_div(t * t, denom0, eps=1e-14)
    N2 = safe_div(s * s, denom2, eps=1e-14)
    N1 = safe_div((u - Ua) * t, denom0, eps=1e-14) + safe_div((Uc - u) * s, denom2, eps=1e-14)

    dN0 = safe_div(-2.0 * t, denom0, eps=1e-14)
    dN2 = safe_div(2.0 * s, denom2, eps=1e-14)
    dN1 = safe_div(Ua + Ub - 2.0 * u, denom0, eps=1e-14) + safe_div(Us + Uc - 2.0 * u, denom2, eps=1e-14)

    one = jnp.ones_like(u)
    d2N0 = safe_div(2.0 * one, denom0, eps=1e-14)
    d2N2 = safe_div(2.0 * one, denom2, eps=1e-14)
    d2N1 = safe_div(-2.0 * one, denom0, eps=1e-14) + safe_div(-2.0 * one, denom2, eps=1e-14)

    N = jnp.stack([N0, N1, N2], axis=-1)
    dN = jnp.stack([dN0, dN1, dN2], axis=-1)
    d2N = jnp.stack([d2N0, d2N1, d2N2], axis=-1)
    return N, dN, d2N


def basis_p3_batch(u: Array, U_local: Array) -> tuple[Array, Array, Array]:
    """Cubic basis + first/second derivatives in batch.

    Args:
        u:       (n_elem, n_q) or (n_elem,)
        U_local: (n_elem, 8) local knot windows

    Returns:
        N, dN, d2N: each (n_elem, n_q, 4)
    """
    u = jnp.asarray(u)
    if u.ndim == 1:
        u = u[:, None]

    t_m2 = U_local[:, 1][:, None]
    t_m1 = U_local[:, 2][:, None]
    t0 = U_local[:, 3][:, None]
    t1 = U_local[:, 4][:, None]
    t2 = U_local[:, 5][:, None]
    t3 = U_local[:, 6][:, None]

    den01 = t1 - t0
    N1_m1 = safe_div(t1 - u, den01, eps=1e-14)
    N1_0 = safe_div(u - t0, den01, eps=1e-14)

    N2_m2 = safe_div(t1 - u, t1 - t_m1, eps=1e-14) * N1_m1
    N2_m1 = safe_div(u - t_m1, t1 - t_m1, eps=1e-14) * N1_m1 + safe_div(t2 - u, t2 - t0, eps=1e-14) * N1_0
    N2_0 = safe_div(u - t0, t2 - t0, eps=1e-14) * N1_0

    N3_m3 = safe_div(t1 - u, t1 - t_m2, eps=1e-14) * N2_m2
    N3_m2 = safe_div(u - t_m2, t1 - t_m2, eps=1e-14) * N2_m2 + safe_div(t2 - u, t2 - t_m1, eps=1e-14) * N2_m1
    N3_m1 = safe_div(u - t_m1, t2 - t_m1, eps=1e-14) * N2_m1 + safe_div(t3 - u, t3 - t0, eps=1e-14) * N2_0
    N3_0 = safe_div(u - t0, t3 - t0, eps=1e-14) * N2_0

    N = jnp.stack([N3_m3, N3_m2, N3_m1, N3_0], axis=-1)

    dN2_m2 = -2.0 * safe_div(N1_m1, t1 - t_m1, eps=1e-14)
    dN2_m1 = 2.0 * safe_div(N1_m1, t1 - t_m1, eps=1e-14) - 2.0 * safe_div(N1_0, t2 - t0, eps=1e-14)
    dN2_0 = 2.0 * safe_div(N1_0, t2 - t0, eps=1e-14)

    dN3_m3 = -3.0 * safe_div(N2_m2, t1 - t_m2, eps=1e-14)
    dN3_m2 = 3.0 * safe_div(N2_m2, t1 - t_m2, eps=1e-14) - 3.0 * safe_div(N2_m1, t2 - t_m1, eps=1e-14)
    dN3_m1 = 3.0 * safe_div(N2_m1, t2 - t_m1, eps=1e-14) - 3.0 * safe_div(N2_0, t3 - t0, eps=1e-14)
    dN3_0 = 3.0 * safe_div(N2_0, t3 - t0, eps=1e-14)

    dN = jnp.stack([dN3_m3, dN3_m2, dN3_m1, dN3_0], axis=-1)

    d2N3_m3 = -3.0 * safe_div(dN2_m2, t1 - t_m2, eps=1e-14)
    d2N3_m2 = 3.0 * safe_div(dN2_m2, t1 - t_m2, eps=1e-14) - 3.0 * safe_div(dN2_m1, t2 - t_m1, eps=1e-14)
    d2N3_m1 = 3.0 * safe_div(dN2_m1, t2 - t_m1, eps=1e-14) - 3.0 * safe_div(dN2_0, t3 - t0, eps=1e-14)
    d2N3_0 = 3.0 * safe_div(dN2_0, t3 - t0, eps=1e-14)

    d2N = jnp.stack([d2N3_m3, d2N3_m2, d2N3_m1, d2N3_0], axis=-1)
    return N, dN, d2N


def _basis_generic_batch(xq: Array, U_local: Array, p: int) -> tuple[Array, Array, Array]:
    """Generic batch basis evaluation via Cox-de Boor for any degree p >= 1.

    Args:
        xq:      (n_elem, n_q)    quadrature points per element
        U_local: (n_elem, 2p+2)   local knot windows per element
        p:       polynomial degree

    Returns (N, dN, d2N) each of shape (n_elem, n_q, p+1).
    """
    p = int(p)
    xq = jnp.asarray(xq)
    if xq.ndim == 1:
        xq = xq[:, None]

    def _eval_single(xi, knot_window):
        span = jnp.asarray(p, dtype=jnp.int32)
        ders = _ders_basis_funs_single(span, xi, knot_window, p, min(2, p))
        N = ders[0]
        dN = ders[1] if p >= 1 else jnp.zeros_like(N)
        d2N = ders[2] if p >= 2 else jnp.zeros_like(N)
        return N, dN, d2N

    def _eval_element(x_elem, window):
        return jax.vmap(lambda xi: _eval_single(xi, window))(x_elem)

    N, dN, d2N = jax.vmap(_eval_element)(xq, U_local)
    return N, dN, d2N


def basis_batch_for_degree(p: int, xq: Array, U_local: Array) -> tuple[Array, Array, Array]:
    """Dispatch to the specialised p=2/p=3 batch kernel, or generic for p >= 4.

    Returns (N, dN, d2N), each of shape ``(n_elem, n_q, p+1)``.
    """
    if int(p) == 2:
        return basis_p2_batch(xq, U_local)
    if int(p) == 3:
        return basis_p3_batch(xq, U_local)
    if int(p) >= 4:
        return _basis_generic_batch(xq, U_local, int(p))
    raise ValueError(f"basis_batch_for_degree: unsupported p={p}; expected >= 2")


__all__ = [
    "Array",
    "safe_div",
    "bspline_basis_local",
    "basis_p2_batch",
    "basis_p3_batch",
    "basis_batch_for_degree",
]
