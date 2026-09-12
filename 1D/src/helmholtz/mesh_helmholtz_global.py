from __future__ import annotations

"""Global-knot mesh + contrast PDE for the reparametrized `helmholtz` experiment.

Reparametrization: the parameter is the WAVENUMBER CONTRAST c = k1/k2 (vary rho1),
not the frequency. Fixed: omega=1, sigma=(1,4), rho2=100*pi^2 (=> k2=5*pi), g_N=10*pi.
    k1 = c*k2 = 5*pi*c ,   rho1 = (c*k2)^2 * sigma1 = 25*pi^2 * c^2 .
c=3.1 reproduces the original instance (k1=31*pi/2).

GLOBAL mesh (replaces the fixed 50:50 per-half family of mesh_helmholtz.py):
a single softmax over the interior-knot sizes on (0,1); x_I=0.5 is ALWAYS inserted
as a fixed C^0 knot of multiplicity p; ends clamped (mult p+1). The element split
across x_I is FREE -> the network can move elements into the short-wavelength
(high-k) left layer. Ported verbatim from the validated diagnostic
``scripts/diag_p5_global_radapt.py`` (matched src to max|u_loc-u_src|=0 at c=3.1).

h_max safeguard (NEW): a per-layer Nyquist-level cap so k_layer*h_e <= kappa_max
(target points-per-wavelength g_min = 2*pi/kappa_max). Implemented as a single-pass
differentiable water-filling on the softmax sizes; when the cap is infeasible at a
given (c,N) it RELAXES (best achievable) -- documented, not a hard guarantee.

This module is SELF-CONTAINED parameterized copies of the assembly/estimator/exact
(the src versions hardwire rho1 at c=3.1); they are identical to src at c=3.1.
Reuses only constant-free generics (bspline basis, GL quadrature, indefinite solve).
"""

import functools
import math
from typing import Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from common.bspline_basis import bspline_basis_local
from common.quadrature import rule_gl_on_elements
from src.helmholtz.solver_indef import solve_indef
from src.helmholtz.mesh_helmholtz import n_elem_from_knots

Array = jnp.ndarray

# ---- fixed data (contrast reparametrization) ----
X_I = 0.5
SIGMA1, SIGMA2 = 1.0, 4.0
RHO2 = 100.0 * math.pi ** 2
G_N = 10.0 * math.pi
OMEGA = 1.0
K2 = OMEGA * math.sqrt(RHO2 / SIGMA2)          # = 5*pi
G_MIN_TARGET = 2.5                              # points-per-wavelength Nyquist safeguard
KAPPA_MAX = 2.0 * math.pi / G_MIN_TARGET        # k*h <= kappa_max  (~2.513)


def rho1_of_c(c: float) -> float:
    return (float(c) * K2) ** 2 * SIGMA1        # k1 = c*K2, rho1 = (k1/omega)^2 sigma1, omega=1


def k1_of_c(c: float) -> float:
    return float(c) * K2


# ==========================================================================
# Global knot map (free split; x_I forced C^0; clamped ends) + h_max
# ==========================================================================
@functools.partial(jax.jit, static_argnames=("N", "p"))
def _softmax_sizes(theta: Array, N: int, p: int, h_min: float) -> Array:
    """Softmax over N-1 segment sizes on (0,1), sum=1, each >= h_min."""
    theta = jnp.asarray(theta, dtype=DEFAULT_DTYPE).reshape(-1)
    n_seg = theta.shape[0]                       # = N-1
    h = jnp.asarray(h_min, dtype=theta.dtype)
    scale = jnp.asarray(1.0, dtype=theta.dtype) - jnp.asarray(n_seg, dtype=theta.dtype) * h
    m = theta - jnp.max(theta)
    return h + scale * (jnp.exp(m) / jnp.sum(jnp.exp(m)))


def _hmax_water_fill(sizes: Array, N: int, c: float, h_min: float) -> Array:
    """Single-pass differentiable per-layer h_max cap (Nyquist safeguard).

    Each segment's layer is read from its cumulative midpoint (left of x_I -> k1,
    else k2); cap h_max_layer = kappa_max / k_layer. Clip sizes to the cap and
    water-fill the freed slack into the segments with remaining headroom, keeping
    sum=1. If the caps cannot sum to 1 (infeasible at this (c,N)) the cap is
    RELAXED (returns the input sizes unchanged)."""
    dtype = sizes.dtype
    mid = jnp.cumsum(sizes) - 0.5 * sizes                       # segment midpoints (uncapped)
    k1 = jnp.asarray(c, dtype) * jnp.asarray(K2, dtype)         # traced-safe (c may be a tracer)
    k2 = jnp.asarray(K2, dtype)
    k_e = jnp.where(mid < X_I, k1, k2)
    hmax = jnp.asarray(KAPPA_MAX, dtype) / k_e                  # per-segment cap
    feasible = jnp.sum(jnp.maximum(hmax, h_min)) >= 1.0         # caps can tile (0,1)?
    clipped = jnp.minimum(sizes, hmax)
    deficit = 1.0 - jnp.sum(clipped)
    headroom = jnp.maximum(hmax - clipped, 0.0)
    hsum = jnp.sum(headroom)
    filled = clipped + jnp.where(hsum > 0, deficit * headroom / jnp.maximum(hsum, 1e-300), 0.0)
    return jnp.where(feasible, filled, sizes)


@functools.partial(jax.jit, static_argnames=("N", "p", "use_hmax"))
def global_knot_map(theta: Array, N: int, p: int, c: float, h_min: float,
                    use_hmax: bool = True) -> Array:
    """N elements over (0,1); x_I=0.5 forced C^0 (mult p); ends clamped; free split.

    theta has length N-1 (N-1 segments -> N-2 free interior knots; +x_I -> N elems).
    """
    sizes = _softmax_sizes(theta, N, p, h_min)
    if use_hmax:
        sizes = _hmax_water_fill(sizes, N, c, h_min)
    free = jnp.cumsum(sizes)[:-1]                               # N-2 free interior knots
    interior = jnp.sort(jnp.concatenate([free, jnp.array([X_I], dtype=sizes.dtype)]))
    extra = jnp.full((p - 1,), X_I, dtype=sizes.dtype)          # raise 0.5 to multiplicity p (C^0)
    interior = jnp.sort(jnp.concatenate([interior, extra]))
    zeros = jnp.zeros((p + 1,), dtype=sizes.dtype); ones = jnp.ones((p + 1,), dtype=sizes.dtype)
    return jnp.concatenate([zeros, interior, ones])


def uniform_logits(N: int) -> Array:
    """theta = 0 (length N-1): the neutral / uniform global warm-start."""
    return jnp.zeros((int(N) - 1,), dtype=DEFAULT_DTYPE)


def effective_g_min(knots, p: int, c: float) -> float:
    """Realized worst-case points-per-wavelength g = 2*pi / max_e(k_e * h_e)."""
    kn = np.asarray(knots); n = n_elem_from_knots(kn, p)
    a = kn[p:p + n]; b = kn[p + 1:p + n + 1]; he = b - a; mid = 0.5 * (a + b)
    he = he[he > 1e-13]; mid = mid[(b - a) > 1e-13]
    k_e = np.where(mid < X_I, k1_of_c(c), K2)
    kh = np.max(k_e * he)
    return float(2.0 * math.pi / kh) if kh > 0 else float("inf")


def split_counts(knots, p: int) -> Tuple[int, int]:
    kn = np.asarray(knots); n = n_elem_from_knots(kn, p)
    a = kn[p:p + n]; b = kn[p + 1:p + n + 1]; sizes = b - a; mid = 0.5 * (a + b)
    nl = int(np.sum((mid < X_I) & (sizes > 1e-12)))
    nr = int(np.sum((mid > X_I) & (sizes > 1e-12)))
    return nl, nr


# ==========================================================================
# Exact transmission solution (parameterized by c)
# ==========================================================================
def matching_matrix(c: float) -> np.ndarray:
    k1, k2 = k1_of_c(c), K2
    s1k1, s2k2 = SIGMA1 * k1, SIGMA2 * k2
    return np.array([
        [math.sin(k1 * X_I), -math.sin(k2 * X_I), -math.cos(k2 * X_I)],
        [s1k1 * math.cos(k1 * X_I), -s2k2 * math.cos(k2 * X_I), s2k2 * math.sin(k2 * X_I)],
        [0.0, s2k2 * math.cos(k2), -s2k2 * math.sin(k2)],
    ], dtype=np.float64)


class ExactContrast:
    def __init__(self, c: float):
        self.c = float(c); self.k1 = k1_of_c(c); self.k2 = K2
        self.A, self.C, self.D = np.linalg.solve(matching_matrix(c), np.array([0.0, 0.0, G_N]))

    def du(self, x):
        x = np.asarray(x)
        left = self.A * self.k1 * np.cos(self.k1 * x)
        right = self.C * self.k2 * np.cos(self.k2 * x) - self.D * self.k2 * np.sin(self.k2 * x)
        return np.where(x <= X_I, left, right)

    def h1_seminorm(self, ng: int = 64, nsub: int = 32) -> float:
        xg, wg = np.polynomial.legendre.leggauss(ng); val = 0.0
        for a, b in ((0.0, X_I), (X_I, 1.0)):
            e = np.linspace(a, b, nsub + 1)
            for i in range(nsub):
                lo, hi = e[i], e[i + 1]
                xm = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo); wm = 0.5 * (hi - lo) * wg
                d = self.du(xm); val += float(np.sum(wm * d * d))
        return math.sqrt(max(val, 0.0))


def resonance_safe_c(c_target: float, tol: float = 0.02, lo: float = 1.3, hi: float = 6.4):
    """Nudge c off any contrast where the 3x3 matching determinant ~ 0."""
    grid = np.linspace(lo, hi, 8000)
    det = np.array([np.linalg.det(matching_matrix(c)) for c in grid])
    detn = det / np.maximum(np.abs(grid) ** 3, 1e-12)
    res = []
    for i in range(len(grid) - 1):
        if detn[i] * detn[i + 1] < 0:
            t = detn[i] / (detn[i] - detn[i + 1]); res.append(grid[i] + t * (grid[i + 1] - grid[i]))
    res = np.array(res)
    c = float(c_target); nudged = False
    if res.size:
        k = 0
        while np.min(np.abs(c - res)) < tol and k < 60:
            c += 0.01; nudged = True; k += 1
    gap = float(np.min(np.abs(c - res))) if res.size else float("inf")
    return c, nudged, gap, res


def sample_contrasts(c_min: float, c_max: float, n: int, tol: float) -> np.ndarray:
    grid = np.linspace(c_min, c_max, 8000)
    _c, _n, _g, res = resonance_safe_c(0.5 * (c_min + c_max), tol)   # reuse scan
    cand = np.linspace(c_min, c_max, max(int(n) * 20, 400))
    if res.size:
        safe = np.all(np.abs(cand[:, None] - res[None, :]) > float(tol), axis=1)
        cand = cand[safe]
    if cand.size <= int(n):
        return cand
    idx = np.linspace(0, cand.size - 1, int(n)).round().astype(int)
    return np.unique(cand[idx])


# ==========================================================================
# Parameterized assembly / indefinite solve / estimator / error (rho1 = rho1_of_c)
# ==========================================================================
def quad_order(p: int) -> int:
    return int(p) + 2


def _spans(knots, p):
    n = n_elem_from_knots(knots, p)
    return n, knots[p:p + n], knots[p + 1:p + n + 1]


@functools.partial(jax.jit, static_argnames=("p", "nq"))
def solve_u(knots: Array, p: int, rho1: Array, nq: int) -> Array:
    dtype = knots.dtype; om = jnp.asarray(OMEGA, dtype)
    n, a, b = _spans(knots, p); n_ctrl = knots.shape[0] - p - 1
    xq, wq = rule_gl_on_elements(a, b, int(nq))
    Nf, dNf, _d2, _sp = bspline_basis_local(xq.reshape(-1), knots, p)
    Nb = Nf.reshape(n, int(nq), p + 1); dNb = dNf.reshape(n, int(nq), p + 1)
    xm = 0.5 * (a + b)
    sig = jnp.where(xm < X_I, jnp.asarray(SIGMA1, dtype), jnp.asarray(SIGMA2, dtype))
    rho = jnp.where(xm < X_I, jnp.asarray(rho1, dtype), jnp.asarray(RHO2, dtype))
    k_loc = jnp.einsum("e,eqi,eq,eqj->eij", sig, dNb, wq, dNb)
    m_loc = jnp.einsum("e,eqi,eq,eqj->eij", rho, Nb, wq, Nb)
    b_loc = k_loc - (om * om) * m_loc
    cols = jnp.arange(n, dtype=jnp.int32)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    B = jnp.zeros((n_ctrl, n_ctrl), dtype).at[cols[:, :, None], cols[:, None, :]].add(b_loc)
    l = jnp.zeros((n_ctrl,), dtype).at[-1].add(jnp.asarray(G_N, dtype))
    cf = solve_indef(B[1:, 1:], l[1:])
    return jnp.concatenate([jnp.zeros((1,), dtype), cf])


@functools.partial(jax.jit, static_argnames=("p",))
def uh_deriv(knots: Array, p: int, u_full: Array, x: Array) -> Array:
    Nf, dNf, _d2, spans = bspline_basis_local(jnp.asarray(x), knots, p)
    idx = (spans - p)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    return jnp.sum(dNf * u_full[idx], axis=1)


@functools.partial(jax.jit, static_argnames=("p", "nq"))
def eta2(knots: Array, p: int, u_full: Array, rho1: Array, nq: int) -> Array:
    dtype = knots.dtype; om = jnp.asarray(OMEGA, dtype)
    n, a, b = _spans(knots, p); he = b - a
    xq, wq = rule_gl_on_elements(a, b, int(nq))
    Nf, _dN, d2f, _sp = bspline_basis_local(xq.reshape(-1), knots, p)
    Nb = Nf.reshape(n, int(nq), p + 1); d2b = d2f.reshape(n, int(nq), p + 1)
    cols = jnp.arange(n, dtype=jnp.int32)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    uloc = u_full[cols]
    uh = jnp.einsum("eqi,ei->eq", Nb, uloc); uxx = jnp.einsum("eqi,ei->eq", d2b, uloc)
    xm = 0.5 * (a + b)
    sig = jnp.where(xm < X_I, jnp.asarray(SIGMA1, dtype), jnp.asarray(SIGMA2, dtype))
    rho = jnp.where(xm < X_I, jnp.asarray(rho1, dtype), jnp.asarray(RHO2, dtype))
    r = sig[:, None] * uxx + (om * om) * rho[:, None] * uh
    e_elem = jnp.sum((he * he) * jnp.sum(wq * r * r, axis=1))
    # Interface face weight h_I (audit R1 fix): the face length is the smaller of
    # the two INTERFACE-ADJACENT element sizes, not the domain-boundary elements
    # (the previous ``min(he[0], he[-1])`` referenced x=0 and x=1). The C^0 knot at
    # x_I has multiplicity p, so p-1 zero-length elements sit exactly at x_I; mask
    # them out (he > tiny) and pick the real left/right neighbours of x_I:
    #   left  neighbour  = real element with the largest right endpoint b (= x_I),
    #   right neighbour  = real element with the smallest left endpoint a (= x_I).
    tiny = jnp.asarray(1e-13, dtype)
    real = he > tiny
    is_left = jnp.logical_and(xm < X_I, real)
    is_right = jnp.logical_and(xm > X_I, real)
    h_left = he[jnp.argmax(jnp.where(is_left, b, jnp.asarray(-jnp.inf, dtype)))]
    h_right = he[jnp.argmin(jnp.where(is_right, a, jnp.asarray(jnp.inf, dtype)))]
    h_I = jnp.minimum(h_left, h_right)
    # One-sided flux limits at x_I (audit R2 hardening): keep the tiny 1e-9 probe
    # offset, but never let it overshoot the interface-adjacent element, so the
    # evaluation stays in the correct span even if that element is smaller than
    # 1e-9. For every realistic mesh (h_left,h_right >~ 1e-6) the offset is exactly
    # 1e-9, so the value and gradient are unchanged from before.
    off_l = jnp.minimum(jnp.asarray(1e-9, dtype), 0.5 * h_left)
    off_r = jnp.minimum(jnp.asarray(1e-9, dtype), 0.5 * h_right)
    duL = uh_deriv(knots, p, u_full, jnp.reshape(X_I - off_l, (1,)))[0]
    duR = uh_deriv(knots, p, u_full, jnp.reshape(X_I + off_r, (1,)))[0]
    jump = jnp.asarray(SIGMA2, dtype) * duR - jnp.asarray(SIGMA1, dtype) * duL
    e_jump = h_I * jump * jump
    du1 = uh_deriv(knots, p, u_full, jnp.asarray([1.0 - 1e-9], dtype))[0]
    fe = jnp.asarray(G_N, dtype) - jnp.asarray(SIGMA2, dtype) * du1
    return e_elem + e_jump + he[-1] * fe * fe


def h1_rel(knots, p, u_full, exact, nq: int = 24) -> float:
    kn = np.asarray(knots); n = int(kn.shape[0] - 2 * p - 1)
    a = kn[p:p + n]; b = kn[p + 1:p + n + 1]
    xg, wg = np.polynomial.legendre.leggauss(nq); e2 = 0.0
    for i in range(n):
        lo, hi = float(a[i]), float(b[i])
        if hi - lo <= 0:
            continue
        xm = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo); wm = 0.5 * (hi - lo) * wg
        d = np.asarray(uh_deriv(jnp.asarray(kn), p, u_full, jnp.asarray(xm))) - exact.du(xm)
        e2 += float(np.sum(wm * d * d))
    ref = exact.h1_seminorm()
    return math.sqrt(max(e2, 0.0)) / ref if ref > 0 else float("nan")


def eff_index(knots, p, u_full, rho1, exact, nq_eta: int) -> float:
    e2 = float(eta2(jnp.asarray(knots), p, jnp.asarray(u_full), jnp.asarray(rho1), nq_eta))
    # absolute H1 error
    kn = np.asarray(knots); n = int(kn.shape[0] - 2 * p - 1)
    a = kn[p:p + n]; b = kn[p + 1:p + n + 1]
    xg, wg = np.polynomial.legendre.leggauss(24); ae2 = 0.0
    for i in range(n):
        lo, hi = float(a[i]), float(b[i])
        if hi - lo <= 0:
            continue
        xm = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo); wm = 0.5 * (hi - lo) * wg
        d = np.asarray(uh_deriv(jnp.asarray(kn), p, u_full, jnp.asarray(xm))) - exact.du(xm)
        ae2 += float(np.sum(wm * d * d))
    ae = math.sqrt(max(ae2, 0.0))
    return (math.sqrt(max(e2, 0.0)) / ae) if ae > 0 else float("nan")


__all__ = [
    "X_I", "SIGMA1", "SIGMA2", "RHO2", "G_N", "OMEGA", "K2", "KAPPA_MAX", "G_MIN_TARGET",
    "rho1_of_c", "k1_of_c", "global_knot_map", "uniform_logits", "effective_g_min",
    "split_counts", "ExactContrast", "matching_matrix", "resonance_safe_c", "sample_contrasts",
    "quad_order", "solve_u", "uh_deriv", "eta2", "h1_rel", "eff_index",
]
