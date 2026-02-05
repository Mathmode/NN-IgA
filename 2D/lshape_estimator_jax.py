from __future__ import annotations

"""Residual estimator for the multipatch L-shape Laplace problem (JAX).

PDE:
    -Δu = 0

Strong residual in each patch:
    R = u_xx + u_yy

Estimator:
    η^2 = Σ_K h_K^2 ||R||^2_{L2(K)}
        + Σ_{E ∈ interfaces} h_E ||[∂_n u_h]||^2_{L2(E)}.

Implementation:
  - Element-local tensor-product evaluation (no global dense eval matrices)
  - Jump terms evaluated on the two interfaces using 1D Gauss rules
  - JIT-friendly for fixed (N,p,q_order)
"""

import functools

import jax
import jax.numpy as jnp

from bspline_basis import basis_p2_batch, basis_p3_batch
from iga2d_jax import eval_1d_on_elements

Array = jnp.ndarray


def _extract_patch_coeffs(u_global: Array, gmap: Array) -> Array:
    """Return per-patch coefficient grid C(iy,ix) from global coeff vector."""
    nBy, nBx = int(gmap.shape[0]), int(gmap.shape[1])
    return jnp.asarray(u_global)[jnp.asarray(gmap).reshape(-1)].reshape(nBy, nBx)


def _dN_at_point_in_element(u: Array, knots: Array, *, p: int, e: int) -> Array:
    """Evaluate dN (size p+1) at one point u inside element with start index e."""
    p = int(p)
    e = int(e)
    u = jnp.asarray(u).reshape(1, 1)
    idx = jnp.asarray(e, dtype=jnp.int32) + jnp.arange(2 * p + 2, dtype=jnp.int32)
    Uloc = jnp.asarray(knots)[idx][None, :]

    if p == 2:
        _, dN, _ = basis_p2_batch(u, Uloc)
    elif p == 3:
        _, dN, _ = basis_p3_batch(u, Uloc)
    else:
        raise ValueError("L-shape JAX estimator supports p in {2,3}")
    return dN[0, 0, :]


@functools.partial(jax.jit, static_argnames=("p", "q_order", "include_jump"))
def estimator_eta2(
    u_global: Array,
    *,
    # Mesh + topology
    bp_xL: Array,
    bp_xR: Array,
    bp_yB: Array,
    bp_yT: Array,
    knots_xL: Array,
    knots_xR: Array,
    knots_yB: Array,
    knots_yT: Array,
    g0: Array,
    g1: Array,
    g2: Array,
    p: int,
    # Quadrature / options
    q_order: int = 12,
    include_jump: bool = True,
    eps_side: float = 1e-6,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Return (eta2_total, (eta2_vol, eta2_jump, eta2_jump01, eta2_jump12))."""
    p = int(p)
    q_order = int(q_order)
    dtype = jnp.asarray(u_global).dtype

    # ------------------------------------------------------------------
    # Volume term (sum over patches)
    # ------------------------------------------------------------------
    eta2_vol = jnp.asarray(0.0, dtype=dtype)

    def _vol_for_patch(C: Array, bp_x: Array, bp_y: Array) -> Array:
        _, _, wx, hx, Gx, Nx, _, D2x = eval_1d_on_elements(bp_x, p, q_order=q_order, n_der=2)
        _, _, wy, hy, Gy, Ny, _, D2y = eval_1d_on_elements(bp_y, p, q_order=q_order, n_der=2)

        C_blocks = C[Gy[:, :, None, None], Gx[None, None, :, :]]  # (Ny,p+1,Nx,p+1)
        C_blocks = jnp.transpose(C_blocks, (0, 2, 1, 3))  # (Ny,Nx,p+1,p+1)

        uxx = jnp.einsum("yai,yxij,xbj->yxab", Ny, C_blocks, D2x)
        uyy = jnp.einsum("yai,yxij,xbj->yxab", D2y, C_blocks, Nx)
        R = uxx + uyy

        w2d = wy[:, None, :, None] * wx[None, :, None, :]
        hK = jnp.maximum(hy[:, None], hx[None, :])
        return jnp.sum(w2d * (hK[:, :, None, None] ** 2) * (R * R))

    C0 = _extract_patch_coeffs(u_global, g0)
    C1 = _extract_patch_coeffs(u_global, g1)
    C2 = _extract_patch_coeffs(u_global, g2)

    eta2_vol = eta2_vol + _vol_for_patch(C0, bp_xL, bp_yB)
    eta2_vol = eta2_vol + _vol_for_patch(C1, bp_xL, bp_yT)
    eta2_vol = eta2_vol + _vol_for_patch(C2, bp_xR, bp_yT)

    # ------------------------------------------------------------------
    # Jump terms on Gamma01 (y=0, x∈[-1,0]) and Gamma12 (x=0, y∈[0,1])
    # ------------------------------------------------------------------
    eta2_jump01 = jnp.asarray(0.0, dtype=dtype)
    eta2_jump12 = jnp.asarray(0.0, dtype=dtype)

    if bool(include_jump):
        eps_t = jnp.asarray(eps_side, dtype=dtype)

        # ---- Gamma01: normal +y, jump in uy between P0(top) and P1(bottom)
        # x quadrature on [-1,0] (shared by P0 and P1)
        _, _, wx, hx, Gx, Nx, _, _ = eval_1d_on_elements(bp_xL, p, q_order=q_order, n_der=0)

        # y-derivative samples near y=0 from each side
        nEyB = int(bp_yB.shape[0] - 1)
        nEyT = int(bp_yT.shape[0] - 1)
        e_last_yB = nEyB - 1
        e_first_yT = 0

        hy0 = bp_yB[-1] - bp_yB[-2]
        hy1 = bp_yT[1] - bp_yT[0]
        y0 = bp_yB[-1] - eps_t * hy0
        y1 = bp_yT[0] + eps_t * hy1

        dNy0 = _dN_at_point_in_element(y0, knots_yB, p=p, e=e_last_yB)
        dNy1 = _dN_at_point_in_element(y1, knots_yT, p=p, e=e_first_yT)

        Gy0 = jnp.asarray(e_last_yB, dtype=jnp.int32) + jnp.arange(p + 1, dtype=jnp.int32)
        Gy1 = jnp.arange(p + 1, dtype=jnp.int32)

        # Gather (per x-element) coefficient blocks for the fixed y-band near interface
        C0x = C0[Gy0[:, None, None], Gx[None, :, :]]  # (p+1,NxE,p+1)
        C0x = jnp.transpose(C0x, (1, 0, 2))  # (NxE,p+1,p+1)
        C1x = C1[Gy1[:, None, None], Gx[None, :, :]]
        C1x = jnp.transpose(C1x, (1, 0, 2))

        du0 = jnp.einsum("i,eij,eqj->eq", dNy0, C0x, Nx)
        du1 = jnp.einsum("i,eij,eqj->eq", dNy1, C1x, Nx)
        J = du0 - du1

        wE = wx * hx[:, None]
        eta2_jump01 = jnp.sum(wE * (J * J))

        # ---- Gamma12: normal +x, jump in ux between P1(right) and P2(left)
        _, _, wy, hy, Gy, Ny, _, _ = eval_1d_on_elements(bp_yT, p, q_order=q_order, n_der=0)

        nExL = int(bp_xL.shape[0] - 1)
        nExR = int(bp_xR.shape[0] - 1)
        e_last_xL = nExL - 1
        e_first_xR = 0

        hx1 = bp_xL[-1] - bp_xL[-2]
        hx2 = bp_xR[1] - bp_xR[0]
        x1 = bp_xL[-1] - eps_t * hx1
        x2 = bp_xR[0] + eps_t * hx2

        dNx1 = _dN_at_point_in_element(x1, knots_xL, p=p, e=e_last_xL)
        dNx2 = _dN_at_point_in_element(x2, knots_xR, p=p, e=e_first_xR)

        Gx1 = jnp.asarray(e_last_xL, dtype=jnp.int32) + jnp.arange(p + 1, dtype=jnp.int32)
        Gx2 = jnp.arange(p + 1, dtype=jnp.int32)

        C1y = C1[Gy[:, :, None], Gx1[None, None, :]]  # (NyE,p+1,p+1)
        C2y = C2[Gy[:, :, None], Gx2[None, None, :]]

        du1x = jnp.einsum("eqi,eij,j->eq", Ny, C1y, dNx1)
        du2x = jnp.einsum("eqi,eij,j->eq", Ny, C2y, dNx2)
        J2 = du1x - du2x

        wE2 = wy * hy[:, None]
        eta2_jump12 = jnp.sum(wE2 * (J2 * J2))

    eta2_jump = eta2_jump01 + eta2_jump12
    eta2_total = eta2_vol + eta2_jump
    return eta2_total, (eta2_vol, eta2_jump, eta2_jump01, eta2_jump12)


__all__ = ["estimator_eta2"]
