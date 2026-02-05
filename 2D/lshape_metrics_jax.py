from __future__ import annotations

"""Error metrics for the multipatch L-shape Laplace problem (JAX)."""

import functools

import jax
import jax.numpy as jnp

from iga2d_jax import eval_1d_on_elements
from reference_exact_jax import u_exact, grad_u_exact

Array = jnp.ndarray


def _extract_patch_coeffs(u_global: Array, gmap: Array) -> Array:
    nBy, nBx = int(gmap.shape[0]), int(gmap.shape[1])
    return jnp.asarray(u_global)[jnp.asarray(gmap).reshape(-1)].reshape(nBy, nBx)


@functools.partial(jax.jit, static_argnames=("p", "q_order"))
def _error_integrals(
    u_global: Array,
    *,
    bp_x: Array,
    bp_y: Array,
    gmap: Array,
    p: int,
    q_order: int,
) -> tuple[Array, Array, Array, Array]:
    """Return (I_L2, I_H1s, I_L2_ref, I_H1s_ref) for one patch."""
    p = int(p)
    q_order = int(q_order)
    dtype = jnp.asarray(u_global).dtype

    C = _extract_patch_coeffs(u_global, gmap)

    _, xq, wx, _, Gx, Nx, Dx, _ = eval_1d_on_elements(bp_x, p, q_order=q_order, n_der=1)
    _, yq, wy, _, Gy, Ny, Dy, _ = eval_1d_on_elements(bp_y, p, q_order=q_order, n_der=1)

    C_blocks = C[Gy[:, :, None, None], Gx[None, None, :, :]]  # (Ny,p+1,Nx,p+1)
    C_blocks = jnp.transpose(C_blocks, (0, 2, 1, 3))  # (Ny,Nx,p+1,p+1)

    uh = jnp.einsum("yai,yxij,xbj->yxab", Ny, C_blocks, Nx)
    ux = jnp.einsum("yai,yxij,xbj->yxab", Ny, C_blocks, Dx)
    uy = jnp.einsum("yai,yxij,xbj->yxab", Dy, C_blocks, Nx)

    NyE, qy = int(yq.shape[0]), int(yq.shape[1])
    NxE, qx = int(xq.shape[0]), int(xq.shape[1])
    X = jnp.broadcast_to(xq[None, :, None, :], (NyE, NxE, qy, qx))
    Y = jnp.broadcast_to(yq[:, None, :, None], (NyE, NxE, qy, qx))

    uex = u_exact(X, Y)
    uex_x, uex_y = grad_u_exact(X, Y)

    w2d = wy[:, None, :, None] * wx[None, :, None, :]

    e0 = uex - uh
    ex0 = uex_x - ux
    ey0 = uex_y - uy

    I_L2 = jnp.sum(w2d * (e0 * e0))
    I_H1s = jnp.sum(w2d * (ex0 * ex0 + ey0 * ey0))

    I_L2_ref = jnp.sum(w2d * (uex * uex))
    I_H1s_ref = jnp.sum(w2d * (uex_x * uex_x + uex_y * uex_y))
    return I_L2, I_H1s, I_L2_ref, I_H1s_ref


def compute_error_metrics(
    u_global: Array,
    *,
    bp_xL: Array,
    bp_xR: Array,
    bp_yB: Array,
    bp_yT: Array,
    g0: Array,
    g1: Array,
    g2: Array,
    p: int,
    q_order: int = 20,
) -> dict:
    """Compute L2/H1 errors and mesh stats as Python floats."""
    p = int(p)

    I_L2_0, I_H1s_0, I_L2r_0, I_H1sr_0 = _error_integrals(u_global, bp_x=bp_xL, bp_y=bp_yB, gmap=g0, p=p, q_order=int(q_order))
    I_L2_1, I_H1s_1, I_L2r_1, I_H1sr_1 = _error_integrals(u_global, bp_x=bp_xL, bp_y=bp_yT, gmap=g1, p=p, q_order=int(q_order))
    I_L2_2, I_H1s_2, I_L2r_2, I_H1sr_2 = _error_integrals(u_global, bp_x=bp_xR, bp_y=bp_yT, gmap=g2, p=p, q_order=int(q_order))

    I_L2 = I_L2_0 + I_L2_1 + I_L2_2
    I_H1s = I_H1s_0 + I_H1s_1 + I_H1s_2
    I_L2_ref = I_L2r_0 + I_L2r_1 + I_L2r_2
    I_H1s_ref = I_H1sr_0 + I_H1sr_1 + I_H1sr_2

    I_L2, I_H1s, I_L2_ref, I_H1s_ref = jax.device_get((I_L2, I_H1s, I_L2_ref, I_H1s_ref))
    I_L2 = float(I_L2)
    I_H1s = float(I_H1s)
    I_L2_ref = float(I_L2_ref)
    I_H1s_ref = float(I_H1s_ref)

    L2_abs = float(jnp.sqrt(I_L2))
    H1_semi_abs = float(jnp.sqrt(I_H1s))
    H1_abs = float(jnp.sqrt(I_L2 + I_H1s))

    L2_rel = float(jnp.sqrt(I_L2 / (I_L2_ref + 1e-30)))
    H1_semi_rel = float(jnp.sqrt(I_H1s / (I_H1s_ref + 1e-30)))
    H1_rel = float(jnp.sqrt((I_L2 + I_H1s) / (I_L2_ref + I_H1s_ref + 1e-30)))

    # Mesh stats (over all four intervals / three patches)
    def _h_stats(bp):
        bp = jnp.asarray(bp)
        h = bp[1:] - bp[:-1]
        return float(jnp.min(h)), float(jnp.max(h))

    hmin_xL, hmax_xL = _h_stats(bp_xL)
    hmin_xR, hmax_xR = _h_stats(bp_xR)
    hmin_yB, hmax_yB = _h_stats(bp_yB)
    hmin_yT, hmax_yT = _h_stats(bp_yT)

    h_min = float(min(hmin_xL, hmin_xR, hmin_yB, hmin_yT))
    h_max = float(max(hmax_xL, hmax_xR, hmax_yB, hmax_yT))

    return {
        "L2_abs": L2_abs,
        "L2_rel": L2_rel,
        "H1_semi_abs": H1_semi_abs,
        "H1_semi_rel": H1_semi_rel,
        "H1_abs": H1_abs,
        "H1_rel": H1_rel,
        "dG_err": H1_semi_abs,
        "h_min": h_min,
        "h_max": h_max,
        "h_ratio": float(h_max / max(h_min, 1e-30)),
        "dofs": int(jnp.asarray(u_global).size),
    }


__all__ = ["compute_error_metrics"]
