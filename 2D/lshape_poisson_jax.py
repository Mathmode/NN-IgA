from __future__ import annotations

r"""Matrix-free JAX solver for the 3-patch L-shape Poisson/Laplace problem.

We solve:
    -Δu = 0  in Ω,
    u = g_D  on ∂Ω,

on the classical L-shaped domain:
    Ω = (-1,1)^2 \ ([0,1] x [-1,0])

Discretization:
  - 3 tensor-product B-spline patches (P0,P1,P2) with conforming interface DOF
    identification (no mortaring).
  - Strong Dirichlet BCs enforced by lifting/elimination.

Performance goals:
  - Matrix-free CG solve for larger problems (supports N>32 per patch).
  - Optional dense direct solve for moderate DOF counts (useful for debugging
    and to mirror the single-patch "dense-ish" solvers in the square examples).
  - JIT-friendly: mesh topology (N per interval) fixed per run.
"""

from dataclasses import dataclass
import functools
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
from jax import lax
from jax.scipy.linalg import solve_triangular

from iga2d_jax import assemble_1d_mats, greville_abscissae
from reference_exact_jax import u_exact

Array = jnp.ndarray


@dataclass(frozen=True)
class PatchSystem:
    name: str
    breakpoints_x: Array
    breakpoints_y: Array
    knots_x: Array
    knots_y: Array
    p: int
    nBx: int
    nBy: int


@dataclass(frozen=True)
class LShapeSystem:
    p: int
    patches: Dict[str, PatchSystem]
    g0: Array
    g1: Array
    g2: Array
    dirichlet: Array  # (n_dir,)
    n_dofs: int


def _apply_patch_laplace(u_loc: Array, *, Mx: Array, Kx: Array, My: Array, Ky: Array, nBx: int, nBy: int) -> Array:
    """Apply A = kron(My,Kx) + kron(Ky,Mx) to a local vector (row-major)."""
    C = u_loc.reshape(int(nBy), int(nBx))
    Y = My @ C @ Kx.T + Ky @ C @ Mx.T
    return Y.reshape(-1)


def _patch_diag(*, Mx: Array, Kx: Array, My: Array, Ky: Array) -> Array:
    """Diagonal of kron(My,Kx)+kron(Ky,Mx) in row-major ordering."""
    dx = jnp.diag(Kx)
    mx = jnp.diag(Mx)
    dy = jnp.diag(Ky)
    my = jnp.diag(My)
    return (my[:, None] * dx[None, :] + dy[:, None] * mx[None, :]).reshape(-1)


def _generalized_eigh_spd(K: Array, M: Array) -> tuple[Array, Array]:
    """Generalized eig for SPD pair: K v = M v lam with M-orthonormal vectors.

    Returns (lam, V) where V^T M V = I and V^T K V = diag(lam).
    """
    L = jnp.linalg.cholesky(M)
    tmp = solve_triangular(L, K, lower=True)
    A_hat = solve_triangular(L, tmp.T, lower=True).T
    A_hat = 0.5 * (A_hat + A_hat.T)
    lam, Q = jnp.linalg.eigh(A_hat)
    V = solve_triangular(L.T, Q, lower=False)
    return lam, V


def _fd_solve_kron_sum(
    r_vec: Array, *, nBx: int, nBy: int, lamx: Array, Vx: Array, lamy: Array, Vy: Array
) -> Array:
    """Solve (kron(My,Kx)+kron(Ky,Mx)) z = r for a tensor-product block via FD."""
    R = r_vec.reshape(int(nBy), int(nBx))
    Rt = Vy.T @ R @ Vx
    denom = lamy[:, None] + lamx[None, :]
    X = Rt / denom
    Z = Vy @ X @ Vx.T
    return Z.reshape(-1)


@functools.partial(
    jax.jit,
    static_argnames=("p", "q_order", "solver", "dense_max_dofs", "cg_tol", "cg_maxiter", "precond", "n_dofs"),
)
def solve_lshape_coeffs(
    bp_xL: Array,
    bp_xR: Array,
    bp_yB: Array,
    bp_yT: Array,
    g0: Array,
    g1: Array,
    g2: Array,
    dirichlet: Array,
    free_idx: Array,
    x0_free: Array,
    *,
    p: int,
    q_order: int = 20,
    solver: str = "cg",
    dense_max_dofs: int = 1500,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 4000,
    precond: str = "jacobi",
    n_dofs: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """Solve and return (u_global, knots_xL, knots_xR, knots_yB, knots_yT).

    Args:
        x0_free: Initial guess for the free-DOF correction w (CG warm-start).
    """
    p = int(p)
    solver_l = str(solver).lower().strip()
    precond_l = str(precond).lower().strip()
    dtype = jnp.asarray(bp_xL).dtype
    n_dofs = int(n_dofs)
    dense_max_dofs = int(dense_max_dofs)

    if solver_l not in ("cg", "dense", "auto"):
        raise ValueError("solver must be one of: cg, dense, auto")
    if precond_l not in ("none", "jacobi", "block_jacobi"):
        raise ValueError("precond must be one of: none, jacobi, block_jacobi")
    if solver_l == "auto":
        solver_l = "dense" if int(n_dofs) <= int(dense_max_dofs) else "cg"
    if solver_l == "dense" and int(n_dofs) > int(dense_max_dofs):
        raise ValueError(
            f"dense solve requested with n_dofs={int(n_dofs)}, exceeds dense_max_dofs={int(dense_max_dofs)}"
        )
    x0_free = jnp.asarray(x0_free, dtype=dtype).reshape(-1)

    # 1D mats on each interval
    knots_xL, MxL, KxL, _ = assemble_1d_mats(bp_xL, p, q_order=int(q_order))
    knots_xR, MxR, KxR, _ = assemble_1d_mats(bp_xR, p, q_order=int(q_order))
    knots_yB, MyB, KyB, _ = assemble_1d_mats(bp_yB, p, q_order=int(q_order))
    knots_yT, MyT, KyT, _ = assemble_1d_mats(bp_yT, p, q_order=int(q_order))

    # Patch sizes (basis count = n_elem + p)
    nBxL = int(MxL.shape[0])
    nBxR = int(MxR.shape[0])
    nByB = int(MyB.shape[0])
    nByT = int(MyT.shape[0])

    g0f = g0.reshape(-1)
    g1f = g1.reshape(-1)
    g2f = g2.reshape(-1)

    # Map global dof -> position in free vector (-1 for Dirichlet dofs)
    n_free = int(free_idx.shape[0])
    inv_free = -jnp.ones((n_dofs,), dtype=jnp.int32)
    inv_free = inv_free.at[free_idx].set(jnp.arange(n_free, dtype=jnp.int32))

    pos0_all = inv_free[g0f]
    pos1_all = inv_free[g1f]
    pos2_all = inv_free[g2f]
    idx0_all = jnp.maximum(pos0_all, jnp.asarray(0, dtype=jnp.int32))
    idx1_all = jnp.maximum(pos1_all, jnp.asarray(0, dtype=jnp.int32))
    idx2_all = jnp.maximum(pos2_all, jnp.asarray(0, dtype=jnp.int32))
    mask0_all = pos0_all >= 0
    mask1_all = pos1_all >= 0
    mask2_all = pos2_all >= 0

    def matvec_full(x_full: Array) -> Array:
        # P0
        x0 = jnp.asarray(x_full)[g0f]
        y0 = _apply_patch_laplace(x0, Mx=MxL, Kx=KxL, My=MyB, Ky=KyB, nBx=nBxL, nBy=nByB)

        # P1
        x1 = jnp.asarray(x_full)[g1f]
        y1 = _apply_patch_laplace(x1, Mx=MxL, Kx=KxL, My=MyT, Ky=KyT, nBx=nBxL, nBy=nByT)

        # P2
        x2 = jnp.asarray(x_full)[g2f]
        y2 = _apply_patch_laplace(x2, Mx=MxR, Kx=KxR, My=MyT, Ky=KyT, nBx=nBxR, nBy=nByT)
        idx = jnp.concatenate([g0f, g1f, g2f], axis=0)
        vals = jnp.concatenate([y0, y1, y2], axis=0)
        y = jnp.zeros((n_dofs,), dtype=dtype)
        return y.at[idx].add(vals)

    # Dirichlet values from Greville abscissae (averaged across patches for shared DOFs)
    gxL = greville_abscissae(knots_xL, p)
    gxR = greville_abscissae(knots_xR, p)
    gyB = greville_abscissae(knots_yB, p)
    gyT = greville_abscissae(knots_yT, p)

    sum_vals = jnp.zeros((n_dofs,), dtype=dtype)
    counts = jnp.zeros((n_dofs,), dtype=dtype)

    def _add_dirichlet_edge(sum_vals_in: Array, counts_in: Array, gids: Array, x: Array, y: Array) -> tuple[Array, Array]:
        v = u_exact(x, y).reshape(-1)
        sum_vals_in = sum_vals_in.at[gids].add(v)
        counts_in = counts_in.at[gids].add(jnp.asarray(1.0, dtype=dtype))
        return sum_vals_in, counts_in

    # Only evaluate the exact solution on Dirichlet boundary DOFs (O(N) points),
    # rather than on the full Greville grids (O(N^2) points).
    #
    # P0 boundaries: x=-1, y=-1, x=0 (cutout)
    sum_vals, counts = _add_dirichlet_edge(
        sum_vals,
        counts,
        g0[:, 0].reshape(-1),
        jnp.full_like(gyB, gxL[0]),
        gyB,
    )
    sum_vals, counts = _add_dirichlet_edge(
        sum_vals,
        counts,
        g0[0, :].reshape(-1),
        gxL,
        jnp.full_like(gxL, gyB[0]),
    )
    sum_vals, counts = _add_dirichlet_edge(
        sum_vals,
        counts,
        g0[:, -1].reshape(-1),
        jnp.full_like(gyB, gxL[-1]),
        gyB,
    )

    # P1 boundaries: x=-1, y=1
    sum_vals, counts = _add_dirichlet_edge(
        sum_vals,
        counts,
        g1[:, 0].reshape(-1),
        jnp.full_like(gyT, gxL[0]),
        gyT,
    )
    sum_vals, counts = _add_dirichlet_edge(
        sum_vals,
        counts,
        g1[-1, :].reshape(-1),
        gxL,
        jnp.full_like(gxL, gyT[-1]),
    )

    # P2 boundaries: x=1, y=1, y=0 (cutout)
    sum_vals, counts = _add_dirichlet_edge(
        sum_vals,
        counts,
        g2[:, -1].reshape(-1),
        jnp.full_like(gyT, gxR[-1]),
        gyT,
    )
    sum_vals, counts = _add_dirichlet_edge(
        sum_vals,
        counts,
        g2[0, :].reshape(-1),
        gxR,
        jnp.full_like(gxR, gyT[0]),
    )
    sum_vals, counts = _add_dirichlet_edge(
        sum_vals,
        counts,
        g2[-1, :].reshape(-1),
        gxR,
        jnp.full_like(gxR, gyT[-1]),
    )

    sum_dir = sum_vals[dirichlet]
    cnt_dir = counts[dirichlet]
    dir_vals = sum_dir / jnp.maximum(cnt_dir, jnp.asarray(1.0, dtype=dtype))

    # Lifting: u = uD + w, w|_Gamma=0
    uD = jnp.zeros((n_dofs,), dtype=dtype).at[dirichlet].set(dir_vals)

    if solver_l == "dense":
        A = jnp.zeros((n_dofs, n_dofs), dtype=dtype)
        A0 = jnp.kron(MyB, KxL) + jnp.kron(KyB, MxL)
        A1 = jnp.kron(MyT, KxL) + jnp.kron(KyT, MxL)
        A2 = jnp.kron(MyT, KxR) + jnp.kron(KyT, MxR)

        A = A.at[g0f[:, None], g0f[None, :]].add(A0)
        A = A.at[g1f[:, None], g1f[None, :]].add(A1)
        A = A.at[g2f[:, None], g2f[None, :]].add(A2)

        b_eff = -(A @ uD)
        b_free = b_eff[free_idx]
        A_ff = A[free_idx[:, None], free_idx[None, :]]

        L = jnp.linalg.cholesky(A_ff)
        y = solve_triangular(L, b_free, lower=True)
        w_free = solve_triangular(L.T, y, lower=False)
        u = uD.at[free_idx].add(w_free)
        return u, knots_xL, knots_xR, knots_yB, knots_yT

    def matvec_free(x_free: Array) -> Array:
        # Gather per-patch vectors from the free vector without constructing a
        # dense global vector (speeds up CG iterations).
        x_free = jnp.asarray(x_free, dtype=dtype).reshape(-1)

        x0 = jnp.take(x_free, idx0_all)
        x0 = jnp.where(mask0_all, x0, jnp.asarray(0.0, dtype=dtype))
        y0 = _apply_patch_laplace(x0, Mx=MxL, Kx=KxL, My=MyB, Ky=KyB, nBx=nBxL, nBy=nByB)

        x1 = jnp.take(x_free, idx1_all)
        x1 = jnp.where(mask1_all, x1, jnp.asarray(0.0, dtype=dtype))
        y1 = _apply_patch_laplace(x1, Mx=MxL, Kx=KxL, My=MyT, Ky=KyT, nBx=nBxL, nBy=nByT)

        x2 = jnp.take(x_free, idx2_all)
        x2 = jnp.where(mask2_all, x2, jnp.asarray(0.0, dtype=dtype))
        y2 = _apply_patch_laplace(x2, Mx=MxR, Kx=KxR, My=MyT, Ky=KyT, nBx=nBxR, nBy=nByT)

        y_free = jnp.zeros((n_free,), dtype=dtype)
        y_free = y_free.at[idx0_all].add(jnp.where(mask0_all, y0, jnp.asarray(0.0, dtype=dtype)))
        y_free = y_free.at[idx1_all].add(jnp.where(mask1_all, y1, jnp.asarray(0.0, dtype=dtype)))
        y_free = y_free.at[idx2_all].add(jnp.where(mask2_all, y2, jnp.asarray(0.0, dtype=dtype)))
        return y_free

    # RHS is zero; only lifting contributes: b = -A*uD restricted to free rows.
    y0D = _apply_patch_laplace(uD[g0f], Mx=MxL, Kx=KxL, My=MyB, Ky=KyB, nBx=nBxL, nBy=nByB)
    y1D = _apply_patch_laplace(uD[g1f], Mx=MxL, Kx=KxL, My=MyT, Ky=KyT, nBx=nBxL, nBy=nByT)
    y2D = _apply_patch_laplace(uD[g2f], Mx=MxR, Kx=KxR, My=MyT, Ky=KyT, nBx=nBxR, nBy=nByT)
    b_free = jnp.zeros((n_free,), dtype=dtype)
    b_free = b_free.at[idx0_all].add(jnp.where(mask0_all, -y0D, jnp.asarray(0.0, dtype=dtype)))
    b_free = b_free.at[idx1_all].add(jnp.where(mask1_all, -y1D, jnp.asarray(0.0, dtype=dtype)))
    b_free = b_free.at[idx2_all].add(jnp.where(mask2_all, -y2D, jnp.asarray(0.0, dtype=dtype)))

    M_apply = None
    if precond_l == "jacobi":
        d0 = _patch_diag(Mx=MxL, Kx=KxL, My=MyB, Ky=KyB)
        d1 = _patch_diag(Mx=MxL, Kx=KxL, My=MyT, Ky=KyT)
        d2 = _patch_diag(Mx=MxR, Kx=KxR, My=MyT, Ky=KyT)

        diag_free = jnp.zeros((n_free,), dtype=dtype)
        diag_free = diag_free.at[idx0_all].add(jnp.where(mask0_all, d0, jnp.asarray(0.0, dtype=dtype)))
        diag_free = diag_free.at[idx1_all].add(jnp.where(mask1_all, d1, jnp.asarray(0.0, dtype=dtype)))
        diag_free = diag_free.at[idx2_all].add(jnp.where(mask2_all, d2, jnp.asarray(0.0, dtype=dtype)))
        inv_diag_free = 1.0 / jnp.maximum(diag_free, jnp.asarray(1e-30, dtype=dtype))

        def M_apply(r_free: Array) -> Array:
            return inv_diag_free * r_free

    if precond_l == "block_jacobi":
        # Block index selectors (match the original torch block-Jacobi scheme).
        kxL0 = jnp.arange(nBxL, dtype=jnp.int32)[1:-1]  # exclude x=-1 and x=0-cutout
        kyB0 = jnp.arange(nByB, dtype=jnp.int32)[1:]  # exclude y=-1; keep interface y=0

        kxL1 = jnp.arange(nBxL, dtype=jnp.int32)[1:-1]  # exclude x=-1 and x=0 interface
        kyT1 = jnp.arange(nByT, dtype=jnp.int32)[1:-1]  # exclude y=0 interface and y=1

        kxR2 = jnp.arange(nBxR, dtype=jnp.int32)[:-1]  # keep x=0 interface; exclude x=1
        kyT2 = jnp.arange(nByT, dtype=jnp.int32)[1:-1]  # exclude y=0 cutout and y=1

        pos0 = inv_free[g0][kyB0[:, None], kxL0[None, :]].reshape(-1)
        pos1 = inv_free[g1][kyT1[:, None], kxL1[None, :]].reshape(-1)
        pos2 = inv_free[g2][kyT2[:, None], kxR2[None, :]].reshape(-1)

        # FD factors for each block (stop gradients: preconditioner only accelerates CG).
        lamx0, Vx0 = _generalized_eigh_spd(KxL[kxL0][:, kxL0], MxL[kxL0][:, kxL0])
        lamy0, Vy0 = _generalized_eigh_spd(KyB[kyB0][:, kyB0], MyB[kyB0][:, kyB0])

        lamx1, Vx1 = _generalized_eigh_spd(KxL[kxL1][:, kxL1], MxL[kxL1][:, kxL1])
        lamy1, Vy1 = _generalized_eigh_spd(KyT[kyT1][:, kyT1], MyT[kyT1][:, kyT1])

        lamx2, Vx2 = _generalized_eigh_spd(KxR[kxR2][:, kxR2], MxR[kxR2][:, kxR2])
        lamy2, Vy2 = _generalized_eigh_spd(KyT[kyT2][:, kyT2], MyT[kyT2][:, kyT2])

        lamx0, Vx0, lamy0, Vy0 = lax.stop_gradient(lamx0), lax.stop_gradient(Vx0), lax.stop_gradient(lamy0), lax.stop_gradient(Vy0)
        lamx1, Vx1, lamy1, Vy1 = lax.stop_gradient(lamx1), lax.stop_gradient(Vx1), lax.stop_gradient(lamy1), lax.stop_gradient(Vy1)
        lamx2, Vx2, lamy2, Vy2 = lax.stop_gradient(lamx2), lax.stop_gradient(Vx2), lax.stop_gradient(lamy2), lax.stop_gradient(Vy2)

        nBx0 = int(kxL0.shape[0])
        nBy0 = int(kyB0.shape[0])
        nBx1 = int(kxL1.shape[0])
        nBy1 = int(kyT1.shape[0])
        nBx2 = int(kxR2.shape[0])
        nBy2 = int(kyT2.shape[0])

        def M_apply(r_free: Array) -> Array:
            z = jnp.zeros_like(r_free)
            r0 = r_free[pos0]
            z0 = _fd_solve_kron_sum(r0, nBx=nBx0, nBy=nBy0, lamx=lamx0, Vx=Vx0, lamy=lamy0, Vy=Vy0)
            z = z.at[pos0].add(z0)

            r1 = r_free[pos1]
            z1 = _fd_solve_kron_sum(r1, nBx=nBx1, nBy=nBy1, lamx=lamx1, Vx=Vx1, lamy=lamy1, Vy=Vy1)
            z = z.at[pos1].add(z1)

            r2 = r_free[pos2]
            z2 = _fd_solve_kron_sum(r2, nBx=nBx2, nBy=nBy2, lamx=lamx2, Vx=Vx2, lamy=lamy2, Vy=Vy2)
            z = z.at[pos2].add(z2)
            return z

    # Warm-start without passing x0 to cg: JAX uses the same x0 for the
    # implicit-diff transpose solve, which may hurt. Instead solve for the
    # correction δ = w - x0:
    #     A δ = (b - A x0),  with δ0 = 0.
    b_corr = b_free - matvec_free(x0_free)
    delta_free, _ = jsla.cg(
        matvec_free,
        b_corr,
        tol=float(cg_tol),
        maxiter=int(cg_maxiter),
        M=M_apply,
    )
    w_free = x0_free + delta_free

    u = uD.at[free_idx].add(w_free)
    return u, knots_xL, knots_xR, knots_yB, knots_yT


def solve_lshape(
    bp_xL: Array,
    bp_xR: Array,
    bp_yB: Array,
    bp_yT: Array,
    *,
    p: int,
    q_order: int,
    solver: str = "cg",
    dense_max_dofs: int = 1500,
    cg_tol: float,
    cg_maxiter: int,
    precond: str = "jacobi",
    g0: Array,
    g1: Array,
    g2: Array,
    dirichlet: Array,
    free_idx: Array,
    n_dofs: int,
) -> tuple[Array, LShapeSystem]:
    """Convenience wrapper returning (u_global, sys) for exporting/postprocessing."""
    x0_free = jnp.zeros((int(jnp.asarray(free_idx).shape[0]),), dtype=jnp.asarray(bp_xL).dtype)
    u, knots_xL, knots_xR, knots_yB, knots_yT = solve_lshape_coeffs(
        bp_xL,
        bp_xR,
        bp_yB,
        bp_yT,
        g0,
        g1,
        g2,
        dirichlet,
        free_idx,
        x0_free,
        p=int(p),
        q_order=int(q_order),
        solver=str(solver),
        dense_max_dofs=int(dense_max_dofs),
        cg_tol=float(cg_tol),
        cg_maxiter=int(cg_maxiter),
        precond=str(precond),
        n_dofs=int(n_dofs),
    )

    nBxL = int(bp_xL.shape[0] - 1 + int(p))
    nBxR = int(bp_xR.shape[0] - 1 + int(p))
    nByB = int(bp_yB.shape[0] - 1 + int(p))
    nByT = int(bp_yT.shape[0] - 1 + int(p))

    patches = {
        "P0": PatchSystem("P0", bp_xL, bp_yB, knots_xL, knots_yB, int(p), nBxL, nByB),
        "P1": PatchSystem("P1", bp_xL, bp_yT, knots_xL, knots_yT, int(p), nBxL, nByT),
        "P2": PatchSystem("P2", bp_xR, bp_yT, knots_xR, knots_yT, int(p), nBxR, nByT),
    }
    sys = LShapeSystem(p=int(p), patches=patches, g0=g0, g1=g1, g2=g2, dirichlet=dirichlet, n_dofs=int(n_dofs))
    return u, sys


__all__ = [
    "PatchSystem",
    "LShapeSystem",
    "solve_lshape_coeffs",
    "solve_lshape",
]
