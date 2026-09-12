from __future__ import annotations

"""Exact transmission solution and PDE data for the helmholtz experiment (1D Helmholtz).

Problem (Omega = (0,1), interface x_I = 1/2, frequency parameter omega):

    -(sigma u')' + alpha(x;omega) u = 0      in (0,x_I) U (x_I,1)
     u(0) = 0                                 (Dirichlet, left)
     sigma(1) u'(1) = g_N = 10 pi             (Neumann, right)
     [u]_{x_I} = 0,  [sigma u']_{x_I} = 0     (transmission)

with the FIXED data

    sigma(x) = 1            on (0,1/2),   4         on (1/2,1)
    alpha    = -omega^2 rho,  rho = (31 pi/2)^2 on (0,1/2),  100 pi^2 on (1/2,1)

Per region the strong form is  u'' + k_j^2 u = 0  with local wavenumber
    k_j(omega) = omega sqrt(rho_j / sigma_j),   k1 = (31 pi/2) omega,  k2 = 5 pi omega.

Source-free => the exact field is piecewise-trigonometric:

    u*(x) = A sin(k1 x)                  on [0, x_I]      (cos killed by u(0)=0)
            C sin(k2 x) + D cos(k2 x)    on [x_I, 1]

with (A, C, D) from the 3x3 system of the two transmission conditions + the
Neumann condition (`matching_matrix`). The matching determinant vanishes at
transmission resonances; `resonance_scan` locates them so the parameter
sampler can exclude a neighbourhood (the relative metrics blow up where
|u*|_{H1} -> 0 / the operator is singular).

Pure NumPy: this is the analytic reference, not part of the autodiff path.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# --- fixed PDE data -------------------------------------------------------
X_I: float = 0.5
SIGMA1: float = 1.0
SIGMA2: float = 4.0
RHO1: float = (31.0 * math.pi / 2.0) ** 2
RHO2: float = 100.0 * math.pi ** 2
G_N: float = 10.0 * math.pi


def wavenumbers(omega: float) -> Tuple[float, float]:
    """Local wavenumbers (k1, k2) = omega * sqrt(rho_j / sigma_j)."""
    k1 = float(omega) * math.sqrt(RHO1 / SIGMA1)
    k2 = float(omega) * math.sqrt(RHO2 / SIGMA2)
    return k1, k2


def matching_matrix(omega: float) -> np.ndarray:
    """3x3 matrix M(omega) of the (A, C, D) system; rhs = (0, 0, g_N)."""
    k1, k2 = wavenumbers(omega)
    s1k1, s2k2 = SIGMA1 * k1, SIGMA2 * k2
    xI = X_I
    return np.array([
        # continuity of u at x_I
        [math.sin(k1 * xI), -math.sin(k2 * xI), -math.cos(k2 * xI)],
        # continuity of sigma u' at x_I
        [s1k1 * math.cos(k1 * xI), -s2k2 * math.cos(k2 * xI), s2k2 * math.sin(k2 * xI)],
        # Neumann at x = 1:  sigma2 u'(1) = g_N
        [0.0, s2k2 * math.cos(k2), -s2k2 * math.sin(k2)],
    ], dtype=np.float64)


def solve_coeffs(omega: float) -> Tuple[float, float, float]:
    """Solve M(omega) [A, C, D]^T = (0, 0, g_N)."""
    M = matching_matrix(omega)
    rhs = np.array([0.0, 0.0, G_N], dtype=np.float64)
    A, C, D = np.linalg.solve(M, rhs)
    return float(A), float(C), float(D)


@dataclass(frozen=True)
class HelmholtzExact:
    """Callable exact field for one omega."""
    omega: float
    A: float
    C: float
    D: float
    k1: float
    k2: float

    def u(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        left = self.A * np.sin(self.k1 * x)
        right = self.C * np.sin(self.k2 * x) + self.D * np.cos(self.k2 * x)
        return np.where(x <= X_I, left, right)

    def du(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        left = self.A * self.k1 * np.cos(self.k1 * x)
        right = self.C * self.k2 * np.cos(self.k2 * x) - self.D * self.k2 * np.sin(self.k2 * x)
        return np.where(x <= X_I, left, right)

    def d2u(self, x: np.ndarray) -> np.ndarray:
        # u'' = -k^2 u per region (strong form).
        x = np.asarray(x, dtype=np.float64)
        left = -(self.k1 ** 2) * (self.A * np.sin(self.k1 * x))
        right = -(self.k2 ** 2) * (self.C * np.sin(self.k2 * x) + self.D * np.cos(self.k2 * x))
        return np.where(x <= X_I, left, right)

    def h1_seminorm(self, n_gauss: int = 64, n_sub: int = 32) -> float:
        """|u*|_{H1-semi} = sqrt( int_0^1 (u*')^2 dx ), composite Gauss per region."""
        val = 0.0
        for a, b in ((0.0, X_I), (X_I, 1.0)):
            edges = np.linspace(a, b, n_sub + 1)
            xg, wg = np.polynomial.legendre.leggauss(n_gauss)
            for e in range(n_sub):
                lo, hi = edges[e], edges[e + 1]
                xm = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
                wm = 0.5 * (hi - lo) * wg
                du = self.du(xm)
                val += float(np.sum(wm * du * du))
        return math.sqrt(max(val, 0.0))

    def sigma_at(self, x: float) -> float:
        return SIGMA1 if x <= X_I else SIGMA2


def helmholtz_exact(omega: float) -> HelmholtzExact:
    """Build the exact field for one omega (solves the 3x3 matching system)."""
    A, C, D = solve_coeffs(omega)
    k1, k2 = wavenumbers(omega)
    return HelmholtzExact(omega=float(omega), A=A, C=C, D=D, k1=k1, k2=k2)


def resonance_scan(omega_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (det(omega) over the grid, detected resonance omegas).

    A resonance is a sign change of the matching determinant; the location is
    refined by linear interpolation of the bracket. `det` is normalised by a
    smooth omega^3 scale so the zero crossings (not the growing amplitude)
    dominate the sign structure.
    """
    omega_grid = np.asarray(omega_grid, dtype=np.float64)
    det = np.array([np.linalg.det(matching_matrix(w)) for w in omega_grid], dtype=np.float64)
    scale = np.maximum(np.abs(omega_grid) ** 3, 1e-12)
    det_n = det / scale
    res = []
    for i in range(len(omega_grid) - 1):
        a, b = det_n[i], det_n[i + 1]
        if a == 0.0:
            res.append(float(omega_grid[i]))
        elif a * b < 0.0:
            t = a / (a - b)
            res.append(float(omega_grid[i] + t * (omega_grid[i + 1] - omega_grid[i])))
    return det, np.array(res, dtype=np.float64)


def sample_omegas(omega_min: float, omega_max: float, n: int,
                  resonance_tol: float, scan_pts: int = 8000) -> np.ndarray:
    """Resonance-safe omega sample of size <= n on [omega_min, omega_max].

    The high fixed densities make [1,4] resonance-rich, so we cannot start from
    n evenly-spaced points (most would fall in an exclusion band). Instead we
    take a dense candidate set, drop every point within +/-resonance_tol of a
    detected resonance, then evenly subsample n of the survivors (endpoints
    kept when safe). Returns the survivors (size <= n)."""
    grid = np.linspace(omega_min, omega_max, int(scan_pts))
    _det, res = resonance_scan(grid)
    cand = np.linspace(omega_min, omega_max, max(int(n) * 20, 400))
    if res.size:
        safe = np.all(np.abs(cand[:, None] - res[None, :]) > float(resonance_tol), axis=1)
        cand = cand[safe]
    if cand.size <= int(n):
        return cand
    idx = np.linspace(0, cand.size - 1, int(n)).round().astype(int)
    return np.unique(cand[idx])


__all__ = [
    "X_I", "SIGMA1", "SIGMA2", "RHO1", "RHO2", "G_N",
    "wavenumbers", "matching_matrix", "solve_coeffs",
    "HelmholtzExact", "helmholtz_exact", "resonance_scan", "sample_omegas",
]
