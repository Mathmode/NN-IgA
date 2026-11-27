from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eig

X_I = 0.5
SIGMA1, SIGMA2 = 1.0, 4.0
K2 = 5.0 * math.pi
RHO2 = 100.0 * math.pi ** 2
G_N = 10.0 * math.pi
LAMBDA_SCALE = K2 ** 2 * SIGMA1


def uniform_knots(N, p):
    half = int(N) // 2
    left = np.linspace(0.0, X_I, half + 1)[1:-1]
    right = np.linspace(X_I, 1.0, half + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), left, np.full(p, X_I), right, np.ones(p + 1)])


def _basis(knots, p, x):
    n = len(knots) - p - 1
    spline = BSpline(np.asarray(knots, float), np.eye(n), p, extrapolate=False)
    return np.nan_to_num(spline(x)), np.nan_to_num(spline.derivative()(x))


def assemble_blocks(knots, p, nq=None):
    nq = nq or p + 4
    knots = np.asarray(knots, float)
    n = len(knots) - p - 1
    xg, wg = np.polynomial.legendre.leggauss(nq)
    S, ML, MR = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    br = np.unique(knots)
    for a, b in zip(br[:-1], br[1:]):
        x = 0.5 * (b - a) * xg + 0.5 * (a + b)
        w = 0.5 * (b - a) * wg
        B, D = _basis(knots, p, x)
        left = 0.5 * (a + b) < X_I
        S += (SIGMA1 if left else SIGMA2) * (D.T * w) @ D
        if left:
            ML += (B.T * w) @ B
        else:
            MR += (B.T * w) @ B
    return S, ML, MR


def system(knots, p, c, blocks=None):
    S, ML, MR = blocks or assemble_blocks(knots, p)
    return S - RHO2 * MR - LAMBDA_SCALE * c ** 2 * ML


def solve(knots, p, c, blocks=None):
    A = system(knots, p, c, blocks)
    F = np.zeros(A.shape[0])
    F[-1] = G_N
    u = np.zeros(A.shape[0])
    u[1:] = np.linalg.solve(A[1:, 1:], F[1:])
    return u


def exact_coefficients(c):
    k1 = K2 * c
    M = np.array([
        [math.sin(k1 * X_I), -math.sin(K2 * X_I), -math.cos(K2 * X_I)],
        [SIGMA1 * k1 * math.cos(k1 * X_I), -SIGMA2 * K2 * math.cos(K2 * X_I), SIGMA2 * K2 * math.sin(K2 * X_I)],
        [0.0, SIGMA2 * K2 * math.cos(K2), -SIGMA2 * K2 * math.sin(K2)],
    ])
    return k1, np.linalg.solve(M, [0.0, 0.0, G_N])


def exact_solution(c):
    k1, (A, C, D) = exact_coefficients(c)
    return lambda x: np.where(x <= X_I, A * np.sin(k1 * x), C * np.sin(K2 * x) + D * np.cos(K2 * x))


def exact_derivative(c):
    k1, (A, C, D) = exact_coefficients(c)
    return lambda x: np.where(x <= X_I, A * k1 * np.cos(k1 * x), C * K2 * np.cos(K2 * x) - D * K2 * np.sin(K2 * x))


def evaluate(knots, p, u, x):
    return BSpline(np.asarray(knots, float), np.asarray(u, float), p)(x)


def h1_rel(knots, p, u, c, nq=24):
    du = exact_derivative(c)
    dspl = BSpline(np.asarray(knots, float), np.asarray(u, float), p).derivative()
    xg, wg = np.polynomial.legendre.leggauss(nq)
    br = np.unique(knots)
    err = ref = 0.0
    for a, b in zip(br[:-1], br[1:]):
        x = 0.5 * (b - a) * xg + 0.5 * (a + b)
        w = 0.5 * (b - a) * wg
        err += np.sum(w * (dspl(x) - du(x)) ** 2)
        ref += np.sum(w * du(x) ** 2)
    return math.sqrt(err / ref)


def discrete_resonances(knots, p, blocks=None):
    S, ML, MR = blocks or assemble_blocks(knots, p)
    lam = eig((S - RHO2 * MR)[1:, 1:], ML[1:, 1:], right=False)
    lam = lam[np.isfinite(lam)]
    lam = np.real(lam[np.abs(lam.imag) <= 1e-6 * np.abs(lam) + 1e-8])
    return np.sort(np.sqrt(lam[lam > 0] / LAMBDA_SCALE))


def continuous_resonances(c_max):
    m = np.arange(1, int(np.ceil(c_max / 0.4)) + 1)
    c = 0.4 * m
    return c[c <= c_max + 1e-12]


def mode_index(c_exact):
    return int(round(c_exact / 0.4)) - 1


def associated_by_index(c_h, c_exact):
    k = mode_index(c_exact)
    return float(c_h[k]) if k < len(c_h) else float("nan")


def associated_in_window(c_h, c_exact, below=0.15, above=0.35):
    sel = c_h[(c_h >= c_exact - below) & (c_h <= c_exact + above)]
    return float(sel.min()) if sel.size else float("nan")


def nearest(c_h, c):
    return float(c_h[np.argmin(np.abs(c_h - c))]) if len(c_h) else float("nan")


__all__ = [
    "uniform_knots", "assemble_blocks", "system", "solve",
    "exact_solution", "exact_derivative", "evaluate", "h1_rel",
    "discrete_resonances", "continuous_resonances", "mode_index",
    "associated_by_index", "associated_in_window", "nearest",
]
