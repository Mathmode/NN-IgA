from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def axis_quadrature(knots, p):
    knots = np.asarray(knots, dtype=np.longdouble)
    spans = np.flatnonzero(np.diff(knots) > 0)
    a = knots[spans]
    h = knots[spans + 1] - a
    z, w = np.polynomial.legendre.leggauss(p + 1)
    x = h[:, None] * ((1 + z.astype(np.longdouble)) / 2)
    basis = np.zeros((*x.shape, p + 1), dtype=x.dtype)
    basis[..., 0] = 1
    for degree in range(1, p + 1):
        if degree == p:
            lower = basis[..., :p].copy()
        saved = np.zeros_like(x)
        for r in range(degree):
            left = knots[spans + 1 - degree + r]
            right = knots[spans + r + 1]
            scale = basis[..., r] / (right - left)[:, None]
            basis[..., r] = saved + ((right - a)[:, None] - x) * scale
            saved = ((a - left)[:, None] + x) * scale
        basis[..., degree] = saved
    indices = spans[:, None] - p + 1 + np.arange(p)
    lower *= (p / (knots[indices + p] - knots[indices]))[:, None, :]
    return spans - p, a, h[:, None] * w / 2, basis, lower


def energy_and_load(coefficients, knots_x, knots_y, p, sigma1, sigma2):
    ix, x, wx, bx, dx = axis_quadrature(knots_x, p)
    iy, y, wy, by, dy = axis_quadrature(knots_y, p)
    coeffs = np.asarray(coefficients, dtype=np.longdouble)
    blocks = sliding_window_view(coeffs, (p + 1, p + 1))[ix[:, None], iy]

    def evaluate(ax, ay, local):
        return np.einsum("iqk,ijkl,jrl->ijqr", ax, local, ay, optimize=True)

    u = evaluate(bx, by, blocks)
    ux = evaluate(dx, by, np.diff(blocks, axis=2))
    uy = evaluate(bx, dy, np.diff(blocks, axis=3))
    left = x[:, None] < 0.5
    bottom = y[None, :] < 0.5
    material = np.where(left & bottom, sigma1, np.where(~left & ~bottom, sigma2, 1))
    weight = wx[:, None, :, None] * wy[None, :, None, :] * (left | ~bottom)[:, :, None, None]
    load = np.sum(weight * u)
    strain = np.sum(weight * material[:, :, None, None] * (ux * ux + uy * uy))
    return strain, load
