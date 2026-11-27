from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

CORNER = 0.5


def support_bounds(knots, p):
    t = np.asarray(knots, dtype=np.float64)
    n = t.shape[0] - p - 1
    return t[:n], t[p + 1 : p + 1 + n]


def quadrant_support_mask(knots_x, knots_y, p, corner=CORNER):
    _, hi_x = support_bounds(knots_x, p)
    lo_y, _ = support_bounds(knots_y, p)
    return (hi_x > corner)[:, None] & (lo_y < corner)[None, :]


def support_rule_free_mask(knots_x, knots_y, p, corner=CORNER):
    free = ~quadrant_support_mask(knots_x, knots_y, p, corner)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    return free


def design_matrix(knots, p, x):
    t = np.asarray(knots, dtype=np.float64)
    xc = np.clip(np.asarray(x, dtype=np.float64), t[p], np.nextafter(t[-p - 1], -np.inf))
    return BSpline.design_matrix(xc, t, p).toarray()


def tensor_values(coeffs, knots_x, knots_y, p, xs, ys):
    return design_matrix(knots_x, p, xs) @ np.asarray(coeffs) @ design_matrix(knots_y, p, ys).T


def removed_quadrant_grid(n=201, corner=CORNER):
    return np.linspace(corner, 1.0, n), np.linspace(0.0, corner, n)


def boundary_lines(n=201):
    s = np.linspace(0.0, 1.0, n)
    return [(np.zeros(1), s), (np.ones(1), s), (s, np.zeros(1)), (s, np.ones(1))]


def active_basis_sum_on_quadrant(knots_x, knots_y, p, free, n=201, corner=CORNER):
    xs, ys = removed_quadrant_grid(n, corner)
    return tensor_values(np.asarray(free, dtype=np.float64), knots_x, knots_y, p, xs, ys)


def constrained_coefficient_max(coeffs, knots_x, knots_y, p, corner=CORNER):
    free = support_rule_free_mask(knots_x, knots_y, p, corner)
    return float(np.max(np.abs(np.asarray(coeffs)) * ~free))


def dirichlet_violation(coeffs, knots_x, knots_y, p, n=201, corner=CORNER):
    xs, ys = removed_quadrant_grid(n, corner)
    quad = float(np.max(np.abs(tensor_values(coeffs, knots_x, knots_y, p, xs, ys))))
    edge = max(
        float(np.max(np.abs(tensor_values(coeffs, knots_x, knots_y, p, a, b))))
        for a, b in boundary_lines(n)
    )
    return {"quadrant": quad, "boundary": edge,
            "coefficients": constrained_coefficient_max(coeffs, knots_x, knots_y, p, corner)}


def random_graded_lshape_knots(rng, p, n_half, corner=CORNER):
    left = np.sort(rng.uniform(0.0, corner, n_half - 1))
    right = np.sort(rng.uniform(corner, 1.0, n_half - 1))
    return np.concatenate([np.zeros(p + 1), left, np.full(p, corner), right, np.ones(p + 1)])


def geometric_lshape_knots(p, n_half, refine, corner=CORNER):
    h = (1.0 / float(refine)) ** np.arange(int(n_half))
    h = h / h.sum() * corner
    left = np.cumsum(h)[:-1]
    right = corner + np.cumsum(h[::-1])[:-1]
    return np.concatenate([np.zeros(p + 1), left, np.full(p, corner), right, np.ones(p + 1)])


__all__ = [
    "support_bounds",
    "quadrant_support_mask",
    "support_rule_free_mask",
    "design_matrix",
    "tensor_values",
    "active_basis_sum_on_quadrant",
    "constrained_coefficient_max",
    "dirichlet_violation",
    "random_graded_lshape_knots",
    "geometric_lshape_knots",
]
