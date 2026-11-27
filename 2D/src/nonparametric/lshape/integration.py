from typing import NamedTuple

import jax.numpy as jnp

from common._precision import DEFAULT_DTYPE
from common.quadrature import leggauss


class SpanRule(NamedTuple):
    left: jnp.ndarray
    width: jnp.ndarray
    weights: jnp.ndarray
    basis: tuple
    index: jnp.ndarray


def span_rule(knots, p, order):
    knots = jnp.asarray(knots, dtype=DEFAULT_DTYPE)
    e = jnp.arange(len(knots) - 2 * p - 1)
    s = e + p
    a = knots[s]
    h = knots[s + 1] - a
    z, w = leggauss(order, knots.dtype)
    left_offset = h[:, None] * ((1 + z) / 2)
    right_offset = h[:, None] * ((1 - z) / 2)
    levels = [jnp.ones((*left_offset.shape, 1), dtype=knots.dtype)]
    for degree in range(1, p + 1):
        k = s[:, None] + jnp.arange(degree)
        lo = knots[k + 1 - degree]
        hi = knots[k + 1]
        length = hi - lo
        inv = jnp.where(length > 0, 1 / jnp.where(length > 0, length, 1), 0)
        scaled = levels[-1] * inv[:, None, :]
        right = ((hi - knots[s + 1, None])[:, None, :] + right_offset[..., None]) * scaled
        left = ((a[:, None] - lo)[:, None, :] + left_offset[..., None]) * scaled
        levels.append(jnp.pad(right, ((0, 0), (0, 0), (0, 1)))
                      + jnp.pad(left, ((0, 0), (0, 0), (1, 0))))
    return SpanRule(a, h, h[:, None] * (w / 2), tuple(levels), e[:, None] + jnp.arange(p + 1))


def derivative_scale(knots, p):
    d = knots[p + 1:-1] - knots[1:-(p + 1)]
    return jnp.where(d > 0, p / jnp.where(d > 0, d, 1), 0)


def derivative_coeffs(c, knots, p):
    return jnp.diff(c, axis=0) * derivative_scale(knots, p)[:, None]


def field(c, x, y, dx=0, dy=0):
    bx, by = x.basis[-1 - dx], y.basis[-1 - dy]
    ix = x.index[:, :bx.shape[-1]]
    iy = y.index[:, :by.shape[-1]]
    block = c[ix[:, None, :, None], iy[None, :, None, :]]
    return jnp.einsum("xqi,xyij,yrj->xyqr", bx, block, by)


def domain_weights(x, y, s1, s2):
    left = x.left[:, None] < 0.5
    bottom = y.left[None, :] < 0.5
    sigma = jnp.where(left & bottom, s1, jnp.where(~left & ~bottom, s2, 1.0))
    weights = x.weights[:, None, :, None] * y.weights[None, :, None, :]
    return sigma, weights * (left | ~bottom)[:, :, None, None]


def ritz(c, kx, ky, p, s1, s2, q_K, q_F):
    x, y = span_rule(kx, p, q_K), span_rule(ky, p, q_K)
    sigma, w = domain_weights(x, y, s1, s2)
    ux = field(derivative_coeffs(c, kx, p), x, y, dx=1)
    uy = field(derivative_coeffs(c.T, ky, p).T, x, y, dy=1)
    strain = jnp.sum(w * sigma[:, :, None, None] * (ux * ux + uy * uy))
    xf, yf = (x, y) if q_F == q_K else (span_rule(kx, p, q_F), span_rule(ky, p, q_F))
    _, wf = domain_weights(xf, yf, s1, s2)
    return strain / 2 - jnp.sum(wf * field(c, xf, yf))


def volume(c, kx, ky, p, s1, s2, order):
    x, y = span_rule(kx, p, order), span_rule(ky, p, order)
    sigma, w = domain_weights(x, y, s1, s2)
    def second(c, k):
        return derivative_coeffs(derivative_coeffs(c, k, p), k[1:-1], p - 1)
    lap = field(second(c, kx), x, y, dx=2) + field(second(c.T, ky).T, x, y, dy=2)
    residual = 1 + sigma[:, :, None, None] * lap
    diameter2 = x.width[:, None] ** 2 + y.width[None, :] ** 2
    return jnp.sum((diameter2 / sigma) * jnp.sum(w * residual**2, axis=(2, 3)))


def flux_face(c, normal, tangent, p, minus, plus, upper, order):
    i = (len(normal) - 2 * p - 1 - (p - 1)) // 2 + p - 1
    dc = derivative_coeffs(c, normal, p)
    jump = plus * dc[i] - minus * dc[i - 1]
    t = span_rule(tangent, p, order)
    trace = jnp.einsum("eqi,ei->eq", t.basis[-1], jump[t.index])
    on_face = (t.left >= 0.5) if upper else (t.left < 0.5)
    return jnp.sum(t.weights * trace**2 * (on_face * t.width)[:, None]) / jnp.maximum(minus, plus)


def element_forms(knots, p, q_K, q_F):
    x = span_rule(knots, p, q_K)
    b = x.basis[-1]
    lower = x.basis[-2] * derivative_scale(knots, p)[x.index[:, :p]][:, None, :]
    db = jnp.pad(lower, ((0, 0), (0, 0), (1, 0))) - jnp.pad(lower, ((0, 0), (0, 0), (0, 1)))
    def gram(b):
        return jnp.einsum("eq,eqi,eqj->eij", x.weights, b, b)
    mass, stiff = gram(b), gram(db)
    f = x if q_F == q_K else span_rule(knots, p, q_F)
    load = jnp.einsum("eq,eqi->ei", f.weights, f.basis[-1])
    return x, mass, stiff, load


def axis_forms(knots, p, q_K, q_F):
    x, mass, stiff, load = element_forms(knots, p, q_K, q_F)
    n = len(knots) - p - 1
    def assemble(mask):
        def matrix(local):
            return jnp.zeros((n, n), dtype=knots.dtype).at[x.index[:, :, None], x.index[:, None, :]].add(
                local * mask[:, None, None])
        force = jnp.zeros(n, dtype=knots.dtype).at[x.index].add(load * mask[:, None])
        return matrix(mass), matrix(stiff), force
    return assemble(x.left < 0.5), assemble(x.left >= 0.5)


def assemble(kx, ky, p, s1, s2, q_K, q_F):
    xl, xr = axis_forms(kx, p, q_K, q_F)
    yb, yt = axis_forms(ky, p, q_K, q_F)
    def patch(x, y):
        return jnp.kron(x[1], y[0]) + jnp.kron(x[0], y[1])
    K = s1 * patch(xl, yb) + patch(xl, yt) + s2 * patch(xr, yt)
    F = jnp.outer(xl[2], yb[2] + yt[2]) + jnp.outer(xr[2], yt[2])
    return K, F.ravel()
