"""Positional density network rho_phi(sigma, xi, axis_id) for 2D r-adapt.

Aballay 4.1.2 architecture:
    Input dim   = sigma_dim + 2  (sigma vector + xi + axis_id)
    Hidden      = 2 layers of width 10, tanh activation
    Output      = scalar logit per (sigma, xi, axis_id)
    Init        = LeCun (std = sqrt(1 / fan_in))

The network is N-INDEPENDENT: collocation at N cell midpoints
``xi_i = (i + 0.5) / n_axis`` gives logits ``z_i in R``. A block-softmax
turns logits into sizes summing to a fixed budget (1 for ARCTAN single-patch,
0.5 per half for LSHAPE).

For LSHAPE we collocate at cell midpoints of EACH half-mesh; the two halves
share the same network but advertise different ``axis_id`` (0 = x-left,
1 = x-right) to specialise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Parameter container (registered as a JAX pytree).
# --------------------------------------------------------------------------


@register_pytree_node_class
@dataclass
class PDN2DParams:
    """Flat list of (W, b) tuples for a tanh MLP with output dim 1."""

    layers: list[tuple[Array, Array]]

    def tree_flatten(self):
        children = tuple(t for layer in self.layers for t in layer)
        aux_data = (len(self.layers),)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (n_layers,) = aux_data
        layers = []
        for i in range(n_layers):
            W, b = children[2 * i], children[2 * i + 1]
            layers.append((W, b))
        return cls(layers=layers)


def _layer_sizes(sigma_dim: int, hidden_dims: Tuple[int, ...]) -> Tuple[int, ...]:
    in_dim = int(sigma_dim) + 2     # sigma_dim + xi + axis_id
    return (in_dim,) + tuple(int(d) for d in hidden_dims) + (1,)


def init_params(
    seed: int,
    *,
    sigma_dim: int = 3,
    hidden_dims: Tuple[int, ...] = (10, 10),
) -> PDN2DParams:
    """LeCun initialisation (std = sqrt(1 / fan_in)) for a tanh MLP."""
    sizes = _layer_sizes(int(sigma_dim), tuple(hidden_dims))
    key = jax.random.PRNGKey(int(seed))
    layers: list[tuple[Array, Array]] = []
    for i in range(len(sizes) - 1):
        k_w, k_b, key = jax.random.split(key, 3)
        fan_in, fan_out = sizes[i], sizes[i + 1]
        std = (1.0 / float(fan_in)) ** 0.5
        W = std * jax.random.normal(k_w, (fan_in, fan_out), dtype=DEFAULT_DTYPE)
        b = jnp.zeros((fan_out,), dtype=DEFAULT_DTYPE)
        layers.append((W, b))
    return PDN2DParams(layers=layers)


# --------------------------------------------------------------------------
# Forward pass
# --------------------------------------------------------------------------


def _input_features(sigma: Array, xi: Array, axis_id: float) -> Array:
    """Build feature vector (sigma..., xi, axis_id), broadcast over xi.

    Shapes:
        sigma:    (sigma_dim,)
        xi:       (n_xi,)
        axis_id:  scalar
    Returns:
        feat:     (n_xi, sigma_dim + 2)
    """
    sigma = jnp.asarray(sigma, dtype=DEFAULT_DTYPE).reshape(-1)
    xi = jnp.asarray(xi, dtype=DEFAULT_DTYPE).reshape(-1)
    aid = jnp.full((xi.shape[0],), float(axis_id), dtype=DEFAULT_DTYPE)
    sigma_b = jnp.broadcast_to(sigma[None, :], (xi.shape[0], sigma.shape[0]))
    return jnp.concatenate([sigma_b, xi[:, None], aid[:, None]], axis=-1)


def forward(
    params: PDN2DParams, sigma: Array, xi: Array, axis_id: float
) -> Array:
    """Evaluate rho_phi(sigma, xi, axis_id) -> shape (n_xi,)."""
    x = _input_features(sigma, xi, axis_id)
    h = x
    for (W, b) in params.layers[:-1]:
        h = jnp.tanh(h @ W + b)
    W_out, b_out = params.layers[-1]
    z = (h @ W_out + b_out).reshape(-1)
    return z


# --------------------------------------------------------------------------
# Collocation helpers (cell midpoints)
# --------------------------------------------------------------------------


def cell_midpoints(n: int) -> Array:
    """``(i + 0.5) / n`` for i = 0..n-1."""
    n = int(n)
    return (jnp.arange(n, dtype=DEFAULT_DTYPE) + 0.5) / float(n)


# --------------------------------------------------------------------------
# Logits -> knot vectors (ARCTAN and LSHAPE)
# --------------------------------------------------------------------------


def softmax_jax(theta: Array) -> Array:
    m = jnp.max(theta)
    e = jnp.exp(theta - m)
    return e / jnp.sum(e)


def policy_step(
    z: Array, *, T: float, h_min: float, budget: float = 1.0,
) -> Array:
    """Unified mesh-policy step (decisions 7-C, 7-D, and the cross-1D/2D
    unification): raw logits ``z`` -> centered logits ``z'`` -> bounded
    logits ``q = T·tanh(z'/T)`` -> sizes ``= h_min + (budget - n·h_min)·
    softmax(q)`` summing to ``budget`` with every entry ≥ h_min.

    Mirrors ``1D/src/nonparametric/knots.py:theta_to_sizes`` so 1D and
    2D run the same sequence of operations:

      1. ``z' = z - mean(z)``          # gauge fix (decision §2 of audit)
      2. ``q  = T·tanh(z'/T)``         # bounded logits in [-T, T]
      3. ``sizes = h_min + (budget - n·h_min) · softmax(q)``  # h_min floor

    Parameters
    ----------
    z       : raw logits, shape (n,).
    T       : bounded-logit cap (T_ARCTAN / T_LSHAPE from config; decision 7-C).
    h_min   : minimum cell size (H_MIN_ARCTAN / H_MIN_LSHAPE; decision 7-D).
    budget  : sum of resulting sizes (1.0 for ARCTAN single-patch, 0.5 per
              half for LSHAPE).

    Returns
    -------
    sizes : (n,) array, ``sizes >= h_min`` elementwise, ``sum(sizes) ==
            budget``.
    """
    n = z.shape[-1]
    z_centered = z - jnp.mean(z)
    T_t = jnp.asarray(float(T), dtype=z.dtype)
    h_min_t = jnp.asarray(float(h_min), dtype=z.dtype)
    budget_t = jnp.asarray(float(budget), dtype=z.dtype)
    q = T_t * jnp.tanh(z_centered / T_t)
    scale = budget_t - jnp.asarray(n, dtype=z.dtype) * h_min_t
    return h_min_t + scale * softmax_jax(q)


def knots_p2_from_network(
    params: PDN2DParams, sigma: Array, n_elem: int, p: int,
    *,
    T: float | None = None, h_min: float | None = None,
) -> Array:
    """Build a ARCTAN knot vector on [0, 1] via the unified mesh policy
    (gauge fix + T·tanh + softmax with h_min floor).

    ``T`` and ``h_min`` default to ``T_ARCTAN`` and ``H_MIN_ARCTAN`` from
    ``src.config``; the keyword arguments exist so tests / smoke runs
    can override.
    """
    from src.config import T_ARCTAN, H_MIN_ARCTAN
    if T is None:
        T = float(T_ARCTAN)
    if h_min is None:
        h_min = float(H_MIN_ARCTAN)
    xi = cell_midpoints(n_elem)
    z = forward(params, sigma, xi, axis_id=0.0)
    sizes = policy_step(z, T=T, h_min=h_min, budget=1.0)
    interior = jnp.cumsum(sizes)[:-1]
    zeros = jnp.zeros((p + 1,), dtype=sizes.dtype)
    ones = jnp.ones((p + 1,), dtype=sizes.dtype)
    return jnp.concatenate([zeros, interior, ones])


def knots_p2_from_network_axis(
    params: PDN2DParams, sigma: Array, n_elem: int, p: int, axis_id: float,
    *,
    T: float | None = None, h_min: float | None = None,
) -> Array:
    """Build a ARCTAN knot vector with explicit ``axis_id`` (0 for x, 1 for y).

    Applies the unified mesh policy (decisions 7-C, 7-D / unification):
    gauge centering + ``T·tanh(z/T)`` + softmax with ``h_min`` floor.
    """
    from src.config import T_ARCTAN, H_MIN_ARCTAN
    if T is None:
        T = float(T_ARCTAN)
    if h_min is None:
        h_min = float(H_MIN_ARCTAN)
    xi = cell_midpoints(n_elem)
    z = forward(params, sigma, xi, axis_id=axis_id)
    sizes = policy_step(z, T=T, h_min=h_min, budget=1.0)
    interior = jnp.cumsum(sizes)[:-1]
    zeros = jnp.zeros((p + 1,), dtype=sizes.dtype)
    ones = jnp.ones((p + 1,), dtype=sizes.dtype)
    return jnp.concatenate([zeros, interior, ones])


def knots_p3_from_network_axis(
    params: PDN2DParams, sigma: Array, n_elem: int, p: int, axis_id: float,
    *,
    T: float | None = None, h_min: float | None = None,
) -> Array:
    """Build a LSHAPE knot vector with the material interface node at 0.5
    inserted with **multiplicity p** (C^0 at the interface — physically
    correct for piecewise-σ Poisson; see audit_p3_multipatch.md).

    Two halves: cell midpoints of [0, 0.5] (axis_id) and [0.5, 1]
    (axis_id + 0.5). Each half's policy step uses budget=0.5 so the two
    halves together cover [0, 1].

    The caller MUST pass ``n_elem_eff = p3_effective_n_elem(n_elem, p)
    = n_elem + (p - 1)`` to ``galerkin_solve_lshape`` and
    ``eta_squared_lshape``; the (p - 1) extra elements are zero-length at 0.5
    and contribute nothing to assembly.

    Applies the unified mesh policy (decisions 7-C, 7-D): gauge centering
    + ``T·tanh(z/T)`` + softmax with ``h_min`` floor PER HALF.
    """
    from src.config import (T_LSHAPE, H_MIN_LSHAPE,
                            T_LSHAPE_BY_P, H_MIN_LSHAPE_BY_P)
    # Per-degree override (p=3 calibration) falls back to the scalar default, so
    # empty maps == byte-identical to T_LSHAPE / H_MIN_LSHAPE. An explicit T /
    # h_min kwarg (tests, --T/--h-min flag) still wins over both.
    if T is None:
        T = float(T_LSHAPE_BY_P.get(int(p), T_LSHAPE))
    if h_min is None:
        h_min = float(H_MIN_LSHAPE_BY_P.get(int(p), H_MIN_LSHAPE))
    half = int(n_elem) // 2
    if 2 * half != int(n_elem):
        raise ValueError(f"LSHAPE needs even n_elem; got {n_elem}")
    xi = cell_midpoints(half)
    z_left = forward(params, sigma, xi, axis_id=axis_id)
    z_right = forward(params, sigma, xi, axis_id=axis_id + 0.5)
    sizes_l = policy_step(z_left, T=T, h_min=h_min, budget=0.5)
    sizes_r = policy_step(z_right, T=T, h_min=h_min, budget=0.5)
    interior_l = jnp.cumsum(sizes_l)[:-1]
    interior_r = jnp.asarray(0.5, dtype=sizes_l.dtype) + jnp.cumsum(sizes_r)[:-1]
    fixed = jnp.full((int(p),), 0.5, dtype=sizes_l.dtype)
    interior = jnp.concatenate([interior_l, fixed, interior_r])
    zeros = jnp.zeros((p + 1,), dtype=sizes_l.dtype)
    ones = jnp.ones((p + 1,), dtype=sizes_l.dtype)
    return jnp.concatenate([zeros, interior, ones])


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def params_to_flat_dict(params: PDN2DParams) -> dict:
    out: dict = {}
    for i, (W, b) in enumerate(params.layers):
        out[f"W{i}"] = jax.device_get(W)
        out[f"b{i}"] = jax.device_get(b)
    return out


def params_from_flat_dict(d) -> PDN2DParams:
    layers: list[tuple[Array, Array]] = []
    i = 0
    while f"W{i}" in d.files:
        W = jnp.asarray(d[f"W{i}"], dtype=DEFAULT_DTYPE)
        b = jnp.asarray(d[f"b{i}"], dtype=DEFAULT_DTYPE)
        layers.append((W, b))
        i += 1
    return PDN2DParams(layers=layers)


__all__ = [
    "PDN2DParams",
    "init_params",
    "forward",
    "cell_midpoints",
    "softmax_jax",
    "policy_step",
    "knots_p2_from_network",
    "knots_p2_from_network_axis",
    "knots_p3_from_network_axis",
    "params_to_flat_dict",
    "params_from_flat_dict",
]
