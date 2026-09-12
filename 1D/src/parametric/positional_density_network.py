"""Positional density network rho_phi(beta, xi) -> bounded logits z(beta, xi).

This module implements the positional, N-independent density network used to
parametrize the cell sizes h_i(beta) of the IGA mesh on [0, 1].

Architecture
------------
- Input encoding: feature vector  phi(beta, xi, N) = (beta, xi, log(xi + 1/N)).
- Hidden layers:  two tanh layers of width 32 each (per src.config.ARCH).
- Output:         scalar pre-saturation logit z_pre per (beta, xi).

Collocation
-----------
For a target level N the network is collocated at the cell midpoints in
parameter space:
    xi_i = (i + 0.5) / N,  i = 0, ..., N - 1.
This produces a vector z_pre in R^N. We apply two transformations in sequence:

1. Gauge fix:        z_centered = z_pre - mean(z_pre)             (sum = 0)
2. Saturation:       theta(beta, N) = T * tanh(z_centered / T)    (in [-T, T])

The saturation cap T comes from src.config.T_SCHEDULE[p][N] and is
*per (p, N)*. This bounded representation feeds directly into the existing
non-parametric primitive `theta_to_sizes` (softmax map) via the standard
solver path.

The network is N-INDEPENDENT: the same rho_phi can be evaluated at any N
without retraining. This is what enables coarse-to-fine continuation
(see continuation.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from src.config import ARCH, T_SCHEDULE

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Parameter container (registered as a JAX pytree so optax can traverse it)
# -----------------------------------------------------------------------------


@register_pytree_node_class
@dataclass
class PDNParams:
    """Flat list of (W, b) tuples for a tanh MLP with output dim 1."""

    layers: list[tuple[Array, Array]]

    def tree_flatten(self):
        # Children: a flat tuple of arrays in fixed order [W0, b0, W1, b1, ...]
        children = tuple(t for layer in self.layers for t in layer)
        aux_data = (len(self.layers),)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        n_layers, = aux_data
        layers = []
        for i in range(n_layers):
            W, b = children[2 * i], children[2 * i + 1]
            layers.append((W, b))
        return cls(layers=layers)


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------


def _layer_sizes() -> Tuple[int, ...]:
    hidden = tuple(int(d) for d in ARCH["hidden_dims"])  # type: ignore[index]
    in_dim = int(ARCH["input_dim"])  # type: ignore[index]
    return (in_dim,) + hidden + (1,)


def init_params(seed: int) -> PDNParams:
    """Glorot-tanh initialization for a 2x32 tanh MLP with scalar output."""
    sizes = _layer_sizes()
    key = jax.random.PRNGKey(int(seed))
    layers: list[tuple[Array, Array]] = []
    for i in range(len(sizes) - 1):
        k_w, k_b, key = jax.random.split(key, 3)
        fan_in, fan_out = sizes[i], sizes[i + 1]
        # Xavier/Glorot for tanh:  std = sqrt(1 / fan_in)
        std = (1.0 / float(fan_in)) ** 0.5
        W = std * jax.random.normal(k_w, (fan_in, fan_out), dtype=DEFAULT_DTYPE)
        b = jnp.zeros((fan_out,), dtype=DEFAULT_DTYPE)
        layers.append((W, b))
    return PDNParams(layers=layers)


# -----------------------------------------------------------------------------
# Forward pass
# -----------------------------------------------------------------------------


def _input_features(beta: Array, xi: Array, N: int) -> Array:
    """Build feature vector phi(beta, xi, N) = (beta, xi, log(xi + 1/N)).

    All shapes are broadcast to (..., 3).
    """
    eps = jnp.asarray(1.0 / float(int(N)), dtype=xi.dtype)
    feat = jnp.stack(
        [
            jnp.broadcast_to(beta, xi.shape),
            xi,
            jnp.log(xi + eps),
        ],
        axis=-1,
    )
    return feat.astype(DEFAULT_DTYPE)


def forward_scalar(params: PDNParams, beta: float, xi: Array, N: int) -> Array:
    """Evaluate the network at one beta and a vector of xi values.

    Returns
    -------
    z_pre : Array, shape (xi.shape[0],)
        Pre-saturation, pre-gauge-fix scalar output.
    """
    beta_t = jnp.asarray(beta, dtype=DEFAULT_DTYPE)
    xi_t = jnp.asarray(xi, dtype=DEFAULT_DTYPE).reshape(-1)
    x = _input_features(beta_t, xi_t, int(N))                # (n_xi, 3)
    h = x
    for (W, b) in params.layers[:-1]:
        h = jnp.tanh(h @ W + b)
    W_out, b_out = params.layers[-1]
    z = (h @ W_out + b_out).reshape(-1)                      # (n_xi,)
    return z


# -----------------------------------------------------------------------------
# Collocation -> bounded logits
# -----------------------------------------------------------------------------


def cell_xi(N: int) -> Array:
    """Cell-midpoint parameter coordinates xi_i = (i + 0.5)/N for i = 0..N-1."""
    return (jnp.arange(int(N), dtype=DEFAULT_DTYPE) + 0.5) / float(int(N))


def gauge_fix(z: Array) -> Array:
    """Subtract the mean: z' = z - mean(z). Always sums to zero."""
    return z - jnp.mean(z)


def saturate(z: Array, T: float) -> Array:
    """Bounded logits in [-T, T] via T * tanh(z/T)."""
    T_t = jnp.asarray(float(T), dtype=z.dtype)
    return T_t * jnp.tanh(z / T_t)


def logits_at_level(
    params: PDNParams,
    beta: float,
    *,
    p: int,
    N: int,
) -> Array:
    """Full collocation pipeline at one (beta, p, N).

    Returns the bounded logits theta in [-T, T] of shape (N,) — ready to
    feed into `theta_to_sizes` from the non-parametric primitives.
    """
    xi = cell_xi(int(N))
    z_pre = forward_scalar(params, beta, xi, int(N))
    z_gauge = gauge_fix(z_pre)
    T = float(T_SCHEDULE[int(p)][int(N)])
    return saturate(z_gauge, T)


def logits_at_xi(
    params: PDNParams,
    beta: float,
    xi: Array,
    *,
    p: int,
    N_for_T: int,
    N_for_features: int | None = None,
) -> Array:
    """Evaluate logits at arbitrary xi (used by tests for shared-position checks)."""
    if N_for_features is None:
        N_for_features = int(N_for_T)
    z_pre = forward_scalar(params, beta, xi, int(N_for_features))
    z_gauge = gauge_fix(z_pre)
    T = float(T_SCHEDULE[int(p)][int(N_for_T)])
    return saturate(z_gauge, T)


# -----------------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------------


def params_to_flat_dict(params: PDNParams) -> dict:
    """Pack params into a flat dict for np.savez."""
    out: dict = {}
    for i, (W, b) in enumerate(params.layers):
        out[f"W{i}"] = jax.device_get(W)
        out[f"b{i}"] = jax.device_get(b)
    return out


def params_from_flat_dict(d) -> PDNParams:
    """Reconstruct PDNParams from a dict-like np.load result."""
    layers: list[tuple[Array, Array]] = []
    i = 0
    while f"W{i}" in d.files:
        W = jnp.asarray(d[f"W{i}"], dtype=DEFAULT_DTYPE)
        b = jnp.asarray(d[f"b{i}"], dtype=DEFAULT_DTYPE)
        layers.append((W, b))
        i += 1
    return PDNParams(layers=layers)


__all__ = [
    "PDNParams",
    "init_params",
    "forward_scalar",
    "cell_xi",
    "gauge_fix",
    "saturate",
    "logits_at_level",
    "logits_at_xi",
    "params_to_flat_dict",
    "params_from_flat_dict",
]
