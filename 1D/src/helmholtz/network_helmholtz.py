from __future__ import annotations

"""Positional density network for the helmholtz experiment (1D Helmholtz, two-patch C^0 mesh).

Input encoding phi(omega, xi, axis_id) = (omega, xi_local, axis_id), a 3-feature
tanh MLP (default 2x32). This MIRRORS the singular encoding (beta -> omega) but:
  * drops singular's log(xi + 1/N) feature -- that feature grades toward the x=0
    power singularity, which helmholtz does not have (the field is oscillatory with a
    material interface, not a corner singularity);
  * adds an axis_id in {0,1} so the same weights specialise the two halves
    [0,1/2] and [1/2,1] independently (the left half carries the shorter
    wavelength k1 > k2) -- the per-half idea ported from the 2D lshape network.

N-independent: collocated at the n_half = N/2 cell midpoints of each half, so a
single trained network serves every level (coarse-to-fine continuation).
"""

from dataclasses import dataclass

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from src.helmholtz.mesh_helmholtz import knots_two_patch

Array = jnp.ndarray


@register_pytree_node_class
@dataclass
class PDNParamsH:
    layers: list[tuple[Array, Array]]

    def tree_flatten(self):
        children = tuple(t for layer in self.layers for t in layer)
        return children, (len(self.layers),)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (n,) = aux
        return cls(layers=[(children[2 * i], children[2 * i + 1]) for i in range(n)])


def init_params(seed: int, *, input_dim: int = 3, hidden_dims=(32, 32)) -> PDNParamsH:
    """LeCun/Glorot-tanh init (std = sqrt(1/fan_in)) for a tanh MLP, output dim 1."""
    sizes = (int(input_dim),) + tuple(int(d) for d in hidden_dims) + (1,)
    key = jax.random.PRNGKey(int(seed))
    layers = []
    for i in range(len(sizes) - 1):
        k_w, _k_b, key = jax.random.split(key, 3)
        fan_in, fan_out = sizes[i], sizes[i + 1]
        std = (1.0 / float(fan_in)) ** 0.5
        W = std * jax.random.normal(k_w, (fan_in, fan_out), dtype=DEFAULT_DTYPE)
        b = jnp.zeros((fan_out,), dtype=DEFAULT_DTYPE)
        layers.append((W, b))
    return PDNParamsH(layers=layers)


def _forward(params: PDNParamsH, feats: Array) -> Array:
    h = feats
    for (W, b) in params.layers[:-1]:
        h = jnp.tanh(h @ W + b)
    W_out, b_out = params.layers[-1]
    return (h @ W_out + b_out).reshape(-1)


def _half_logits(params: PDNParamsH, omega: Array, n_half: int, axis_id: float, T: float) -> Array:
    xi = (jnp.arange(int(n_half), dtype=DEFAULT_DTYPE) + 0.5) / float(int(n_half))
    omega_b = jnp.broadcast_to(jnp.asarray(omega, dtype=DEFAULT_DTYPE), xi.shape)
    aid = jnp.full(xi.shape, float(axis_id), dtype=DEFAULT_DTYPE)
    feats = jnp.stack([omega_b, xi, aid], axis=-1)
    z = _forward(params, feats)
    z = z - jnp.mean(z)                                    # gauge fix (per half)
    T_t = jnp.asarray(float(T), dtype=z.dtype)
    return T_t * jnp.tanh(z / T_t)                         # bounded logits in [-T, T]


def knots_from_network(params: PDNParamsH, omega: Array, n_elem: int, degree: int,
                       *, T: float, h_min: float) -> Array:
    """Build the two-patch C^0 knot vector predicted by the network at one omega."""
    n_half = int(n_elem) // 2
    if 2 * n_half != int(n_elem):
        raise ValueError(f"helmholtz needs even N; got {n_elem}.")
    theta_l = _half_logits(params, omega, n_half, axis_id=0.0, T=T)
    theta_r = _half_logits(params, omega, n_half, axis_id=1.0, T=T)
    return knots_two_patch(theta_l, theta_r, int(degree), h_min=h_min)


# --------------------------------------------------------------------------
# Global-mesh variant (helmholtz reparametrization: parameter = contrast c).
# Input encoding (c, xi, side): side = 0 on the left (high-k) layer, 1 on the
# right, so the network can raise element density in the short-wavelength layer.
# Collocates N-1 global cell midpoints (the global softmax has N-1 segments).
# Neutral init (no c-biased warm start); gauge-fix + T*tanh as in the per-half map.
# --------------------------------------------------------------------------
def _global_logits(params: PDNParamsH, c: Array, N: int, T: float) -> Array:
    n_seg = int(N) - 1
    xi = (jnp.arange(n_seg, dtype=DEFAULT_DTYPE) + 0.5) / float(n_seg)
    c_b = jnp.broadcast_to(jnp.asarray(c, dtype=DEFAULT_DTYPE), xi.shape)
    side = jnp.where(xi < 0.5, jnp.asarray(0.0, DEFAULT_DTYPE), jnp.asarray(1.0, DEFAULT_DTYPE))
    feats = jnp.stack([c_b, xi, side], axis=-1)
    z = _forward(params, feats)
    z = z - jnp.mean(z)
    T_t = jnp.asarray(float(T), dtype=z.dtype)
    return T_t * jnp.tanh(z / T_t)


def knots_from_network_global(params: PDNParamsH, c: Array, n_elem: int, degree: int,
                              *, T: float, h_min: float, use_hmax: bool = True) -> Array:
    """Global-mesh knot vector predicted by the network at one contrast c
    (free split across the C^0 interface; Nyquist h_max safeguard)."""
    from src.helmholtz.mesh_helmholtz_global import global_knot_map
    theta = _global_logits(params, c, int(n_elem), T)
    return global_knot_map(theta, int(n_elem), int(degree), c, h_min, use_hmax)


def params_to_flat_dict(params: PDNParamsH) -> dict:
    out = {}
    for i, (W, b) in enumerate(params.layers):
        out[f"W{i}"] = jax.device_get(W)
        out[f"b{i}"] = jax.device_get(b)
    return out


def params_from_flat_dict(d) -> PDNParamsH:
    layers = []
    i = 0
    while f"W{i}" in d.files:
        layers.append((jnp.asarray(d[f"W{i}"], dtype=DEFAULT_DTYPE),
                       jnp.asarray(d[f"b{i}"], dtype=DEFAULT_DTYPE)))
        i += 1
    return PDNParamsH(layers=layers)


__all__ = [
    "PDNParamsH", "init_params", "knots_from_network",
    "knots_from_network_global", "_global_logits",
    "params_to_flat_dict", "params_from_flat_dict",
]
