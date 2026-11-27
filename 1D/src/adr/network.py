from __future__ import annotations

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from src.adr.problem import B_RANGE, LOG_EPS_RANGE


def init_params(seed, hidden=(32, 32), n_in=3):
    sizes = (int(n_in),) + tuple(int(h) for h in hidden) + (1,)
    key = jax.random.PRNGKey(int(seed))
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (fan_in, fan_out), dtype=DEFAULT_DTYPE) / np.sqrt(fan_in)
        params.append((W, jnp.zeros((fan_out,), DEFAULT_DTYPE)))
    return params


def _normalize(lo_hi, v):
    lo, hi = lo_hi
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def features(nu, N):
    xi = (jnp.arange(int(N), dtype=DEFAULT_DTYPE) + 0.5) / int(N)
    n_eps = jnp.broadcast_to(_normalize(LOG_EPS_RANGE, nu[0]), xi.shape)
    n_b = jnp.broadcast_to(_normalize(B_RANGE, nu[1]), xi.shape)
    return jnp.stack([n_eps, n_b, 2.0 * xi - 1.0], axis=-1)


def forward(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return (x @ W + b).reshape(-1)


def logits(params, nu, N, T):
    z = forward(params, features(jnp.asarray(nu, DEFAULT_DTYPE), N))
    z = z - jnp.mean(z)
    return T * jnp.tanh(z / T)


def to_flat(params):
    out = {}
    for i, (W, b) in enumerate(params):
        out[f"W{i}"], out[f"b{i}"] = np.asarray(W), np.asarray(b)
    return out


def from_flat(d):
    n = sum(1 for k in d.keys() if k.startswith("W"))
    return [(jnp.asarray(d[f"W{i}"], DEFAULT_DTYPE), jnp.asarray(d[f"b{i}"], DEFAULT_DTYPE)) for i in range(n)]


__all__ = ["init_params", "features", "forward", "logits", "to_flat", "from_flat"]
