from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

LOG_EPS_RANGE = (-2.0, -1.0)
B_RANGE = (0.5, 2.0)
T_CAP = 5.0
H_MIN = 1e-7
LEVELS = (8, 16, 32, 64)
EVAL_LEVELS = (8, 16, 32, 64, 128)
N_SAMPLES = 200


def unpack(nu):
    nu = jnp.asarray(nu)
    return 10.0 ** nu[..., 0], nu[..., 1]


def _layer(x, eps, b):
    return jnp.exp((x - 1.0) * b / eps)


def exact(x, eps, b):
    return x * (1.0 - _layer(x, eps, b))


def exact_dx(x, eps, b):
    e = _layer(x, eps, b)
    return 1.0 - e - x * e * b / eps


def exact_dxx(x, eps, b):
    d = eps / b
    return -(2.0 / d + x / d ** 2) * _layer(x, eps, b)


def rhs(x, eps, b):
    return -eps * exact_dxx(x, eps, b) + b * exact_dx(x, eps, b) + exact(x, eps, b)


def sample_parameters(n=N_SAMPLES, seed=0):
    pts = qmc.LatinHypercube(d=2, seed=int(seed)).random(int(n))
    lo = np.array([LOG_EPS_RANGE[0], B_RANGE[0]])
    hi = np.array([LOG_EPS_RANGE[1], B_RANGE[1]])
    return qmc.scale(pts, lo, hi)


def split(nu, seed, train_frac=0.70, val_frac=0.15):
    idx = np.random.default_rng(int(seed)).permutation(len(nu))
    n_tr = int(round(train_frac * len(nu)))
    n_va = int(round(val_frac * len(nu)))
    return nu[idx[:n_tr]], nu[idx[n_tr:n_tr + n_va]], nu[idx[n_tr + n_va:]]


__all__ = ["LOG_EPS_RANGE", "B_RANGE", "T_CAP", "H_MIN", "LEVELS", "EVAL_LEVELS", "N_SAMPLES",
           "unpack", "exact", "exact_dx", "exact_dxx", "rhs", "sample_parameters", "split"]
