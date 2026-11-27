from __future__ import annotations

import time

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.adr import discretization as D
from src.adr.network import logits
from src.adr.problem import H_MIN, T_CAP, unpack
from src.config import CLIP_NORM, EPSILON_DENOM, VAL, WARMUP_EPOCHS


def eta2_of_theta(theta, nu, p):
    eps, b = unpack(nu)
    kn = D.knots_from_theta(theta, p, H_MIN)
    u = D.solve(kn, p, eps, b, D.quad_order(p))
    return D.eta2(kn, p, u, eps, b, D.quad_order(p)), kn, u


def make_uniform_eta2(p, N):
    theta0 = jnp.zeros((int(N),), DEFAULT_DTYPE)
    return jax.jit(jax.vmap(lambda nu: eta2_of_theta(theta0, nu, p)[0]))


def make_batch_loss(p, N):
    def per(params, nu, denom):
        return eta2_of_theta(logits(params, nu, N, T_CAP), nu, p)[0] / (denom + EPSILON_DENOM)

    return lambda params, nus, dens: 0.5 * jnp.mean(jax.vmap(lambda n, d: per(params, n, d))(nus, dens))


def make_val_forward(p, N):
    def one(params, nu, denom):
        e2, kn, u = eta2_of_theta(logits(params, nu, N, T_CAP), nu, p)
        eps, b = unpack(nu)
        return e2 / (denom + EPSILON_DENOM), D.energy_errors(kn, p, u, eps, b)[1]

    return jax.jit(lambda params, nus, dens: jax.vmap(lambda n, d: one(params, n, d))(nus, dens))


def _optimizer(n_batches, max_epochs):
    lr0, lr1 = float(VAL["lr_init"]), float(VAL["lr_end"])
    sched = optax.warmup_exponential_decay_schedule(
        init_value=0.0, peak_value=lr0, warmup_steps=int(WARMUP_EPOCHS) * n_batches,
        transition_steps=max(int(max_epochs) * n_batches, 1), decay_rate=lr1 / lr0, end_value=lr1)
    return optax.chain(optax.clip_by_global_norm(float(CLIP_NORM)), optax.adam(sched))


def train_level(params, p, N, train, val, seed):
    bs, max_ep = int(VAL["batch_size"]), int(VAL["max_epochs"])
    patience, min_ep, tol = int(VAL["patience"]), int(VAL["min_epochs"]), float(VAL["tol"])
    unif = make_uniform_eta2(p, N)
    d_tr, d_va = np.asarray(unif(jnp.asarray(train))), np.asarray(unif(jnp.asarray(val)))
    vg = jax.jit(jax.value_and_grad(make_batch_loss(p, N)))
    val_fwd = make_val_forward(p, N)
    nb = max(len(train) // bs, 1)
    opt = _optimizer(nb, max_ep)
    state = opt.init(params)
    rng = np.random.default_rng(int(seed))
    best, best_params, best_ep, hist, t0 = np.inf, params, -1, [], time.perf_counter()
    for ep in range(max_ep):
        idx, acc = rng.permutation(len(train)), []
        for k in range(0, len(train), bs):
            sel = idx[k:k + bs]
            v, g = vg(params, jnp.asarray(train[sel]), jnp.asarray(d_tr[sel]))
            upd, state = opt.update(g, state, params)
            params = optax.apply_updates(params, upd)
            acc.append(float(v))
        per, err = val_fwd(params, jnp.asarray(val), jnp.asarray(d_va))
        vl = 0.5 * float(jnp.mean(per))
        improved = vl < best * (1.0 - tol)
        if improved:
            best, best_params, best_ep = vl, params, ep
        hist.append(dict(level_N=int(N), epoch=ep, train_loss=float(np.mean(acc)), val_loss=vl,
                         err_E_val_median=float(jnp.median(err)), time_sec=time.perf_counter() - t0))
        if ep + 1 >= min_ep and ep - best_ep >= patience:
            break
    return best_params, hist


def train_continuation(params, p, levels, train, val, seed):
    history, t0 = [], time.perf_counter()
    for N in levels:
        params, h = train_level(params, p, int(N), train, val, int(seed) * 100 + int(N))
        history += h
        print(f"  [adr] p={p} N={N}: epochs={len(h)} best_val={min(r['val_loss'] for r in h):.4e} "
              f"err_E_val={h[-1]['err_E_val_median']:.3e} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return params, history, time.perf_counter() - t0


__all__ = ["eta2_of_theta", "make_uniform_eta2", "make_batch_loss", "make_val_forward",
           "train_level", "train_continuation"]
