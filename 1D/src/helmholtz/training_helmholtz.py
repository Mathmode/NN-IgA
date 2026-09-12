from __future__ import annotations

"""Coarse-to-fine continuation training for the helmholtz experiment (1D Helmholtz).

Loss (residual-normalized, IDENTICAL normalization to the singular pipeline -- eta^2
divided by the squared analytic H1 seminorm of the exact field):

    L(params; omega, p, N) = eta^2(theta_phi(omega), p, N, omega) / (|u*_omega|^2_{H1} + eps)

The per-omega denominator |u*_omega|^2_{H1} is precomputed on the host from the
analytic transmission solution (it does not depend on the network), so the loss
is a clean parameter-only objective; mini-batches over omega mix gracefully.

Continuation: the N-independent network is trained level-by-level (16->32->64->
128), warm-starting each level from the previous (same weights serve every N).
AdamW, mirroring singular.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
from src.helmholtz.network_helmholtz import params_to_flat_dict as _params_to_flat_dict
import optax

from src.config import EPSILON_DENOM
from src.helmholtz.discretization_helmholtz import quad_order, solve_state_helmholtz
from src.helmholtz.estimator_helmholtz import eta_squared
from src.helmholtz.network_helmholtz import PDNParamsH, init_params, knots_from_network
from src.helmholtz.pde_helmholtz import helmholtz_exact

Array = jnp.ndarray


def h1_denominators(omegas: np.ndarray) -> np.ndarray:
    """Per-omega |u*_omega|^2_{H1} (host, analytic) used as the loss denominator."""
    return np.array([helmholtz_exact(float(w)).h1_seminorm() ** 2 for w in np.asarray(omegas)],
                    dtype=np.float64)


def _make_per_omega_loss(p: int, N: int, T: float, h_min: float):
    nq = quad_order(int(p))

    def per_omega_loss(params: PDNParamsH, omega: Array, denom: Array) -> Array:
        omega_t = jnp.asarray(omega, dtype=DEFAULT_DTYPE)
        knots = knots_from_network(params, omega_t, int(N), int(p), T=float(T), h_min=float(h_min))
        u = solve_state_helmholtz(knots, int(p), omega_t, nq)
        e2 = eta_squared(knots, int(p), u, omega_t, nq)
        return e2 / (jnp.asarray(denom, dtype=DEFAULT_DTYPE) + jnp.asarray(EPSILON_DENOM, dtype=DEFAULT_DTYPE))

    return per_omega_loss


@dataclass
class LevelResult:
    params: PDNParamsH
    final_loss: float
    history: List[dict]


@dataclass
class ContinuationResultH:
    params_final: PDNParamsH
    per_level: List[LevelResult]
    levels_done: List[int]
    elapsed_sec: float


def train_with_continuation(
    *,
    p: int,
    seed: int,
    train_omegas: np.ndarray,
    levels: Tuple[int, ...],
    epochs_per_level: Dict[int, int],
    hidden_dims=(32, 32),
    input_dim: int = 3,
    T: float = 5.0,
    h_min: float = 1e-7,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    batch_size: int = 16,
) -> ContinuationResultH:
    train_omegas = np.asarray(train_omegas, dtype=np.float64).reshape(-1)
    denoms = h1_denominators(train_omegas)

    params = init_params(int(seed), input_dim=int(input_dim), hidden_dims=tuple(hidden_dims))
    optimizer = optax.adamw(float(lr), weight_decay=float(weight_decay))
    opt_state = optimizer.init(params)

    per_level: List[LevelResult] = []
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    n = train_omegas.shape[0]

    for li, N in enumerate(int(x) for x in levels):
        per_omega_loss = _make_per_omega_loss(int(p), int(N), float(T), float(h_min))

        def batch_loss(params, oms, dns):
            return jnp.mean(jax.vmap(lambda o, d: per_omega_loss(params, o, d))(oms, dns))

        loss_and_grad = jax.jit(jax.value_and_grad(batch_loss))
        ep = int(epochs_per_level.get(int(N), 100))
        hist: List[dict] = []
        last = float("nan")
        for epoch in range(ep):
            idx = rng.permutation(n)
            acc, nb = 0.0, 0
            for b0 in range(0, n, int(batch_size)):
                sel = idx[b0 : b0 + int(batch_size)]
                oms = jnp.asarray(train_omegas[sel], dtype=DEFAULT_DTYPE)
                dns = jnp.asarray(denoms[sel], dtype=DEFAULT_DTYPE)
                lv, grads = loss_and_grad(params, oms, dns)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                acc += float(lv); nb += 1
            last = acc / max(nb, 1)
            hist.append({"level_N": int(N), "epoch": int(epoch), "train_loss": float(last),
                         "time_sec": float(time.perf_counter() - t0)})
        per_level.append(LevelResult(params=params, final_loss=float(last), history=hist))
        print(f"  [HELMHOLTZ continuation] level N={N} done: {ep} epochs, final_loss={last:.4e}", flush=True)

    return ContinuationResultH(
        params_final=params, per_level=per_level,
        levels_done=[int(x) for x in levels], elapsed_sec=float(time.perf_counter() - t0),
    )


# ==========================================================================
# helmholtz: contrast-reparametrized training with a global mesh + contrast-continuation
# ==========================================================================
def contrast_denominators(cs: np.ndarray) -> np.ndarray:
    """DEPRECATED for training (eq-26 release: the loss divides by
    η²(θ_unif^{(N)}; c) — see ``make_uniform_eta2_c``). Kept as the analytic
    |u*_c|²_{H¹} metric denominator for evaluation use."""
    from src.helmholtz.mesh_helmholtz_global import ExactContrast
    return np.array([ExactContrast(float(c)).h1_seminorm() ** 2 for c in np.asarray(cs)],
                    dtype=np.float64)


def make_uniform_eta2_c(p: int, N: int, h_min: float):
    """Vmapped η²(θ_unif^{(N)}; c) — the eq-26 loss denominator at level N.

    θ_unif = ``uniform_logits`` with NO h_max water-fill: the PLAIN uniform
    mesh of manuscript eq. (26) (the water-fill would pre-adapt it)."""
    from src.helmholtz.mesh_helmholtz_global import (
        solve_u, eta2, quad_order, K2, SIGMA1, global_knot_map, uniform_logits)
    nq = quad_order(int(p))
    theta_u = uniform_logits(int(N))

    def one(c):
        c_t = jnp.asarray(c, dtype=DEFAULT_DTYPE)
        rho1 = (c_t * jnp.asarray(K2, DEFAULT_DTYPE)) ** 2 * jnp.asarray(SIGMA1, DEFAULT_DTYPE)
        knots = global_knot_map(theta_u, int(N), int(p), c_t, float(h_min),
                                use_hmax=False)
        u = solve_u(knots, int(p), rho1, nq)
        return eta2(knots, int(p), u, rho1, nq)

    return jax.jit(jax.vmap(one))


def _level_denoms(train_cs, val_cs, N, p, h_min, loss_denom, val_protocol):
    """Eq-26 loss denominators (train AND val) at level N for the chosen mode.

    The denominator is the ONLY degree of freedom of the denominator study; every
    other part of training is identical across modes. Returns ``(denoms, val_dn)``
    as numpy arrays (``val_dn`` is None when not in the val protocol).

      uniform : eta^2(theta_unif^{(N)}; c)      -- eq-26 production; per level; NO clip
      exact   : |u*_c|^2_{H1} (analytic)        -- N-independent; NO clip
      robust  : uniform, clipped to the per-level 90th percentile of the TRAIN
                denoms; the SAME threshold (taken from the TRAIN distribution) clips
                val, so the val-eta^2 early-stop monitor stays calibrated to train.

    With ``loss_denom="uniform"`` (the default) this is bit-identical to the eq-26
    release: ``make_uniform_eta2_c(p, N, h_min)`` evaluated with no clipping.
    """
    if str(loss_denom) == "exact":
        # Analytic |u*_c|^2_{H1}; independent of N (recomputed each level for code
        # simplicity, returns the same vector every level).
        denoms = contrast_denominators(train_cs)
        val_dn = contrast_denominators(val_cs) if val_protocol else None
        return denoms, val_dn
    # uniform and robust both start from the per-level uniform estimator.
    unif_fn = make_uniform_eta2_c(int(p), int(N), float(h_min))
    denoms = np.asarray(unif_fn(jnp.asarray(train_cs, dtype=DEFAULT_DTYPE)))
    val_dn = (np.asarray(unif_fn(jnp.asarray(val_cs, dtype=DEFAULT_DTYPE)))
              if val_protocol else None)
    if str(loss_denom) == "robust":
        # p90 threshold from the TRAIN distribution, recomputed at THIS level;
        # clips BOTH train and val so they share the (train-derived) umbral.
        thr = float(np.percentile(denoms, 90))
        denoms = np.minimum(denoms, thr)
        if val_dn is not None:
            val_dn = np.minimum(val_dn, thr)
    return denoms, val_dn


def _make_per_c_loss(p: int, N: int, T: float, h_min: float, use_hmax: bool):
    """Eq-26 per-sample term: η²(θ_φ(c); c) / (η²_unif^{(N)}(c) + ε).
    The batch wrapper applies the ½·mean (the 1/(2n) of eq. 26)."""
    from src.helmholtz.mesh_helmholtz_global import solve_u, eta2, quad_order, K2, SIGMA1
    from src.helmholtz.network_helmholtz import knots_from_network_global
    nq = quad_order(int(p))

    def per_c_loss(params, c, denom):
        c_t = jnp.asarray(c, dtype=DEFAULT_DTYPE)
        rho1 = (c_t * jnp.asarray(K2, DEFAULT_DTYPE)) ** 2 * jnp.asarray(SIGMA1, DEFAULT_DTYPE)
        knots = knots_from_network_global(params, c_t, int(N), int(p), T=float(T),
                                          h_min=float(h_min), use_hmax=bool(use_hmax))
        u = solve_u(knots, int(p), rho1, nq)
        e2 = eta2(knots, int(p), u, rho1, nq)
        return e2 / (jnp.asarray(denom, DEFAULT_DTYPE) + jnp.asarray(EPSILON_DENOM, DEFAULT_DTYPE))

    return per_c_loss


def _make_val_forward_c(p: int, N: int, T: float, h_min: float, use_hmax: bool):
    """Combined val forward (one solve per c): eq-26 term + (u, knots) stacks
    for the host-side H¹-error monitor (``mesh_helmholtz_global.h1_rel``)."""
    from src.helmholtz.mesh_helmholtz_global import solve_u, eta2, quad_order, K2, SIGMA1
    from src.helmholtz.network_helmholtz import knots_from_network_global
    nq = quad_order(int(p))

    def one(params, c, denom):
        c_t = jnp.asarray(c, dtype=DEFAULT_DTYPE)
        rho1 = (c_t * jnp.asarray(K2, DEFAULT_DTYPE)) ** 2 * jnp.asarray(SIGMA1, DEFAULT_DTYPE)
        knots = knots_from_network_global(params, c_t, int(N), int(p), T=float(T),
                                          h_min=float(h_min), use_hmax=bool(use_hmax))
        u = solve_u(knots, int(p), rho1, nq)
        e2 = eta2(knots, int(p), u, rho1, nq)
        per = e2 / (jnp.asarray(denom, DEFAULT_DTYPE) + jnp.asarray(EPSILON_DENOM, DEFAULT_DTYPE))
        return per, u, knots

    return jax.jit(lambda params, cs, dns: jax.vmap(
        lambda c, d: one(params, c, d))(cs, dns))


def train_with_continuation_contrast(
    *, p: int, seed: int, train_cs: np.ndarray,
    levels: Tuple[int, ...], epochs_per_level: Dict[int, int],
    warm_epochs: int = 30, c_high_frac: float = 0.34,
    hidden_dims=(32, 32), input_dim: int = 3, T: float = 5.0, h_min: float = 1e-7,
    lr: float = 1e-3, lr_explore: float = 3e-3, weight_decay: float = 1e-6,
    batch_size: int = 16, use_hmax: bool = True,
    val_cs: np.ndarray = None, protocol: str = "fixed",
    init_mode: str = "warmstart",
    loss_denom: str = "uniform",
    ckpt_dir=None,
) -> ContinuationResultH:
    """Coarse-to-fine N continuation + CONTRAST-CONTINUATION anti-stall curriculum.

    The diagnostics showed uniform is a flat minimum at fine N (the optimizer
    stalls). Cure: at the COARSEST level first do a `warm_epochs` exploration
    phase at higher LR on the HIGH-CONTRAST subset (top `c_high_frac` by c) -- the
    strong-signal / under-resolved end where the residual gradient clearly favours
    skewing elements into the high-k layer -- THEN train the full c range; deeper
    levels train the full range from the warmed weights. Neutral (uniform) init;
    no c-biased mesh is injected.

    protocol="val" (unified protocol, config.VAL): the full-range phase B at each
    level monitors the held-out validation η² (mean normalized loss over ``val_cs``
    at the current N) every epoch and EARLY-STOPS on it (relative-1% / patience-40
    / min-40 / cap-400), with phase B's LR a two-stage 1e-2 -> 1e-4 exponential
    schedule; the best-val params are checkpointed and warm-start the next level.
    The high-c warm phase A is kept as the fixed anti-stall curriculum (it is
    exploration, not convergence). protocol="fixed" (default) is byte-identical
    to the legacy fixed-epoch recipe.
    """
    train_cs = np.sort(np.asarray(train_cs, dtype=np.float64).reshape(-1))
    n = train_cs.shape[0]
    n_high = max(1, int(round(float(c_high_frac) * n)))
    high_idx = np.argsort(train_cs)[-n_high:]                      # the top-c (high-contrast) subset

    val_protocol = (str(protocol) == "val") and (val_cs is not None)
    if val_protocol:
        from src.config import VAL
        val_cs = np.sort(np.asarray(val_cs, dtype=np.float64).reshape(-1))
        es_patience = int(VAL["patience"]); es_min = int(VAL["min_epochs"]); es_tol = float(VAL["tol"])
        lr_init = float(VAL["lr_init"]); lr_end = float(VAL["lr_end"])

    params = init_params(int(seed), input_dim=int(input_dim), hidden_dims=tuple(hidden_dims))
    per_level: List[LevelResult] = []
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    levels = tuple(int(x) for x in levels)

    for li, N in enumerate(levels):
        # init_mode="uniform" (study variant): discard the warm-started weights
        # and re-initialise θ0 at every level after the first (no transfer);
        # "warmstart" (default) inherits level_best_params from the prior level.
        if str(init_mode) == "uniform" and li > 0:
            params = init_params(int(seed), input_dim=int(input_dim),
                                 hidden_dims=tuple(hidden_dims))
            print(f"  [helmholtz continuation] init_mode=uniform: re-initialised θ0 at "
                  f"level N={N} (no warm-start transfer)", flush=True)
        per_c_loss = _make_per_c_loss(int(p), int(N), float(T), float(h_min), bool(use_hmax))

        # Eq-26 denominators at THIS level (train AND val), recomputed per level.
        # loss_denom selects the denominator (the ONLY degree of freedom of the
        # denominator study); "uniform" (default) is bit-identical to eq-26.
        denoms, val_dn = _level_denoms(train_cs, val_cs, int(N), int(p), float(h_min),
                                       str(loss_denom), bool(val_protocol))
        print(f"  [helmholtz {loss_denom}] denom at N={int(N)}: train median="
              f"{float(np.median(denoms)):.4e}" +
              (f", val median={float(np.median(val_dn)):.4e}" if val_protocol else "")
              + " (recomputed per level)", flush=True)

        def batch_loss(params, cs, dns):
            per = jax.vmap(lambda c, d: per_c_loss(params, c, d))(cs, dns)
            return 0.5 * jnp.mean(per)          # the 1/(2n) of eq. (26)
        loss_and_grad = jax.jit(jax.value_and_grad(batch_loss))
        # Combined val forward: ONE solve per c -> eq-26 val loss + (u, knots)
        # for the host-side h1_rel_val_median monitor.
        val_forward = (_make_val_forward_c(int(p), int(N), float(T), float(h_min),
                                           bool(use_hmax)) if val_protocol else None)
        # val_dn comes from _level_denoms above (clipped to the train p90 for
        # robust); do NOT recompute it here or robust's val monitor would be unclipped.
        val_cs_j = jnp.asarray(val_cs) if val_protocol else None
        val_dn_j = jnp.asarray(val_dn) if val_protocol else None

        def run_phase(params, opt, opt_state, idx_pool, n_epochs, tag, early_stop):
            """Train one phase. Returns params, opt_state, history, last_train_loss,
            best_params, best_monitored, best_epoch, stop_reason, epochs_used. The
            monitored quantity is the val η² (val protocol) else the train loss."""
            hist = []; last = float("nan")
            best_params = params; best_mon = float("inf"); best_epoch = -1
            stop_reason = "cap"; used = 0
            for epoch in range(int(n_epochs)):
                idx = rng.permutation(idx_pool)
                acc, nb = 0.0, 0
                for b0 in range(0, idx.shape[0], int(batch_size)):
                    sel = idx[b0:b0 + int(batch_size)]
                    lv, g = loss_and_grad(params, jnp.asarray(train_cs[sel]), jnp.asarray(denoms[sel]))
                    upd, opt_state = opt.update(g, opt_state, params)
                    params = optax.apply_updates(params, upd)
                    acc += float(lv); nb += 1
                last = acc / max(nb, 1)
                # Combined val pass: one solve per c gives the eq-26 val loss
                # AND (u, knots) for the H¹-error monitor (host h1_rel).
                ve = float("nan"); h1_med = float("nan")
                if val_protocol:
                    per_v, u_v, kn_v = val_forward(params, val_cs_j, val_dn_j)
                    ve = 0.5 * float(jnp.mean(per_v))    # == batch_loss (eq. 26)
                    from src.helmholtz.mesh_helmholtz_global import ExactContrast, h1_rel
                    u_np = np.asarray(u_v); kn_np = np.asarray(kn_v)
                    h1s = [float(h1_rel(kn_np[i], int(p), u_np[i],
                                        ExactContrast(float(val_cs[i]))))
                           for i in range(u_np.shape[0])]
                    h1s = [v for v in h1s if np.isfinite(v)]
                    h1_med = float(np.median(h1s)) if h1s else float("nan")
                hist.append({"level_N": int(N), "phase": tag, "epoch": int(epoch),
                             "train_loss": float(last), "val_eta2": float(ve),
                             "h1_rel_val_median": float(h1_med),
                             "time_sec": float(time.perf_counter() - t0)})
                used = epoch + 1
                mon = ve if val_protocol else last
                tol = es_tol if val_protocol else 0.0
                if mon < best_mon * (1.0 - tol):
                    best_mon = mon; best_epoch = epoch; best_params = params
                if early_stop and val_protocol and (epoch + 1) >= es_min and (epoch - best_epoch) >= es_patience:
                    stop_reason = "plateau"; break
            return params, opt_state, hist, last, best_params, best_mon, best_epoch, stop_reason, used

        hist_all: List[dict] = []
        level_best_params = params; level_best_mon = float("inf")
        if li == 0:
            # Phase A: high-contrast exploration at higher LR (anti-stall) — fixed.
            optA = optax.adamw(float(lr_explore), weight_decay=float(weight_decay))
            (params, _sA, hA, _la, bpA, bmA, _beA, _srA, _uA) = run_phase(
                params, optA, optA.init(params), high_idx, warm_epochs, "warm_highc", early_stop=False)
            hist_all += hA
            if bmA < level_best_mon:
                level_best_params, level_best_mon = bpA, bmA
        # Phase B: full c range. Val protocol: 1e-2->1e-4 schedule + val early stop.
        ep = int(epochs_per_level.get(int(N), 100))
        if val_protocol:
            from src.config import CLIP_NORM, WARMUP_EPOCHS
            nb_full = max(n // int(batch_size), 1)
            warmup_steps = max(int(WARMUP_EPOCHS) * nb_full, 0)
            # Eq-26 release optimizer: 0 -> lr_init warmup + exponential decay,
            # with global-norm gradient clipping (config.CLIP_NORM).
            scheduleB = optax.warmup_exponential_decay_schedule(
                init_value=0.0, peak_value=lr_init, warmup_steps=warmup_steps,
                transition_steps=max(int(ep) * nb_full, 1),
                decay_rate=lr_end / lr_init, end_value=lr_end)
            optB = optax.chain(
                optax.clip_by_global_norm(float(CLIP_NORM)),
                optax.adam(learning_rate=scheduleB),
            )
            print(f"  [optim] clip_norm={float(CLIP_NORM):g} warmup={int(WARMUP_EPOCHS)}ep "
                  f"({warmup_steps} steps): lr(0)={float(scheduleB(0)):.2e} -> "
                  f"peak={lr_init:g} -> end={lr_end:g}", flush=True)
        else:
            optB = optax.adamw(float(lr), weight_decay=float(weight_decay))
        (params, _sB, hB, last, bpB, bmB, _beB, stop_reason, used) = run_phase(
            params, optB, optB.init(params), np.arange(n), ep, "full", early_stop=val_protocol)
        hist_all += hB
        if bmB < level_best_mon:
            level_best_params, level_best_mon = bpB, bmB
        if val_protocol:
            params = level_best_params       # best-val checkpoint + warm-start next level
            print(f"  [helmholtz val] N={N} stop_reason={stop_reason} epochs_used(B)={used}/{ep} "
                  f"best_val_eta2={level_best_mon:.4e}", flush=True)
        per_level.append(LevelResult(params=params, final_loss=float(last), history=hist_all))
        if ckpt_dir is not None:
            np.savez(ckpt_dir / f"checkpoint_N{N}.npz", **_params_to_flat_dict(params))
        print(f"  [helmholtz continuation] level N={N} done (warm={warm_epochs if li==0 else 0}+{ep}); "
              f"final_loss={last:.4e}", flush=True)

    return ContinuationResultH(params_final=params, per_level=per_level,
                               levels_done=list(levels), elapsed_sec=float(time.perf_counter() - t0))


__all__ = ["h1_denominators", "train_with_continuation", "ContinuationResultH", "LevelResult",
           "contrast_denominators", "train_with_continuation_contrast"]
