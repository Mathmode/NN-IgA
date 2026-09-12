"""Regression tests: confirm corrector and r-adapt minimise η² (not Ritz).

This is the contract introduced by the audit-driven Ritz elimination pass.
The training loss is the residual a-posteriori estimator (η²/η²_uniform);
every other optimisation primitive (online corrector, non-parametric
r-adapt) must minimise the SAME η² objective so trajectories are
methodologically consistent.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.nonparametric.eta_estimator_2d import eta_squared_arctan, eta_squared_lshape
from src.nonparametric.r_adapt import (
    p3_effective_n_elem,
    radapt_p3_one_sample,
    theta_to_knots_arctan,
    theta_to_knots_lshape,
)
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape
from src.parametric.corrector_v2 import corrector_arctan, corrector_lshape


def _eta_p2(knots_x, knots_y, p, n_elem, sigma, q_est):
    res = galerkin_solve_arctan(
        knots_x, knots_y, p, n_elem, n_elem,
        sigma[0], sigma[1], sigma[2],
        q_K=p + 1, q_F=q_est,
    )
    return float(
        eta_squared_arctan(
            res.u_h, knots_x, knots_y, p, n_elem, n_elem,
            sigma[0], sigma[1], sigma[2], q_est=q_est,
        )
    )


def _eta_p3(knots_x, knots_y, p, n_elem, sigma, q_est):
    n_eff = p3_effective_n_elem(int(n_elem), int(p))
    res = galerkin_solve_lshape(
        knots_x, knots_y, p, n_eff, n_eff,
        sigma[0], sigma[1],
        q_K=p + 1, q_F=q_est,
    )
    return float(
        eta_squared_lshape(
            res.u_h, knots_x, knots_y, p, n_eff, n_eff,
            sigma[0], sigma[1], q_est=q_est,
        )
    )


def test_corrector_p2_reduces_eta_squared():
    """corrector_arctan should reduce η² (residual estimator)."""
    p, n_elem = 2, 6
    sigma = jnp.array([8.0, 0.5, 0.5], dtype=jnp.float64)

    # Slightly-perturbed uniform initial mesh — a non-degenerate start.
    rng = np.random.default_rng(0)
    theta_x_init = jnp.asarray(0.15 * rng.standard_normal(n_elem))
    theta_y_init = jnp.asarray(0.15 * rng.standard_normal(n_elem))

    kx0 = theta_to_knots_arctan(theta_x_init, p)
    ky0 = theta_to_knots_arctan(theta_y_init, p)
    eta_before = _eta_p2(kx0, ky0, p, n_elem, sigma, q_est=12)

    result, kx_opt, ky_opt = corrector_arctan(
        theta_x_init, theta_y_init, sigma,
        p=p, n_elem=n_elem, q_K=p + 1, q_F=12, max_iter=20,
    )
    eta_after = _eta_p2(kx_opt, ky_opt, p, n_elem, sigma, q_est=12)

    print(f"\nP2 corrector η²: before={eta_before:.4e}, after={eta_after:.4e}, "
          f"iters={result.n_iter}")
    assert eta_after < eta_before, (
        f"corrector_arctan should reduce η²: before={eta_before:.6e}, after={eta_after:.6e}"
    )


def test_corrector_p3_reduces_eta_squared():
    """corrector_lshape should reduce η² (residual estimator)."""
    p, n_elem = 2, 8
    sigma = jnp.array([0.5, 5.0], dtype=jnp.float64)
    half = n_elem // 2

    rng = np.random.default_rng(0)
    tx_l = jnp.asarray(0.1 * rng.standard_normal(half))
    tx_r = jnp.asarray(0.1 * rng.standard_normal(half))
    ty_l = jnp.asarray(0.1 * rng.standard_normal(half))
    ty_r = jnp.asarray(0.1 * rng.standard_normal(half))

    kx0 = theta_to_knots_lshape(tx_l, tx_r, p)
    ky0 = theta_to_knots_lshape(ty_l, ty_r, p)
    eta_before = _eta_p3(kx0, ky0, p, n_elem, sigma, q_est=2)

    result, kx_opt, ky_opt = corrector_lshape(
        tx_l, tx_r, ty_l, ty_r, sigma,
        p=p, n_elem=n_elem, q_K=p + 1, q_F=2, max_iter=20,
    )
    eta_after = _eta_p3(kx_opt, ky_opt, p, n_elem, sigma, q_est=2)

    print(f"\nP3 corrector η²: before={eta_before:.4e}, after={eta_after:.4e}, "
          f"iters={result.n_iter}")
    assert eta_after < eta_before, (
        f"corrector_lshape should reduce η²: before={eta_before:.6e}, after={eta_after:.6e}"
    )


def test_radapt_p3_one_sample_reduces_eta_squared():
    """radapt_p3_one_sample should reduce η²."""
    p, n_elem = 2, 8
    sigma1, sigma2 = 0.3, 3.0
    half = n_elem // 2

    # Uniform-mesh baseline.
    kx0 = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
    ky0 = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
    eta_before = _eta_p3(kx0, ky0, p, n_elem, jnp.array([sigma1, sigma2]), q_est=2)

    result, kx_opt = radapt_p3_one_sample(
        sigma1, sigma2, n_elem=n_elem, p=p, max_iter=20,
    )
    eta_after = _eta_p3(kx_opt, ky0, p, n_elem, jnp.array([sigma1, sigma2]), q_est=2)

    print(f"\nP3 r-adapt η²: before={eta_before:.4e}, after={eta_after:.4e}, "
          f"iters={result.n_iter}, energy_final={result.energy_final:.4e}")
    assert eta_after < eta_before, (
        f"radapt_p3_one_sample should reduce η²: "
        f"before={eta_before:.6e}, after={eta_after:.6e}"
    )
    # The optimisation result.energy_final IS the η² value now (post-fix).
    # It should match the η² we just measured at the optimised knots.
    assert result.energy_final == pytest.approx(eta_after, rel=1e-10), (
        "radapt result.energy_final must equal η² at the optimised mesh "
        "(the loss function the optimiser saw)."
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
