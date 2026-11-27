from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.nonparametric.alg1_ladder_2d import lshape_problem, mesh_quality
from src.nonparametric.lshape.schedule import policy_at, read_schedule
from src.parametric.continuation import load_checkpoint, save_checkpoint
from src.parametric.positional_density_network_2d import (
    PDN2DParams, cell_midpoints, corner_feature, forward, init_params, knots_p3_from_network_axis,
)
from src.parametric.training import make_batch_loss_lshape
from src.parametric.ritz_loss_2d import make_batch_loss_ritz


SCHEDULE = Path(__file__).resolve().parents[2] / "data_results_v3/lshape_alg1_Tscan/validation_schedule.csv"


@pytest.mark.parametrize("p", [2, 3])
def test_level_feature_policy_and_restore(tmp_path, p):
    schedule = read_schedule(SCHEDULE, p)
    params = init_params(0, sigma_dim=2, hidden_dims=(32, 32), level_feature=True, mesh_schedule=schedule)
    sigma = jnp.array([0.4, 2.0])
    checkpoint = tmp_path / "net.npz"
    save_checkpoint(params, checkpoint)
    restored = load_checkpoint(checkpoint)
    assert params.layers[0][0].shape == restored.layers[0][0].shape == (5, 32)
    for n, t, h in schedule:
        mid = cell_midpoints(n // 2)
        for right in (False, True):
            expected = np.log(abs((np.asarray(mid) + int(right)) / 2 - 0.5) + 1/n)
            np.testing.assert_allclose(corner_feature(mid, n, right), expected, rtol=2e-15)
        z = [forward(params, sigma, mid, axis, corner_feature(mid, n, right))
             for axis, right in [(0.0, False), (0.5, True)]]
        bounded = [t * jnp.tanh((v - v.mean())/t) for v in z]
        theta = jnp.concatenate(bounded * 2)
        x, _, _ = lshape_problem(sigma, mesh_schedule=schedule).knots(theta, n, p)
        kn = knots_p3_from_network_axis(restored, sigma, n, p, 0.0)
        np.testing.assert_allclose(kn, x, rtol=0, atol=4*np.finfo(float).eps)
        assert np.sum(np.asarray(kn) == 0.5) == p
        hvals = np.diff(np.unique(kn))
        assert min(hvals) >= h - 3e-16
        for b in np.split(hvals, 2):
            assert max(b)/min(b) <= np.exp(2*t)
    with pytest.raises(ValueError):
        knots_p3_from_network_axis(restored, sigma, 10, p, 1.0)


def test_actual_network_cross_block_bound():
    n, p = 32, 2
    t, h = policy_at(read_schedule(SCHEDULE, p), 2*n)
    w = np.zeros((4, 32))
    w[2, 0], w[3, 0] = -1024, -2048*(1-2/n)
    b = np.zeros(32)
    b[0] = 1024*(1-1/n)
    middle = np.zeros((32, 32))
    middle[0, 0] = 1
    out = np.zeros((32, 1))
    out[0, 0] = 1e4
    params = PDN2DParams([(jnp.asarray(a), jnp.asarray(bias)) for a, bias in
                          [(w, b), (middle, np.zeros(32)), (out, np.zeros(1))]])
    knots = knots_p3_from_network_axis(params, jnp.ones(2), 2*n, p, 0.0, T=t, h_min=h)
    gamma, _ = mesh_quality(knots, knots, p)
    r = np.exp(2*t)
    lower = h + (0.5 - n*h)/(1+(n-1)*r)
    upper = h + (0.5 - n*h)*r/(r+n-1)
    assert r < gamma <= (upper/lower)*(1+2e-10)


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("kind", ["eta", "ritz"])
def test_loss_gradient_through_interface(p, kind):
    params = init_params(4, sigma_dim=2, hidden_dims=(6, 6), level_feature=True,
                         mesh_schedule=read_schedule(SCHEDULE, p))
    kw = dict(p=p, n_elem=8, q_K=p+1, q_F=4)
    loss = (make_batch_loss_lshape(q_est=4, **kw) if kind == "eta" else
            make_batch_loss_ritz(experiment="lshape", **kw))
    def value(shift):
        layers = params.layers.copy()
        w, b = layers[-1]
        layers[-1] = (w + shift, b)
        return loss(PDN2DParams(layers, True, params.mesh_schedule), jnp.array([[0.3, 1.7]]), jnp.ones(1))
    diff = jax.jit(value)
    grad = float(jax.jit(jax.grad(value))(0.0))
    step = 2e-5
    fd = float((diff(step)-diff(-step))/(2*step))
    assert np.isfinite(grad) and abs(grad) > 1e-9
    assert grad == pytest.approx(fd, rel=3e-7, abs=1e-11)


@pytest.mark.parametrize("p", [2, 3])
def test_element_integrals_against_exact_polynomials(p):
    from scripts.audit_lshape_integration import check
    metrics = check(p, "thin")
    for k, err in metrics.items():
        if k not in ("p", "mesh", "h_min"):
            assert np.isfinite(err) and err <= 2e-14, (k, err)
