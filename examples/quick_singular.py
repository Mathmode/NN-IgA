"""Reduced PDN training -> knot map -> Galerkin state -> residual check."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "1D")]
import json
import numpy as np
import jax.numpy as jnp
from src.config import H_MIN_SCHEDULE, BETA_RANGE
from src.parametric.positional_density_network import init_params, logits_at_level
from src.parametric.training import train_at_level
from src.nonparametric.discretization import solve_state, stiffness_quadrature_order
from src.nonparametric.quadrature_analytic import reduced_loss

p, N, beta = 2, 4, 1.7
result = train_at_level(init_params(seed=0), p=p, N=N,
    train_betas=np.linspace(*BETA_RANGE, 4),
    val_betas=np.asarray([1.65, 1.85]), seed=0, protocol="val", epochs_override=3)
assert np.isfinite(result.best_val_loss), "Training validation loss must be finite"
theta = logits_at_level(result.params_best, beta=beta, p=p, N=N)
h_min = float(H_MIN_SCHEDULE[p][N])
q = stiffness_quadrature_order(p)
u, cache = solve_state(theta, p, q, h_min, beta=beta)
loss = float(reduced_loss(theta, p, q, h_min, beta=beta))
assert np.isfinite(loss) and np.all(np.isfinite(np.asarray(u)))
assert np.all(np.diff(np.asarray(cache.knots)) >= 0)
print(json.dumps({"degree": p, "elements": N, "seed": 0, "epochs": 3,
    "state": np.asarray(u).tolist(), "loss": loss}, sort_keys=True))
