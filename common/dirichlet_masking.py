"""JIT- and AD-compatible Dirichlet masking via sign-based indicator functions.

Used by lshape (L-shape) to enforce ``u = 0`` on the removed quadrant
``(0.5, 1) x (0, 0.5)`` while keeping a tensor-product mesh on the full
unit square. The sign-trick yields hard 0/1 masks while remaining
differentiable through ``jax.grad`` (the gradient on the indicator's
support is zero, which is exactly what we want for r-adaptivity).

Notes
-----
``jax.numpy.sign`` is well-defined: sign(t) = -1 if t<0, 0 if t==0,
+1 if t>0. We use ``0.5 * (1 + sign(x - threshold))`` so the value at
``x == threshold`` evaluates to ``0.5``. In practice we always evaluate
at nodal coordinates that lie strictly above or below 0.5 (the fixed
node at 0.5 is the boundary itself), so the 0.5 mass at the boundary
is irrelevant. If the user must include a node exactly at 0.5, treat
that node as Dirichlet=0 by separate logic, not by this mask.
"""
from __future__ import annotations

from common._precision import ensure_double_precision

ensure_double_precision()

import jax.numpy as jnp

Array = jnp.ndarray


def sign_indicator(x: Array, threshold) -> Array:
    """Strict-inequality indicator via sign: 0.5 * (1 + sign(x - threshold)).

    Returns
        0.0 when x <  threshold,
        0.5 when x == threshold (boundary; see `step_ge` for closed form),
        1.0 when x >  threshold.
    """
    x = jnp.asarray(x)
    t = jnp.asarray(threshold, dtype=x.dtype)
    return 0.5 * (jnp.asarray(1.0, dtype=x.dtype) + jnp.sign(x - t))


def step_ge(x: Array, threshold) -> Array:
    """Closed-from-the-right Heaviside: 1 if x >= threshold else 0.

    Uses ``jnp.heaviside`` (JIT- and AD-compatible). Convention here is
    ``heaviside(0, 1.0) = 1.0``, so the boundary is captured. ``threshold``
    may be a Python scalar or a JAX array.
    """
    x = jnp.asarray(x)
    t = jnp.asarray(threshold, dtype=x.dtype)
    return jnp.heaviside(x - t, jnp.asarray(1.0, dtype=x.dtype))


def step_le(x: Array, threshold) -> Array:
    """Closed-from-the-left Heaviside: 1 if x <= threshold else 0."""
    x = jnp.asarray(x)
    t = jnp.asarray(threshold, dtype=x.dtype)
    return jnp.heaviside(t - x, jnp.asarray(1.0, dtype=x.dtype))


def lshape_removed_quadrant_mask(x_coord: Array, y_coord: Array) -> Array:
    """Mask = 1 inside the CLOSED removed quadrant (x >= 0.5 AND y <= 0.5).

    The closure is taken intentionally: nodes lying on the two internal
    interface edges (x = 0.5 with y <= 0.5, and x >= 0.5 with y = 0.5)
    sit on the L-shape boundary where u = 0. The reentrant corner
    (0.5, 0.5) is included as well, since u = 0 there too.

    Inputs are broadcastable arrays (e.g., a meshgrid of nodal coords).
    """
    in_x = step_ge(x_coord, 0.5)   # 1 if x >= 0.5
    in_y = step_le(y_coord, 0.5)   # 1 if y <= 0.5
    return in_x * in_y


def lshape_dirichlet_node_mask(x_coord: Array, y_coord: Array) -> Array:
    """Mask = 1 for Dirichlet-tagged nodes, 0 otherwise.

    Tagged nodes (u = 0):
        * removed quadrant: x > 0.5 AND y < 0.5
        * external boundary of the unit square: x in {0, 1} OR y in {0, 1}

    The full set of constrained nodes = union of these two.
    """
    x_coord = jnp.asarray(x_coord)
    y_coord = jnp.asarray(y_coord, dtype=x_coord.dtype)
    inside_removed = lshape_removed_quadrant_mask(x_coord, y_coord)

    # Boundary tags. We use eps-tight equality via abs() < eps.
    eps = jnp.asarray(1e-12, dtype=x_coord.dtype)
    on_left = jnp.asarray(jnp.abs(x_coord) < eps, dtype=x_coord.dtype)
    on_right = jnp.asarray(jnp.abs(x_coord - 1.0) < eps, dtype=x_coord.dtype)
    on_bottom = jnp.asarray(jnp.abs(y_coord) < eps, dtype=x_coord.dtype)
    on_top = jnp.asarray(jnp.abs(y_coord - 1.0) < eps, dtype=x_coord.dtype)
    on_boundary = on_left + on_right + on_bottom + on_top
    # Clamp to [0, 1] in case a node is at a corner.
    on_boundary = jnp.minimum(on_boundary, jnp.asarray(1.0, dtype=x_coord.dtype))

    mask = inside_removed + on_boundary
    return jnp.minimum(mask, jnp.asarray(1.0, dtype=x_coord.dtype))


def free_node_mask(x_coord: Array, y_coord: Array) -> Array:
    """Complement of the Dirichlet mask: 1 if the node is free, 0 if constrained."""
    return jnp.asarray(1.0, dtype=x_coord.dtype) - lshape_dirichlet_node_mask(
        x_coord, y_coord
    )


__all__ = [
    "Array",
    "sign_indicator",
    "step_ge",
    "step_le",
    "lshape_removed_quadrant_mask",
    "lshape_dirichlet_node_mask",
    "free_node_mask",
]
