from __future__ import annotations

"""Shared precision policy for the 1D experiment modules."""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

DEFAULT_DTYPE = jnp.float64


def ensure_double_precision() -> None:
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("Double precision is required: set JAX_ENABLE_X64=1.")


__all__ = ["DEFAULT_DTYPE", "ensure_double_precision"]
