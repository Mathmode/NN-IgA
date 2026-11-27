"""Experiment 6 (revision v2): parametric 1D advection-diffusion-reaction.

    -eps u'' + b u' + u = f  on (0, 1),  u(0) = u(1) = 0,
    u*(x) = x (1 - exp((x - 1)/delta)),  delta = eps / b,

with (log10 eps, b) in [-2, -1] x [0.5, 2]. Non-symmetric, coercive, mu = 1,
c_* = 1: every assumption of the theory holds. Modules: problem, discretization,
network, training.
"""
