"""The `helmholtz` experiment: parametric 1D Helmholtz transmission family.

Self-contained subpackage alongside the `singular` experiment (which it does not
modify). The parameter is the wavenumber contrast c = k1/k2. Modules:
  mesh_helmholtz_global    contrast PDE data, exact solution, resonance-safe sampling,
                           global C^0 knot map (+ Nyquist h_max cap), assembly,
                           solve, residual eta^2, H1 error, uniform 50:50 baseline
  solver_indef             indefinite symmetric LU solve with discrete-adjoint VJP
  network_helmholtz        positional density network (c, xi, side)
  training_helmholtz       coarse-to-fine + contrast continuation, eq-26 loss
"""
