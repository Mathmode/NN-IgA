"""The `helmholtz` experiment: parametric 1D Helmholtz transmission family.

Self-contained subpackage alongside the `singular` experiment (which it does not
modify). Modules:
  pde_helmholtz            exact transmission solution + resonance scan
  solver_indef             indefinite symmetric LU solve with discrete-adjoint VJP
  mesh_helmholtz           two-patch C^0 mesh policy (per-half grading)
  discretization_helmholtz K(sigma), M(rho), B=K-omega^2 M, solve, H1 error
  estimator_helmholtz      residual eta^2 (reaction + interface jump + Neumann)
  network_helmholtz        positional density network (omega, xi, axis_id)
  training_helmholtz       coarse-to-fine continuation + residual-normalized loss
"""
