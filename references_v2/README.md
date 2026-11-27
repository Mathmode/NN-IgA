# References, revision v2

| File | Content | Producer |
|---|---|---|
| `lshape_energy_ngsolve.csv` | Exact-energy reference `F(u) = int_Omega u` for the 60 test pairs `(sigma1, sigma2)` of Experiment 4, hp-FEM (NGSolve) on the three subdomains, geometric corner refinement (16 levels, factor 0.15), orders 12 and 14 | `2D/scripts/lshape_ngsolve_energies.py` |
| `reference_lshape.pkl` | Degree-5 IGA reference with the corrected Dirichlet mask (20x20 sigma grid). Used only for field-based figures. **Pending: Hyperion job** `slurm_v2/regen_reference_lshape.sbatch` | `2D/scripts/generate_reference_lshape.py` |

Accuracy of `F(u)`: the orders 12 and 14 agree to `3.9e-9` (relative, worst pair);
14 versus 16 geometric levels at order 14 agree to `7.8e-10`. Twenty levels break
the sparse Cholesky (smallest element `0.15^20 ~ 3e-17`).

The degree-5 IGA reference is not accurate enough to measure errors at `N = 64`:
against `F(u)`, its relative energy error is `1.8e-4` at `sigma = (1.13, 1.13)` and
`2.7e-4` at `sigma = (0.1, 10)`. Errors are therefore measured with the energy
identity `||u - u_h||_E^2 / ||u||_E^2 = 1 - F(u_h)/F(u)` (`2D/scripts/lshape_energy_errors.py`).
