# Revision v2: lessons on the new code

The new code of revision v2 carries no comments. This document explains it instead, one function at a time. Each lesson gives the idea first and then shows how the code turns it into a few lines of NumPy, SciPy, JAX or NGSolve.

Contents

1. [L-shape conformity](#1-l-shape-conformity-2dsrcnonparametriclshapeconformitypy)
2. [The energy identity and reference-free errors](#2-the-energy-identity-and-reference-free-errors-2dsrcnonparametriclshapeenergy_errorpy)
3. [An independent hp reference with NGSolve](#3-an-independent-hp-reference-with-ngsolve-2dsrcnonparametriclshapengsolve_energypy)
4. [Helmholtz resonances](#4-helmholtz-resonances-1dsrchelmholtzresonancepy)
5. [Scripts of Phase 2](#5-scripts-of-phase-2)
6. [The Ritz loss](#6-the-ritz-loss-1dsrcparametricritz_losspy-2dsrcparametricritz_loss_2dpy)
7. [Algorithm 1 with a continuation ladder](#7-algorithm-1-with-a-continuation-ladder-1dsrcnonparametricalg1_ladderpy-2dsrcnonparametricalg1_ladder_2dpy)
8. [Experiment 6: advection–diffusion–reaction](#8-experiment-6-advectiondiffusionreaction-1dsrcadr)
9. [The oblique layer](#9-the-oblique-layer-2dsrcnonparametricobliquepy)
10. [Measuring cost and constants](#10-measuring-cost-and-constants)

---

## 1. L-shape conformity (`2D/src/nonparametric/lshape/conformity.py`)

### The problem

The L-shape $\Omega=(0,1)^2\setminus Q$, with $Q=[1/2,1]\times[0,1/2]$, is discretized with one tensor-product spline space on the whole square. The knot $1/2$ has multiplicity $p$ in both directions. A basis function $N_{ij}=B_i(x)B_j(y)$ is *active* if it is not on the outer boundary and its support does not meet the open quadrant $Q^\circ$. Lemma 5.1 of the paper says that every active function vanishes on the closed quadrant, so the space of active functions is $\{v\in\mathbb S: v|_Q=0,\ v|_{\partial(0,1)^2}=0\}$. That makes the discretization conforming.

The proof uses two facts about B-splines. First, $B_i$ is supported on $[t_i,t_{i+p+1}]$. Second, with multiplicity $p$ at $1/2$, only one univariate function is nonzero at $1/2$. So the "does the support meet $Q^\circ$" test is a pure knot-vector test. No Greville abscissa and no floating-point comparison with $1/2$ is needed. That is why the bug of the v1 reference (a Greville mean drifting below $1/2$ by one ULP) cannot happen with this rule.

### `support_bounds(knots, p)`

Returns the two arrays $t_i$ and $t_{i+p+1}$, $i=0,\dots,n-1$, the left and right ends of every support. In practice it slices the knot vector twice.

### `quadrant_support_mask(knots_x, knots_y, p, corner=0.5)`

The support of $N_{ij}$ is $[t^x_i,t^x_{i+p+1}]\times[t^y_j,t^y_{j+p+1}]$. It meets $Q^\circ=(1/2,1)\times(0,1/2)$ exactly when $t^x_{i+p+1}>1/2$ and $t^y_j<1/2$; the other two inequalities always hold. The function returns this boolean matrix with an outer product of the two 1D tests.

### `support_rule_free_mask(knots_x, knots_y, p, corner=0.5)`

The free (active) degrees of freedom: the complement of the quadrant mask, with the outer rows and columns removed. This is the "truth" against which the solver's Greville-based `free_mask_lshape` is tested.

### `design_matrix(knots, p, x)` and `tensor_values(coeffs, knots_x, knots_y, p, xs, ys)`

`design_matrix` wraps `scipy.interpolate.BSpline.design_matrix`. SciPy refuses the right end point of an open knot vector, so the points are clipped to the last representable number below it (`np.nextafter`). `tensor_values` evaluates $u(x,y)=\sum_{ij}C_{ij}B_i(x)B_j(y)$ on a grid as $B_x\,C\,B_y^{\mathsf T}$, a single matrix product.

### `active_basis_sum_on_quadrant(knots_x, knots_y, p, free, n=201)`

Because $B_i\ge0$, the sum of all active functions vanishes on a set only if every active function does. So passing the 0/1 matrix `free` as "coefficients" to `tensor_values` on a grid of the *closed* quadrant (edges included) tests Lemma 5.1 in one evaluation. The test asserts the result is exactly `0.0`, not merely small.

### `constrained_coefficient_max(coeffs, knots_x, knots_y, p)` and `dirichlet_violation(...)`

B-splines are locally linearly independent. A spline vanishes on $Q$ and on the outer boundary exactly when all its non-free coefficients are zero. `constrained_coefficient_max` checks this algebraically. `dirichlet_violation` also samples $|u|$ on the closed quadrant and on the four outer edges, and returns the three numbers in a dictionary. The reference generator stores this dictionary with each entry and refuses to write a cache whose worst value exceeds $10^{-14}$.

### `random_graded_lshape_knots(rng, p, n_half)` and `geometric_lshape_knots(p, n_half, refine)`

These are test inputs. The first draws sorted uniform breakpoints on each side of $1/2$, which gives strongly non-uniform meshes. The second reproduces the geometric corner grading of the degree-5 reference: sizes $r^k$ with $r=1/\text{refine}$, normalized to sum to $1/2$ and mirrored about $1/2$. With `refine=1` it gives the uniform knots used for nested-space tests.

---

## 2. The energy identity and reference-free errors (`2D/src/nonparametric/lshape/energy_error.py`)

### Theory

Take $a(u,v)=\int\sigma\nabla u\cdot\nabla v$, $F(v)=\int f v$ with $f=1$, $V_h\subset H^1_0(\Omega)$ (conforming) and $u_h$ the Galerkin solution. Then $a(u,u)=F(u)$ and $a(u_h,u_h)=F(u_h)$. Galerkin orthogonality $a(u-u_h,u_h)=0$ gives

$$\|u-u_h\|_E^2=a(u,u)-2a(u,u_h)+a(u_h,u_h)=F(u)-F(u_h).$$

Dividing by $\|u\|_E^2=F(u)$ gives

$$\frac{\|u-u_h\|_E^2}{\|u\|_E^2}=1-\frac{F(u_h)}{F(u)}.$$

Three practical consequences follow:

- The error needs one scalar of the exact solution, $F(u)$, and no field. Interpolating a reference onto the mesh adds no error of its own.
- $F(u_h)=\mathbf F^{\mathsf T}\mathbf U$ is exact for the discrete problem. The Ritz energy is $J(u_h)=\tfrac12\mathbf U^{\mathsf T}\mathbf K\mathbf U-\mathbf F^{\mathsf T}\mathbf U=-\tfrac12\mathbf F^{\mathsf T}\mathbf U$, so $F(u_h)=-2J(u_h)$, read from the stored `ritz_energy` column.
- Conformity is testable. A conforming method must give $F(u_h)\le F(u)$ on every row.

### Functions

| Function | Formula | Remark |
|---|---|---|
| `discrete_energy(ritz_energy)` | $F(u_h)=-2J(u_h)$ | |
| `relative_energy_gap(F_h, F)` | $(F-F_h)/F$ | Signed, kept for diagnostics. A negative gap above the reference tolerance would reveal a non-conforming space. |
| `relative_energy_error(F_h, F)` | $\sqrt{\max(\text{gap},0)}$ | Clipped, because a gap of $-10^{-10}$ is only reference noise. |
| `absolute_energy_error(F_h, F)` | $\sqrt{F-F_h}=\|u-u_h\|_E$ | Used for the effectivity index $I_{\rm eff}=\eta/\|u-u_h\|_E$. |

`attach_energy_errors(rows, energies, column)` merges the per-$(\sigma_1,\sigma_2)$ reference energies into the evaluation rows. The merge uses `validate="many_to_one"`, so a duplicated reference row fails loudly. It then adds `F_h`, `F_ref`, `gap`, `err_E` and `I_eff_E`.

`median_table` and `improvement_factors` produce the medians per $(p,N,\text{method})$ and the ratio uniform / adapted.

### Effective orders

For errors $e_N$ at levels $N$:

- `endpoint_order`: $\log(e_{N_0}/e_{N_1})/\log(N_1/N_0)$, the slope between the first and last level, as used in Table 4.
- `local_orders`: the same slope between consecutive levels.
- `fitted_order`: minus the least-squares slope of $\log e$ against $\log N$. The paper uses it for the uniform rate over $N\in[16,64]$.

`order_summary(table, method, fit_from=16)` collects the three per degree.

---

## 3. An independent hp reference with NGSolve (`2D/src/nonparametric/lshape/ngsolve_energy.py`)

### Why NGSolve

A reference computed with the same code family cannot reveal a bias shared by that family. The reviewer's point is exactly this, and it is how the v1 reference bug was found. NGSolve is an independent code. With $hp$-FEM on a mesh fitted to the three subdomains and geometric refinement toward the re-entrant corner, it converges exponentially for the $r^{2/3}$ corner singularity.

### `build_lshape_mesh(levels=16, factor=0.15, maxh=0.2)`

The domain is glued from three rectangles named `bl` ($\sigma_1$), `tl` ($1$) and `tr` ($\sigma_2$). The OCC kernel makes the shared edges conforming.

**Pitfall 1: edge names.** The two internal interfaces ($x=1/2,y>1/2$ and $y=1/2,x<1/2$) are named `iface`. Every other edge, including the two re-entrant edges, is named `dir`. Only `dir` gets the Dirichlet condition. Imposing $u=0$ on the interfaces would solve a different problem.

The corner vertex gets `hpref=1`, and `RefineHP(levels, factor)` creates `levels` layers of elements shrinking by `factor` toward it. Twenty levels make elements of size $0.15^{20}\approx3\times10^{-17}$ and the sparse Cholesky fails, so 16 is the practical maximum.

### `boundary_names(mesh)`

Returns the set of boundary names. The script checks that it is exactly `{dir, iface}` before solving.

### `solve_energy(mesh, sigma1, sigma2, order)`

It assembles $a$ and $F$ on `H1(order, dirichlet="dir")` and solves with sparse Cholesky on the free DOFs.

**Pitfall 2: integration order.** It returns $F(u)$ as `InnerProduct(f.vec, gfu.vec)`, the algebraic product $\mathbf F^{\mathsf T}\mathbf U$. That is exact for the discrete solution. `Integrate(gfu, mesh)` uses a default quadrature order of 5, which is far too low for order-14 polynomials.

Accuracy is measured by running two orders (12 and 14), and separately two refinement depths.

---

## 4. Helmholtz resonances (`1D/src/helmholtz/resonance.py`)

### The continuous resonances

The transmission problem $-(\sigma u')'-\rho u=0$ with $u(0)=0$, $\sigma u'(1)=g_N$ has $k_2=5\pi$ fixed and $k_1=5\pi c$. The homogeneous problem ($g_N=0$) has $u=A\sin(k_1x)$ on the left and $u=B\cos(k_2(1-x))$ on the right, the latter satisfying the homogeneous Neumann condition. Continuity at $x_I=1/2$ gives $A\sin(k_1/2)=B\cos(k_2/2)$, and $\cos(5\pi/2)=0$. So either $\sin(k_1/2)=0$, or $A=0$, and then flux continuity forces $B=0$. Nontrivial solutions therefore exist exactly when $5\pi c/2=m\pi$, that is $c=2m/5$. `continuous_resonances(c_max)` returns $0.4,0.8,\dots$, and `mode_index(c)` returns the zero-based index $m-1$.

### The discrete resonances

The Galerkin matrix is $\mathbf A(c)=\mathbf S-\rho_2\mathbf M_R-25\pi^2c^2\,\mathbf M_L$, where $\mathbf S$ is the $\sigma$-weighted stiffness and $\mathbf M_L,\mathbf M_R$ are the mass matrices of the left and right subdomains. It is singular when $\lambda=25\pi^2c^2$ is a generalized eigenvalue of the pencil $(\mathbf S-\rho_2\mathbf M_R,\ \mathbf M_L)$ on the free DOFs ($u_0=0$ removed). $\mathbf M_L$ is singular, because right-only functions have no left mass, so the pencil has infinite eigenvalues. `discrete_resonances` drops them and the complex ones, keeps $\lambda>0$, and returns $c_h=\sqrt{\lambda}/(5\pi)$ sorted.

### Associating a discrete with a continuous resonance

The discrete spectrum is shifted upward. On coarse meshes the shift can exceed the gap $0.4$ between continuous resonances, so "the nearest $c_h$" is the wrong partner. Example: uniform $p=2$, $N=32$, where the mode of $c=4.8$ sits at $5.007$, closer to $5.2$. `associated_by_index` pairs the $k$-th discrete with the $k$-th continuous resonance (Sturm ordering). `associated_in_window` reproduces the window rule of the pilot scripts ($[c-0.15,c+0.35]$, smallest) for comparison. `nearest` is used only to measure how close a test contrast is to *some* discrete resonance.

### The independent solver

`uniform_knots`, `assemble_blocks`, `system` and `solve` re-implement the Helmholtz IGA with SciPy B-splines and Gauss–Legendre quadrature. They share no code with `mesh_helmholtz_global`.

- `_basis` evaluates all basis functions and their derivatives at once, by giving `BSpline` the identity matrix as coefficients.
- `exact_coefficients` solves the $3\times3$ system of the two interface conditions and the Neumann condition.
- `exact_solution` and `exact_derivative` build $u^\ast$ and $u^{\ast\prime}$ from it.
- `h1_rel` integrates $|u_h'-u^{\ast\prime}|^2$ element by element with 24 Gauss points, so the derivative jump at $x_I$ is never inside a quadrature cell.
- `evaluate` samples $u_h$ for the figures.

The tests check that both solvers give the same relative $H^1$ error to $10^{-8}$.

---

## 5. Scripts of Phase 2

- `2D/scripts/lshape_ngsolve_energies.py`: collects the test pairs $(\sigma_1,\sigma_2)$ from the evaluation CSVs and calls `solve_energy` at two orders on one mesh. It writes `references_v2/lshape_energy_ngsolve.csv`.
- `2D/scripts/lshape_energy_errors.py`: merges the energies with the 4800 evaluation rows, checks $F(u_h)\le F(u)$, and writes the corrected Table 4 / R5 and the effective orders. To bound the effect of the reference tolerance, `bracket_energies` replaces $F$ by $F\pm|F_{14}-F_{12}|$ and `factor_range` recomputes the improvement factor at both ends.
- `1D/scripts/helmholtz_resonance_study.py`: computes Table R2, Table R1, the error scan of Figure R1, the cases where the adapted mesh meets its own resonance, and the profiles of Fig. 6. Every error is computed by both solvers.
- `1D/scripts/helmholtz_v2_figures.py`: writes the pgfplots sources and their CSV data.

---

## 6. The Ritz loss (`1D/src/parametric/ritz_loss.py`, `2D/src/parametric/ritz_loss_2d.py`)

### Theory

For a symmetric coercive problem $a(u,v)=F(v)$ with a conforming space, the Ritz energy $J(v)=\tfrac12a(v,v)-F(v)$ satisfies

$$J(u_h)-J(u)=\tfrac12\|u-u_h\|_E^2.$$

So minimizing $J(u_h(\theta))$ over the mesh parameters $\theta$ minimizes the true energy error, without knowing $u$. This makes it the natural upper bound for what an estimator-based loss can achieve, but it exists only for symmetric problems. At the Galerkin solution $J(u_h)=-\tfrac12\mathbf F^{\mathsf T}\mathbf U$.

### The loss and its normalization

Each sample contributes $J(u_h(\theta_\phi(\nu)))/|J(u_{h,\mathrm{unif}})|$, with the uniform mesh at the *same* level. This mirrors the $\eta^2/\eta^2_{\mathrm{unif}}$ normalization of eq. (26), so samples with large and small energies mix in one batch.

### Why early stopping uses `tol = 0`

$J(u_h)/|J_{\rm unif}| = -1 + (\|e\|_E^2-\|e_{\rm unif}\|_E^2)/(2|J_{\rm unif}|)$ stays within $10^{-4}$ of $-1$. Once the error is well below the uniform one, a 10% gain in $\|e\|^2$ moves the loss by far less than 1%. The 1% relative tolerance of the validation protocol therefore stops training after the first epoch. With `--loss ritz` the tolerance is set to 0 (any decrease counts), and patience, minimum epochs and cap stay as before.

The improvement test was also rewritten as $v<b-\mathrm{tol}\,|b|$, because $v<b(1-\mathrm{tol})$ is wrong for negative $b$. For positive losses the two are identical. The first finite value always counts as an improvement.

### Functions

- `ritz_energy(theta, p, q, h_min, beta)` (1D) builds the cache, assembles the reduced system, solves it and returns $-\tfrac12\mathbf F\cdot\mathbf u$.
- `network_theta` evaluates the network logits at one level.
- `make_per_beta_ritz_loss`, `make_uniform_ritz_fn` and `make_val_forward_ritz` are the Ritz twins of the estimator-loss factories. `train_at_level(..., loss="ritz")` selects them.
- In 2D, `network_solve` builds both knot vectors from the network and solves. `uniform_ritz` gives the uniform-mesh energy. `make_batch_loss_ritz`, `make_val_forward_ritz` and `make_ritz_denoms_fn` plug into `train_with_continuation(..., loss="ritz")` exactly where the estimator versions plug in.

---

## 7. Algorithm 1 with a continuation ladder (`1D/src/nonparametric/alg1_ladder.py`, `2D/src/nonparametric/alg1_ladder_2d.py`)

Algorithm 1 optimizes the mesh of a single instance. Cold-started at a fine level it can stall in poor local minima: in the pilot, $p=3$ was 4–7 times worse than the network. The ladder gives it the same coarse-to-fine path as the network. Optimize at $N_0$, refine, and use the result as the starting point at $2N_0$.

### Prolongation by repetition

With sizes $h=\operatorname{softmax}(\theta)$, the vector $\theta'=(\theta_1,\theta_1,\theta_2,\theta_2,\dots)$ gives $\operatorname{softmax}(\theta')_{2i}=\operatorname{softmax}(\theta')_{2i+1}=h_i/2$. Every element is split into two equal halves: the refined mesh contains the old one, so the starting error at $2N$ is no larger than the final error at $N$. `prolong` is `jnp.repeat(theta, 2)`. In 2D it is applied block by block (2 blocks for tensor meshes, 4 for the L-shape).

### Functions

- 1D: `eta_objective` and `ritz_objective` return $\tfrac12\eta^2$ or $J$ as functions of $\theta$; `normalized_objective` divides them by the absolute value on the uniform mesh of the same level. `adam_minimize` runs projected Adam with exponential decay (every iterate is clipped to $[-T,T]$, with $T$ from `bound_for(p, N)`, i.e. `T_SCHEDULE`) and keeps the best iterate. `run_ladder` and `run_cold` run the two strategies. Timings exclude JIT compilation, because one warm-up call is made before the clock starts.
- 2D: a `Problem` record bundles the number of blocks, the knot map, the solver, the estimator and the bound $T$. `arctan_problem`, `lshape_problem` and `oblique_problem` fill it. `objective` returns a jitted value-and-gradient of the loss normalized by the uniform mesh of the level. `lbfgs` wraps SciPy's L-BFGS-B, as in `compare_nonparametric_2d.py`, with box constraints. `solution` re-solves for the metrics. `mesh_quality` returns $\gamma_{\rm loc}$ and the maximum aspect ratio.

### The admissible set $\Theta_{\rm ad}$

The network saturates its logits, $q_i=T\tanh(\tilde s_i/T)\in[-T,T]$, so any two element sizes of a predicted mesh satisfy $h_i/h_j\le e^{2T}$. Section 3 of the paper restricts $\Theta_{\rm ad}$ in the same way for Algorithm 1. Up to the first Hyperion run, Algorithm 1 did not: the 2D `lbfgs` called L-BFGS-B without bounds, and the 1D `adam_minimize` ran plain Adam. Both searched all of $\mathbb R^n$.

That matters for the Ritz loss. On a degenerate mesh, the stiffness matrix is close to singular and the computed $J(u_h)$ is dominated by round-off; it can fall below $J(u)$, which is impossible in exact arithmetic, since $J(u_h)-J(u)=\tfrac12\|u-u_h\|_{\mathcal E}^2\ge0$. The optimizer then sees a "better" value and moves on. The estimator does not have this failure: $\eta^2$ grows on such meshes. In the first 2D run, for the L-shape with $p=2$ and $\sigma=(0.695,7.848)$, a long line-search step at $N=4$ reached a logit spread of $27.3$ (size ratio $6.8\times10^{11}$), gave $J(u_h)-J(u)=-1.0\times10^{-3}$ and $\eta=5.6\times10^{9}$, and the next levels returned NaN. Aspect ratios above $e^{2T}$ also appeared in other instances, with both losses. In 1D, with $p=2$ and $2T=4$ up to $N=32$, the estimator ladder exceeded the spread $2T$ for 43 of the 67 test exponents at $N=32$ (maximum $6.4$), and the Ritz ladder for 9.

The fix is the box $|\theta_i|\le T$. In 2D it is passed to L-BFGS-B (`bounds=[(-T, T)] * n`); in 1D every Adam iterate is projected onto it (`jnp.clip`). Because the softmax is invariant under a common shift of $\theta$, the box describes the same set of meshes as the spread condition $\max_i\theta_i-\min_i\theta_i\le 2T$; the box is simply the form the optimizers accept. The starting point $\theta=0$ is inside the box, and the prolongation by repetition keeps every value (in 1D, $T$ does not decrease along the ladder), so each level starts feasible. $T$ is not a constant of these modules. In 2D, `Problem.bound(p)` returns the value of the network of the same experiment from `2D/src/config.py`: `T_ARCTAN` for the arctangent layer and for the oblique layer (a rotated Experiment 3), and `T_LSHAPE_BY_P.get(p, T_LSHAPE)` for the L-shape. In 1D, `bound_for(p, N)` reads `T_SCHEDULE[p][N]` from `1D/src/config.py`. The bound applies to both losses.

### Normalizing the loss by the uniform mesh

The box alone did not fix the Ritz runs in 2D; it made them stop earlier. SciPy's L-BFGS-B stops when $(f_k-f_{k+1})/\max(|f_k|,|f_{k+1}|,1)\le$ `ftol`. For the L-shape $|J|\approx5\times10^{-3}$, so the denominator is $1$ and the test is absolute: with `ftol=1e-12`, the optimizer stopped after one iteration at $N=16$ and $N=32$ (`REL_REDUCTION_OF_F`) while the gradient was still $10^{-7}$. The run without bounds had the same problem (median $2.5$ iterations at $N=32$ for $p=3$).

Both versions of Algorithm 1 now minimize the loss divided by its absolute value on the uniform mesh of the same level, $\tfrac12\eta^2/\tfrac12\eta^2_{\rm unif}$ or $J/|J_{\rm unif}|$, as the network does (eq. 26 and the normalized Ritz loss). The value at $\theta=0$ is $\pm1$, so the stopping tests are relative. Normalization fixes the scale of the gradient, but not the offset: $J/|J_{\rm unif}|=-1+O(\|e\|_{\mathcal E}^2/\|e_{\rm unif}\|_{\mathcal E}^2)$, and at $N=32$ all that is left to gain is of order $10^{-4}$, so a step still lowers $f$ by less than $10^{-12}$ and the `ftol` test fires (in the first normalized run, 11 of the 16 L-shape instances with $p=3$ stopped after one iteration at $N=32$). The 2D `lbfgs` therefore uses `ftol=0`: it stops on the projected gradient (`gtol=1e-9`), on a line search that cannot decrease $f$ any further (round-off floor, `ABNORMAL_TERMINATION_IN_LNSRCH`, typical for the estimator at fine levels), or at `maxiter`. On the L-shape with $p=3$ and $\sigma=(0.428,2.336)$ the Ritz ladder then takes 27 iterations at $N=32$ instead of 1, and the error drops from $1.09\times10^{-3}$ to $4.80\times10^{-4}$. The scale is computed once per level, before the clock starts. Adam is almost invariant under this scaling (only its $\epsilon$ is not), so in 1D the change matters much less than the projection; it is applied there too for consistency.

On the instance above, the bounded and normalized ladder converges by the projected-gradient test (`PGTOL`) in $15$–$23$ iterations per level. $J(u_h)-J(u)$ goes from $1.7\times10^{-5}$ ($N=4$) to $9.2\times10^{-9}$ ($N=32$), the spread stays at or below $4.5$, and the error at $N=32$ is $1.37\times10^{-3}$ (uniform mesh: $6.54\times10^{-3}$). The bound is not active at the optimum; it stops the line search from leaving the region of admissible meshes.

### Tests

- `2D/tests/test_alg1_bounds.py`: $T$ comes from the configuration; the objective is $\pm1$ on the uniform mesh; on the instance above the Ritz ladder stays in the box, keeps $h_{\max}/h_{\min}\le e^{2T}$, gives $J(u_h)\ge J(u)$ at every level (with $J(u)=-\tfrac12F(u)$ from the NGSolve reference), its error decreases along the ladder, and at $N=32$ it is at most $0.3$ times the uniform-mesh error.
- `1D/tests/test_alg1_bounds_1d.py`: `bound_for` reads `T_SCHEDULE`; the objective is $\pm1$ on the uniform mesh; for the exponent with the largest violation in the old run ($\beta=1.554$, $p=2$) both ladders stay in the box at every level and improve on the uniform mesh.

### Results kept as a record

The results without bounds stay in git as a record: `data_results_v2/alg1_2d/` (2D comparison and oblique layer) and `data_results_v2/loss_ablation/exp1_alg1_*.csv` (1D). The bounded run without normalization (job 8038158) is kept outside git in `data_results_v2/alg1_2d_bounded_unnormalized/`. The paper and the letter use only `data_results_v2/alg1_2d_bounded/` and `data_results_v2/loss_ablation_bounded/`. `scripts/summarize_alg1_2d.py --in-dir` selects the directory, and its column `n_valid` counts the finite errors of each group, so a failed instance can no longer hide behind the median.

---

## 8. Experiment 6: advection–diffusion–reaction (`1D/src/adr/`)

### The problem and why it was chosen

$-\varepsilon u''+bu'+u=f$ on $(0,1)$, $u(0)=u(1)=0$, with $u^\ast=x(1-e^{(x-1)/\delta})$ and $\delta=\varepsilon/b$. It has an outflow layer of width $\delta$ at $x=1$. With constant $b$ and $\alpha=1$ we get $\mu=\alpha-\tfrac12b'=1$, and Assumption III holds with $c_\ast=1$. It is non-symmetric, so the Ritz approach does not apply. In 1D shape regularity is automatic. Every hypothesis of the theory is therefore met. The forcing is computed from the closed forms `exact`, `exact_dx` and `exact_dxx` in `problem.py`.

### Discretization (`discretization.py`)

- `_points(knots, p, nq, n_sub)` places Gauss points on every element, optionally on `n_sub` equal sub-cells, and evaluates the B-splines there. The index of the first active function comes from the span returned by `bspline_basis_local`, not from the element number. This is robust to repeated knots.
- `solve` assembles $a(u,v)=\varepsilon(u',v')+(bu',v)+(u,v)$. Test functions are rows and trial functions columns, so the convection entry is $bN_i N_j'$. It removes the two Dirichlet DOFs and solves the non-symmetric system with `jnp.linalg.solve`, which JAX can differentiate.
- `eta2_elements` computes the residual estimator $\rho_E^2\|f+\varepsilon u_h''-bu_h'-u_h\|_E^2$ with $\rho_E=\min(h_E/\sqrt\varepsilon,1)$ (because $\mu=1$). Splines of degree $p\ge2$ with simple knots are $C^1$, so there are no flux jumps in 1D.
- `energy_errors` integrates $\varepsilon|e'|^2+|e|^2$ on 8 sub-cells per element, so the layer is resolved even when one coarse element contains it. It returns the absolute and relative energy error.
- `gamma_loc` takes the maximum, over windows of $2p+1$ consecutive elements, of the ratio of the largest to the smallest size.

### Network and training (`network.py`, `training.py`)

The inputs $(\log_{10}\varepsilon,b,\xi)$ are each mapped to $[-1,1]$. Logits are gauge-fixed and bounded by $T\tanh(z/T)$, as in the other experiments. Training follows the validation protocol of `config.VAL`: Adam with warm-up and decay, gradient clipping, batch 16, and early stopping on the validation $\eta^2$ with a 1% tolerance. Continuation runs $8\to64$, then zero-shot evaluation at $128$.

---

## 9. The oblique layer (`2D/src/nonparametric/oblique.py`)

Experiment 3 is a product of two 1D layers aligned with the axes, which is the best case for a tensor-product mesh. To measure the restriction, we rotate the layer: $u=xy\arctan\bigl(t(x\cos\vartheta+y\sin\vartheta-s)\bigr)$, with $t=20$ and $s=1/2$. The factor $xy$ enforces $u=0$ on $x=0$ and $y=0$ without changing the layer.

Instead of deriving $\Delta u$ and the Neumann data by hand, `u_scalar` is written once. `jax.grad` gives the gradient and `grad(grad)` the second derivatives. `jnp.vectorize` maps these over arrays, and `forcing`, `neumann_right` and `neumann_top` follow from them.

`galerkin_solve_oblique` reuses the assembly of `solver_2d`, passing callables. `eta_squared_oblique` is the arctan estimator with these callables: volume residual weighted by $h_x^2+h_y^2$ and Neumann residuals on $x=1$ and $y=1$ weighted by $h$.

---

## 10. Measuring cost and constants

- `common/resources.py`: `peak_rss_mb` reads the peak resident memory. On Linux it uses `getrusage`. On Windows it calls `GetProcessMemoryInfo`, with the handle and argument types declared, because without them the call silently returns zeros. `hardware_info` records the platform, CPU model, core count and Slurm allocation. All training scripts now store both in their metadata.
- `1D/scripts/mesh_diagnostics_1d.py` and `2D/scripts/mesh_diagnostics_2d.py` compute:
  - the realized $\gamma_{\rm loc}$ (window of $2p+1$ elements per direction), the global ratio, the maximum aspect ratio (on the L only, for the L-shape) and $e^{2T}$;
  - condition numbers on the free DOFs, raw and after symmetric Jacobi scaling $D^{-1/2}KD^{-1/2}$;
  - the online time, the median of repeated forward passes plus one solve, after a warm-up call.
- `scripts/fixed_accuracy.py`: the target is the uniform error at $N_{\max}$. The number of elements per direction the adapted mesh needs is found by linear interpolation of $\log e$ against $\log N$, and the DOF ratio is $((N_{\max}+p)/(N_a+p))^d$.
- `scripts/cost_table.py --root <dir>` builds the cost table from a single directory written on one CPU model, so that $n^\ast=T_{\rm train}/(T_{\rm alg1}-T_{\rm online})$ does not mix machines. $T_{\rm train}$ is the median over the seeds of `<root>/timing` (with min, max and peak memory), $T_{\rm online}$ comes from `<root>/mesh_diagnostics`, and $T_{\rm alg1}$ from `<root>/loss_ablation` (1D) and `<root>/alg1_2d` (2D). The column `cpu_train` lists the CPU models found in the timing metadata. The median of the original paper runs (`data_results/`, mixed nodes) is kept in `T_train_paper_s` for comparison only.

### The same-machine round on Hyperion

The feature `icelake` of the partition covers two CPU models: Xeon Platinum 8362 (64 cores, 2.8 GHz) and Platinum 8368 (76 cores, 2.4 GHz, nodes `hyperion-[225-233]`). All jobs of this round use `--constraint=icelake --exclude=hyperion-[225-233]`, which leaves only the 8362 nodes, and write under `data_results_v2/icelake/`.

- `slurm_v2/timing_runs.sbatch`: array 0–39; `task % 10` selects the experiment and degree as before and `task / 10` is the seed, so that the table reports the median of four seeds. Every driver writes per-seed files (`checkpoints/seed<s>`, `meta_*_seed<s>.json`), so the seeds can share an output directory.
- `slurm_v2/alg1_1d.sbatch`: Algorithm 1 for Experiment 1 (task 0: ladder over the 67 test exponents; task 1: cold-start pilot) on the cluster, so that $T_{\rm alg1}$ in 1D is measured on the same nodes as $T_{\rm train}$.
- `slurm_v2/mesh_diagnostics.sbatch`: the two diagnostics scripts (task 0: 1D, task 1: 2D), for $T_{\rm online}$ on the same nodes.
- `slurm_v2/alg1_2d.sbatch`: `MAXITER` can be overridden (default 200); this round uses 1000, because with 200 most arctangent instances with $p=3$ stopped at the iteration limit at $N=8$ and $N=16$.
- `2D/src/nonparametric/alg1_ladder_2d.py`: `lbfgs` also returns the stopping reason (`stop_reason`: `gradient` for the projected-gradient test, `line_search` when the line search cannot decrease $f$ any further, `maxiter`) and the number of components with $|\theta_i|=T$ at the final iterate. `2D/scripts/alg1_2d.py` writes them as the columns `stop` and `n_active`.
- `1D/src/nonparametric/alg1_ladder.py` compiles the objective once per (loss, $p$, $N$), with $\beta$ and the normalization as arguments (`compiled_objective`, `normalized_value_and_grad`). The first version built a new `jax.jit` for every exponent and level, about 1600 compilations for the 67 exponents; on Hyperion the JIT cache grew until LLVM could not allocate memory for code (job 8040604, task 0, segmentation fault after 27 min with 3.8 GB RSS of 8 GB). Now the cache holds one entry per (loss, $p$, $N$) and the memory does not grow with the number of exponents. The results agree with the previous version up to round-off (median relative difference $0$, 90th percentile $4\times10^{-9}$, maximum $10^{-3}$ for one Ritz entry at $p=3$, $N=64$, where Adam amplifies the round-off of a flat landscape over 1000 iterations), and the time per instance is unchanged (ratio $0.98$–$1.00$), since a warm-up call before the clock still keeps the compilation out of `time_s`. `1D/scripts/loss_ablation_1d.py --beta-range LO HI --tag chunk<k>` and the array of `slurm_v2/alg1_1d.sbatch` (tasks 0–7: chunks of nine exponents, task 8: pilot) write one CSV per chunk.

## v3: L-shape

### Validation protocol and the earlier smoke

The 20 × 20 conductivity grid uses the existing seeded split with 280 training pairs, 60 validation pairs and 60 test pairs. Algorithm 1 takes 16 equally spaced ranks after sorting each split by the same conductivity key as the published comparison. The jobs for phase B use the validation subset. The test subset is reserved for the final comparison after model selection. Computing the independent reference load functional on the entire grid does not run a model on test.

On 23 November 2026 the earlier Hyperion smoke job 8166849 ran tasks 0 (validation) and 31 (one test pair, degree 3, bound 15). Both used levels 4 and 8 and at most five iterations. There was no v3 reference, and the recorded relative errors were NaN. These files remain under the original remote nniga_smoke directory and are excluded from selection, summaries and final evaluation. The four exploratory test instances mentioned in the revision prompt are also excluded as sources of results. The earlier NGSolve log ended at pair 174 of 400 without a full-precision reference CSV. The v3 reference is recomputed from the parameter grid.

The Slurm array maps each of 32 tasks to a degree, a bound and four of the 16 validation pairs. An explicit SCAN_SPLIT=test switches the split for the final experiment. Smoke runs require validation and have separate output directories. Each scan checks that the reference contains the requested parameter pair with the same split label. The raw signed squared error is retained even when its square root is unresolved. Timing is cumulative optimizer time along the ladder after JIT warm-up.

### Reference and selection rules

The NGSolve mesh is translated so that the re-entrant vertex is the origin. This preserves the domain, coefficients and loading under translation and keeps the small distances to the origin representable when refining deeply. The corner has hp refinement weight 1 and the other vertices 0.7. The mesh uses the three original materials and the outer Dirichlet boundary. The two cost pilots used orders 14 and 16, factor 0.15 and levels 18 and 20. For the full grid we use levels 20 and orders 16 and 18, and repeat order 18 at level 19. Every parameter has its own order and level differences. The local py312 environment has NGSolve 6.2.2604. Hyperion p1d_v2 is used for optimization; NGSolve is already installed in the independent ngsolve_ref environment.

Before reading validation outcomes, the bound at each degree and level is defined as the smallest tested T for which no larger T reduces the median energy error by 2 percent or more. The floor is one tenth of the smallest physical element realized across the 16 validation instances at that bound, with lower limit 1e-12. Rates use the logarithm base two of the ratio of the median errors at levels 32 and 64. A reference agreement target of 1e-11 in squared relative energy gives a nominal resolution of about 3.2e-6 in relative energy norm. The summary counts observations at or below this threshold and retains all 16 ranks when determining the median.

Three regression cases exercise the smaller spans admitted by the scan. At width 2⁻³⁵ the previous quadratic basis could sum to zero because its denominators contain products of lengths and used an absolute cutoff of 1e-14. The degree-two and degree-three kernels now use the zero-denominator convention for repeated knots. At the same width the Greville snapping tolerance removed valid degrees of freedom. The Dirichlet mask now compares the ends of B-spline supports to the material corner, following the independent conformity test. At an interior point 2e-13 from the interface the previous conductivity mask could return zero. Each open material quadrant now receives its prescribed conductivity up to its boundary. The quadratic first derivative uses differences of local distances to reduce cancellation near the corner.

The two-pair pilots took about 4 s at level 18 and 7 s at level 20 on the Mac (the latter includes the level-19 check). In the corrected Hyperion smoke 8175182, both degrees completed the short ladder with a finite error. Job 8175183 then took one optimizer step at N=128 on validation. Slurm recorded 70 s and 7,315,408 KiB peak RSS for p=2 and 76 s and 7,986,824 KiB for p=3, within the 16 GiB task allocation. The complete validation ladders therefore include N=128.

The full reference completed in 989.963 s with four processes. The largest relative difference was 6.634e-13 between orders and 6.453e-12 between levels. Across the 60 shared v2 pairs the largest change was 1.707e-9. The signed energy gap is positive for all 4,800 stored evaluation rows, with minimum 9.655e-10. Full-precision values are in references_v3/lshape_energy_checks.csv. Hyperion validation job 8175327 uses the verified 400-pair CSV, all four bounds, both degrees, six levels and 16 validation pairs.

### Measuring energy on strongly graded meshes

During the validation scan the matrix evaluation of the Ritz functional and the discrete load ceased to agree at N=128. In the first matrix-only run, degree 3, T=10, validation instance 8 even gave a signed squared error of -1.321e-10. This exceeds the reference resolution. The geometry, optimizer, objective and selection criterion stay fixed across the scan. All reported validation errors use the same element integral of the Ritz functional, including uniform and coarse meshes.

For any conforming field v the squared relative error is [F(u) + a(v,v) − 2F(v)] / F(u), by the weak equation for u. This identity is valid for the actual computed coefficients even when the floating-point solve has lost Galerkin orthogonality. The code integrates F(v) and a(v,v) on the three material patches with p+1 Gauss points per axis, exact for their polynomial integrands. It uses span-local coordinates, derivatives written in terms of adjacent coefficient differences and NumPy long double for the contractions on Hyperion. The tensor-product tests integrate xy over the L with two distinct conductivities and spans of width 2⁻³⁵. The CSV keeps the matrix Ritz energy, the load vector times the coefficients, and both element integrals alongside the reported gap. Comparisons with the load-only estimate diagnose linear-solve and assembly precision.

Quadrature smoke jobs 8175438 and 8175461 completed on validation. Ladders that had already run the older measurement are repeated for their full set of levels because the first files contain no coefficient vectors. Matrix-only output files are excluded from the selection summary. One-instance recovery tasks use CHUNK_SIZE=1 and the same optimization settings. The quadrature rule applies to every p, T, level and instance.

The original negative row is preserved in data_results_v3/lshape_alg1_Tscan/diagnostics/val_p3_T10_i8_matrix.csv. The full 2D suite with the quadrature regression completed with 223 passes, one conditional skip and the expected v1 conformity failure.

### Phase B validation results and first decision

All 32 main array tasks and 65 one-instance quadrature reruns completed. The recovery arrays were 8175480, 8175537, 8175579, 8175701 and 8175747. The original 65 matrix-only ladders were moved on Hyperion to data_results_v3/lshape_alg1_Tscan/assembled_only_8175327. The 128 canonical per-instance CSVs contain 1,536 rows (uniform and adapted for six levels). The final summary reads only the 16 expected validation pairs for each degree and bound, checks the full ladder and recomputes the gaps from the stored integrals.

The median 32-to-64 order is 2.033 for p=2 and 2.963 for p=3 across all four bounds. The smallest sufficient bound under the prespecified 2 percent rule is 5 except for p=3 at N=128, where 8 reduces the median by 9.524 percent compared with 5. At p=3, N=64, the validation pair (0.1, 0.12742749857031338) has relative error 0.00483678 at T=5 and 0.00123223 at T=8. Thus the median decision can differ from the decision for one of the more difficult individual pairs.

At N=32, all four tested bounds give the same median validation error, 0.002098955385865806 for p=2 and 0.0003924586228180104 for p=3. Across paired instances the largest relative change from T=5 to T=8 for p=2 is a 0.00005601 percent increase. For p=3, the pair (0.1, 0.12742749857031338) improves from 0.007844266357131379 to 0.007080353276783956, a 9.73849 percent reduction and the only change larger than 2 percent among the 16 pairs. This pair gives the same N=32 error at T=8, 10 and 15. Three cubic runs at T=5 have an active box constraint at their N=32 iterate. All of these comparisons use the estimator objective of Algorithm 1.

Element integration gives a positive gap on every completed ladder. The same p=3 validation pair at N=128 falls just below the nominal resolution at each of the four bounds. Including its rank affects none of the reported medians. Two assembled-matrix gaps were negative. For p=3, T=10, instance 8 at N=128, the same computed field gives -1.321e-10 with the matrix functional and 2.053e-10 with element integration. At N=128 and p=3, four optimizations (two each at T=10 and 15) took zero iterations after prolongation and stopped in the line search. These rows are included. The reported errors describe the attained iterates.

The per-level median errors, 32-to-64 orders and proposed physical size floors are stored in validation_medians.csv, validation_orders.csv and validation_schedule.csv under data_results_v3/lshape_alg1_Tscan. No network variant has been selected at this checkpoint.

### Phase C. Level dependence and admissibility across the corner

Each half-axis uses the same MLP at reference cell centres. For local coordinate `s=(i+1/2)/(N/2)`, the two distances to the interface are `(1-s)/2` and `s/2`. The fifth input is `log(d+1/N)`. The number of weights stays fixed across N. The checkpoint carries the feature switch and the numerical schedule, including the extrapolation levels. The historical four-input checkpoints retain their original scalar policy. Training, validation, checkpoint restoration and the corrector use the same map. The scheduled Algorithm 1 optimizes bounded logits directly and applies the same affine size floor as the network. Phase B calibrated this floor with a zero-floor scan. The comparison for the paper must use the floor as well as the box.

For a half-axis of length 1/2, put $n=N/2$, $r=\exp(2T)$, $f=h_{\min}$ and $b=1/2-nf>0$. Each softmax weight has the form $w_i=1/(1+\sum_{j\ne i}\exp(q_j-q_i))$. Bounding every exponential between $1/r$ and $r$ gives
\[
 f+\frac{b}{1+(n-1)r}\ \le h_i\le\ f+\frac{br}{r+n-1}.
\]
Within one block the common normalizer cancels and the size ratio is at most $r$. Across two independently normalized equal blocks, dividing the upper endpoint by the lower endpoint gives a uniform bound for every pair, and therefore for the implemented sliding-window $\gamma_{\rm loc}$. A basis touching both sides of the pinned knot can see a larger ratio than $r$. For $p=2$, $N=64$, the regression constructs a width-32 two-hidden-layer network giving adjacent spans of about $8.973\times10^{-7}$ and $0.499292$, with ratio 556461.31. The within-block maxima are 17975.78 and 21867.29. The author approved retaining the block map and correcting the cross-interface bound. The text and caption currently using a global $e^{2T}$ require the distinction at Phase F.

### Phase C. Element quadrature, losses and double precision

On the three physical rectangles, conductivity is constant and the load is one. On each tensor-product cell the stiffness integrands have coordinate degree at most $2p$, and the load has degree at most $p$. The squared strong residual and squared flux traces have coordinate degree at most $2p$. Gauss–Legendre with $p+1$ points per axis is therefore exact for stiffness and strain in exact arithmetic. Four points suffice for the estimator and load for both tested degrees. The estimator uses the squared rectangle diameter divided by the cell conductivity. On each material interface it uses the tangential edge length divided by the larger neighbouring conductivity. The external and cut boundaries carry homogeneous Dirichlet data.

The JAX float64 production path computes the two distances from a quadrature node to the span endpoints directly from the reference node and width. B-spline derivatives of a field use scaled successive coefficient differences. Structural zero spans carry zero integration weight. Half-axis mass, derivative Gram and load integrals assemble the three patch contributions to the Galerkin system. Ritz training now integrates $a(v,v)/2-F(v)$ on these patches. The validation monitor uses the same normalization as each training objective, including the factor $1/2$ for the estimator loss. Parameter and mesh derivatives pass through the Galerkin solve.

The independent audit constructs piecewise B-spline polynomials with rational Cox–de Boor recursion and integrates monomial products exactly on each cell and each interface. It exercises two non-symmetric meshes, both degrees, distinct conductivities and a nontrivial conforming coefficient field. The thinner mesh has minimum width $2^{-32}$. The tests compare every physical element matrix and vector, the assembled free system, the two face integrals individually, the squared estimator and the Ritz functional. Matrix and load discrepancies use the largest absolute entry difference divided by the largest reference entry of each element or assembled block. Scalar discrepancies are relative. A separate comparison evaluates Ritz on the actual Galerkin coefficients against the host quadrature used in Phase B and records the normwise backward error of the solve. These are integration and backward stability measurements on specified inputs. Forward accuracy still depends on conditioning, and resolution of a relative energy error still depends on the NGSolve gap.

The Mac and Ice Lake outputs are in `data_results_v3/lshape_pilot/integration_audit{,_icelake}.csv`. Across both platforms the largest local stiffness and load discrepancies are $1.60\times10^{-15}$ and $1.30\times10^{-15}$ (rounded upwards). The largest estimator and manufactured-field Ritz discrepancies are $7.48\times10^{-16}$ and $5.23\times10^{-16}$. The independent Ritz check on solved coefficients differs by at most $2.69\times10^{-15}$. The float64 loss gradients are also checked against central differences through multiplicity $p$ for each objective and degree.

### Phase D. Validation protocol and provenance

The seed-zero pilot uses 280 training and 60 validation parameters with split seed 20240601. V0 loads the historical final N=32 checkpoint. V1 adds the corner-distance input, V2 also uses the approved physical floor and saturation, and V3 also trains at N=64. Each variant evaluates its single final checkpoint at both N=32 and N=64 on all 60 validation parameters. The row files store the parameter pair, reference, the load and strain of the attained field, the signed gap, the mesh ratio and the checkpoint SHA-256. Selection compares median relative energy error at both levels, with a 5 percent equivalence band favouring the simpler variant. The reported order is the base-two logarithm of the ratio of medians. The suite's historical V0 mesh-invariant regression previously swept the holdout coordinates and the full grid. It only checked knot geometry. It now sweeps validation and training coordinates. All pilot energy evaluations use validation exclusively.

Arrays 8179166 (eight smoke tasks), 8179175 (V0) and 8179174 (two full-size timing probes) use the Ice Lake partition constraint and four cores. Audit 8179374 independently checked the production quadrature on Ice Lake. The timing probe uses the full 280/60 split, batch size 16, one cold epoch including JIT and a second epoch with warm executables. Its checkpoints are isolated from pilot training. The historical training time reported alongside V0 belongs to the supplied checkpoint. Its additional pilot training cost is zero.

For reviewer 2, comment 2 requires a paired comparison of both network losses and both Algorithm 1 objectives on the same mesh policy. Comment 4 requires the conforming cut-boundary treatment, independent reference and an error identity valid on the computed field. Comment 6 requires same-hardware offline, online, memory and fixed-accuracy measurements. Validation results guide the model choice. The publication tables and response require the four-seed and held-out measurements at the later approval gates.

The full 2D suite, including slow tests, completed with 232 passes, one conditional NGSolve skip and the expected negative control for the old nonconforming reference. The comparison driver was also exercised at N=4 and 8 for both Algorithm 1 losses with the scheduled floor, the v3 reference and one validation pair. The short run checks the driver and has no role in model selection.

The warm full-split N=64 epochs on the Xeon Platinum 8368 took 339.20 s and 342.17 s for degrees 2 and 3. `timing_projection.csv` combines these measured values with the historical 8362 seed-zero prefix cost and its final trained level's epoch count as proxies. The two central projections are 15.88 h and 33.18 h. With 400 N=64 epochs they become 39.34 h and 42.02 h, holding the prefix estimate fixed. The threshold crossing for degree 3 was reported before submission. The Ice Lake constraint matches both 8368 and 8362. Publication timings require verifying the 8362 CPU model for each comparison job.

The author approved keeping the 400-epoch cap for degree 3, including the 42.02 h scenario. Production pilot array 8179727 runs V1 and V2 for both degrees and V3 for degree 2. Array task 8179732_7 runs V3 for degree 3. Each output directory is distinct from the V0, smoke and timing paths.

For interpreting the ablation, all scheduled pilot levels through N=64 have T=5 in both degrees. Consequently any V2–V1 difference through N=64 is attributable to the change in the floor and its effect on training, within seed-zero variability. V3–V2 additionally changes the finest trained level. The increase to T=8 concerns p=3 at N=128. A claim that increasing T caused an aggregate gain at N=64 would require an additional contrast. The Phase B comparison across T already separates median and individual-instance behaviour.

In Experiment 4, N=64 is a zero-shot resolution for the original checkpoint and for V1 and V2: training updates their neural weights only at N=4, 8, 16 and 32. The pilot uses validation at N=64 for model comparison and Phase B calibrated V2's level policy there. The term zero-shot in resolution refers specifically to weight fitting. V3 additionally updates weights at N=64. Every variant still predicts the mesh for a new conductivity pair with a frozen field evaluated on the two blocks of each axis, followed by the Galerkin solution. The V2–V3 pair isolates the additional training level with the same architecture and policy. An extrapolation claim for V3 at N=128 would require a network evaluation there.

### Phase D decision at the author's zero-shot stop

The author prioritized extrapolation from N=32 to N=64 and explicitly requested cancellation of V3 for degree 3. Slurm records 8179732_7 as cancelled after 70,754 s (19 h 39 min 14 s). The final logged minibatch was iteration 3650 of zero-based epoch 202. The trainer saves its selected checkpoint at the end of each level. The preserved N=32 checkpoint matches V2 for degree 3 byte for byte, with SHA-256 987510befe88b5cb645c8116f2fbf1e1cd920deb3cb10fcbd0f15740acc973c6. There is no final N=64 model or paired validation result for this run. The status row has empty accuracy fields. The completed V3 result for degree 2 remains a trained-resolution control.

The seven completed variant/degree pairs yield 840 observations, each with a reference from validation. Each median uses all 60 validation pairs and every gap lies above the nominal reference resolution. The full rows and checkpoint hashes are under `data_results_v3/lshape_pilot/V*/p*/`. The full and zero-shot summaries are `validation_summary.csv` and `validation_summary_V0_V1_V2.csv`. They can be regenerated with `python 2D/scripts/summarize_lshape_pilot.py` and the same command with `--variants V0 V1 V2`. The recorded training durations cover the whole ladder. V0 uses the historical duration and zero additional training. Costs here describe the pilot with its actual nodes. Phase E requires matched hardware timings for the publication.

| Variant | Degree | Median error at 32 | Median error at 64 | Order | Median gamma at 32 / 64 | Train time (h) |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 2 | 0.00224542 | 0.000841697 | 1.416 | 18.30 / 5.75 | 0.751 |
| V1 | 2 | 0.00216709 | 0.000619179 | 1.807 | 19.01 / 8.35 | 0.768 |
| V2 | 2 | 0.00216552 | 0.000618134 | 1.809 | 19.61 / 8.46 | 0.706 |
| V0 | 3 | 0.000440390 | 0.000119588 | 1.881 | 124.08 / 17.36 | 1.990 |
| V1 | 3 | 0.000422197 | 0.0000764152 | 2.466 | 157.25 / 31.34 | 2.361 |
| V2 | 3 | 0.000420859 | 0.0000772151 | 2.446 | 156.28 / 30.83 | 2.111 |
| V3 (trained at 64) | 2 | 0.00213130 | 0.000545748 | 1.965 | 31.56 / 17.65 | 7.726 |

The proposed zero-shot variant for Phase E is V1 for both degrees. Relative to V0, its median N=64 validation error falls by 26.44 percent for degree 2 and 36.10 percent for degree 3. At N=32 the decreases are 3.49 and 4.13 percent. All four V1–V2 median differences are within 1.05 percent, below the 5 percent equivalence threshold. V1 uses the simpler scalar policy. For degree 2, training at 64 reduces V3's N=64 median by a further 11.86 percent relative to V1 at a total training cost of 7.73 h. The cancellation changes the scope of selection to the three zero-shot configurations. These numbers support a seed-zero validation decision. Four-seed estimator and Ritz training, the paired held-out Algorithm 1 comparison and the responses for publication await Phase E approval.

### Phase E. Frozen experiment and units of analysis

The author approved V1 for both degrees before the final experiment. Each loss trains four independent initializations (seeds 0 to 3). All sixteen runs use the same split, level-dependent corner-distance input and levels 4, 8, 16 and 32. Each physical half has a minimum span of 1e-7 and the logits use T=5. The original patience, per-level cap and objective-specific improvement tolerances apply. Seed zero is trained on the publication hardware alongside the other seeds. Test N=64 uses the frozen N=32 weights, with no corrector, optimization step or checkpoint choice on test. N=64 validation participated in the earlier pilot decision.

The comparison uses the order-18, level-20 reference. Each network supplies five records for each of the 60 test pairs. The deterministic uniform mesh supplies one record per pair, degree and level. Conditional on the preselected 16 test pairs, Algorithm 1 optimizes each objective separately with T=5 and a physical floor of 1e-7 at every level, including N=64. Network values for this comparison are taken from the same 60-pair evaluation. The four-bound scan deferred from Phase B uses its original block map and six-level ladder in its own directory. It measures sensitivity of the per-instance optimizer and supplies no model-selection inputs.

The error uses the strain and load integrated independently on the computed spline field. The effectivity denominator is the absolute energy error, the relative error multiplied by the square root of the reference load. Each observation carries the signed squared gap and a resolution flag. The aspect ratio uses the three physical quadrants. The condition numbers apply to the free stiffness matrix and its symmetric Jacobi scaling. Extremal eigenvalues are computed directly for small matrices, or by sparse symmetric iteration with a shift-invert solve for the smallest eigenvalue. The fixed-accuracy table uses the paper's log-log interpolation and nominal tensor-product DOF ratio. It also records the first measured level attaining the target and the exact free-DOF ratio at that level.

Network timing is a synchronized forward mesh prediction plus Galerkin solve after a synthetic warm-up per resolution. The estimator, error and spectral diagnostics are outside that timer. Algorithm 1 timing sums the measured L-BFGS-B optimization intervals up to each level after compilation. Cost comparisons use identical parameter coordinates on the Platinum 8362, with four CPU cores. The summary records both the median and largest process peak RSS over the four training seeds, in MiB. Network rows retain their checkpoint SHA-256 and are saved atomically per parameter and level. Completed test rows can be read on restart. Figures use the frozen estimator seed-zero checkpoints and the original illustration parameters.

Array 8210968 exercised all four degree/objective training paths. Array 8210973 exercised all four network evaluation paths and both uniform controls. Array 8210974 exercised both optimizer objectives for each degree. The twelve smoke tasks completed with exit code zero on validation. Their two-level, low-budget runs are isolated from publication measurements.

The mesh-and-solution template currently reads the adapted heat map under the uniform knot overlay and the manuscript caption orders the second conductivity pair differently from the generator. The v3 generator saves the solution on each mesh and a parameter manifest. Its param2 is (0.1, 10) and param3 is (10, 0.1). At the manuscript gate the lower panels can reference param3 to match the existing caption, and the uniform panels can read their own solution fields. Knot-line coordinates preserve distinct physical spans at full binary64 text precision.

The full local 2D suite, including slow tests, passed 236 tests with one conditional skip and the expected failure for the historical nonconforming reference. The two figure drivers were exercised independently with pilot weights. All six pairs of adapted and uniform fields were finite on the 81 by 81 plotting grid, differed on the physical domain and vanished inside the removed quadrant.

The first production training allocation (8211172) used the original Ice Lake feature. Tasks 0, 1 and 2 started on Platinum 8362 nodes. The other thirteen allocations exited with code 42 on Platinum 8368 before launching Python. Only successful 8362 measurements contribute to the final tables.

The precision table records the normalized discrepancy between the production Ritz value and the independent host strain/load quadrature, and the Galerkin defect measured by the normalized difference between strain and load. For each signed gap g it also perturbs the reference load by a relative ±1e-11 and tabulates the resulting median error interval, using the squared error (g+δ)/(1+δ). The resulting conservative improvement-factor interval compares identical full-test populations. This is a sensitivity analysis at the agreed reference tolerance.

The author approved excluding the 8368 group, hyperion-[225-233], and checking the CPU model inside each allocation. The other thirteen training tasks use array 8211266. Arrays 8211280 (networks and uniform), 8211287 (paired estimator/Ritz optimization with the V1 floor) and 8211290 (deferred four-bound scan) require successful completion of tasks 0, 1 and 2 of 8211172 and every task of 8211266. All three evaluation submissions carry the same Ice Lake constraint and 8368 exclusion.

### Phase E. Held-out numerical evidence

The full sixteen-run training finished at 23:00:40 Madrid time on 24 November. All 18 network and uniform evaluation tasks completed by 23:02:16 and the last of the 32 paired Algorithm 1 tasks completed at 23:10:23. The postprocessor checked 5400 network/uniform rows and 320 per-instance observations against the held-out split, exact reference, physical free-DOF count, CPU, checkpoint provenance, energy identity and effectivity. Every signed squared gap exceeded 1e-11. The complete numerical tables, illustrative fields and links to the primary data are in [the Phase E test report](../data_results_v3/lshape/STOP_fase_E.md). The final comprehensive 2D test run passed 237 tests with one conditional skip and the expected historical failure.

The separate six-level, four-bound sensitivity run (8211290) exhausted its 16 GiB limit at the finest resolution. It produced zero completed per-instance files. A validation probe with the Phase C integration kernels (8248119) demonstrated the larger memory requirement at N=128 for each degree. Its process peaks were 28,207,332 and 29,954,716 KiB according to Slurm. The one-instance recovery array 8248172 requests 96 GiB and keeps the reference, parameters, four CPU cores, processor model, objective, bounds, ladder and stopping settings. The completed V1 test cases and the paired comparisons are read from their original outputs.

### Completed bound sensitivity and manuscript revision

All 128 single-instance jobs completed on 25 November 2026. The last finished at 12:53:37 Madrid time. The diagnostic includes 768 estimator optimizations and 768 uniform controls on four Platinum 8362 cores. The 16 paired test parameters are fixed for all four bounds. At N=32 all four bounds give identical median errors for each degree. The cubic case (0.1, 0.206914) improves by 3.67 percent above T=5. At N=64 individual errors improve by as much as 30.8 percent for quadratic and 76.0 percent for cubic splines. The medians agree to four significant digits. At N=128 the cubic median drops by 25.9 percent between T=5 and 8. One cubic parameter at N=128 falls below the reference resolution for all four bounds. Interval medians keep those ranks and all 48 medians are resolved. The diagnostic uses zero floor and does not change the validation-selected V1 policy. [Test medians](../data_results_v3/lshape_alg1_Tscan/test_medians.csv) · [Paired results and test tables](../data_results_v3/lshape/STOP_fase_E.md)
