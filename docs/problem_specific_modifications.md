# Departures from the generic method, per experiment (draft for main_v2.tex)

**The generic method.** It is the same in every experiment:
- **Network:** a tanh MLP with inputs (ν, ξ) at the cell midpoints, gauge fix and bounded logits T·tanh(z/T).
- **Mesh map:** a softmax of the sizes with an h_min floor.
- **Loss (eq. 26):** η²(θ_φ(ν); ν)/(η²_unif^{(N)}(ν) + ε), with the residual estimator and the weights ρ_E of the theory.
- **Training:** coarse-to-fine continuation and the validation protocol, with a 70/15/15 split, early stopping on the validation loss, Adam and 4 seeds.

The table lists every departure from it. "Loss" says whether the departure changes the training objective. "Map" says whether it changes the mesh map or the network input. The last column gives the ablation evidence where it exists.

| Exp. | Departure | Purpose | Loss | Map | Evidence (revision v2) |
|---|---|---|---|---|---|
| 1 | Level-dependent cap T(p, N) (p=2: 2 up to N=32, 3 at 64, 5 at 128–256; p=3: 6, then 7 at 256) | Allow stronger grading at fine levels | no | yes | not ablated |
| 1 | Extra input feature log(ξ + 1/N) | Resolve the x^β grading toward x = 0 | no | yes | no effect for p=2; p=3 worse without it at N ≤ 16 (N=2: 6.5e-3 vs 4.0e-3), equal for N ≥ 32 (`data_results_v2/singular_nolog`) |
| 1 | Level-dependent h_min (1e-7 for N ≤ 32, else 1e-8, p=2) | Floor below the expected smallest element | no | yes | not ablated |
| 2 | Global mesh map with the interface knot x_I inserted with multiplicity p, free element split across x_I | Move elements into the short-wavelength layer | no | yes | — (the split is the adaptation itself) |
| 2 | Nyquist cap h_e ≤ 2π/(2.5 k_loc) by differentiable water-filling | Avoid under-resolved oscillations on coarse meshes | no | yes | p=2 neutral (factor 2.74 vs 2.79 at N=32, equal for N ≥ 64); p=3 small gain (3.64 vs 3.25 at N=32, 2.55 vs 2.41 at N=256) (`data_results_v2/helmholtz_nohmax`) |
| 2 | Coefficient-independent estimator weights h_E², h_B (instead of h_E²/σ_E, h_B/σ_B) | Simpler; σ is piecewise constant and O(1) | yes (weights) | no | η changes by ≤ 13 % on the trained meshes (manuscript) |
| 2 | Contrast sampling excludes ±0.02 bands around the resonances | Avoid near-singular training samples | no (data) | no | not ablated |
| 2 | Warm phase: 60 epochs on the top third of contrasts at lr 4e-3, at the coarsest level | Escape the uniform-mesh stall at high contrast | no (schedule) | no | not ablated |
| 2 | Input "side" flag (left/right of x_I) | Let the network distinguish the two layers | no | yes | not ablated |
| 3 | Input "axis" id (one network for both directions) | Two knot vectors from one network | no | yes | — (structural) |
| 4 | Four knot-spacing blocks (two per axis), interface knot 1/2 of multiplicity p fixed | Conforming three-patch C⁰ space with exact Dirichlet data (Lemma 5.1) | no | yes | — (structural) |
| 4 | Input axis id ∈ {0, 0.5, 1, 1.5} | One network for the four blocks | no | yes | — (structural) |
| 5 | Wider network (3 × 64) | Capacity for the thin boundary layer | no | yes | not ablated |
| 5 | Estimator weight h_E²/ε applied to the reported I_eff after the run | Match eq. (12) | no: ε is constant per sample, so the eq-26 ratio is unchanged | no | — |
| 6 | Inputs (log₁₀ ε, b, ξ) mapped to [−1, 1] | Conditioning of the input layer | no | yes | — (no problem-specific feature) |

Experiments 3, 4 and 6 use no problem-specific feature, cap schedule or loss change beyond the structural items listed. All 2D runs used lr 1e-3 → 1e-4, while the 1D runs used the 1e-2 → 1e-4 of `config.VAL` (`PROVENANCE.md`).
