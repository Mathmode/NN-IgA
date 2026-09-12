# L-shape reference cache

`reference_lshape.pkl` is the supplied immersed IGA reference used by the L-shape
H1 metric. It is included because evaluation depends on it (60,162,728 bytes).
The historical provenance describes degree 5, corner grading and a 20 by 20
parameter grid; these are not newly certified accuracy claims.

SHA-256: `2885495982e617a6049bcb9284b838cb8260cf6d32166487e0125728d31a27c3`

The retained regeneration command is:

```bash
python 2D/scripts/generate_reference_lshape.py --out references/reference_lshape.pkl
```

Run in a disposable copy; this replaces the cache and can require substantial
memory and runtime. `--spot-check` still performs large solves. The generator
uses the project's own IGA solver and does not require NGSolve. No missing AMR,
conforming or Ritz generator is advertised. Only load this known local pickle;
it is a Python serialization artifact, not a general-purpose exchange format.
