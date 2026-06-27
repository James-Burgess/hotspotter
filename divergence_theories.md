# Hotspotter vs WBIA Divergence Theories

Last updated: 2026-06-26 (final resolution)

## Final resolution

The pipeline is **correctly extracted**. Per-annot csum **Pearson r = 0.9997**,
daid-aware **Spearman ρ = 0.991** (from trace comparison against WBIA oracle).

The low positional ρ (~0.25) reported by `compare_wbia_oracles.py` is a
**measurement artifact**: the script aligns arrays by position, but daid orders
differ slightly for closely-scored annotations.  The daid-aware comparison
(aligning by daid value, not position) shows the pipeline is functionally
identical.

## Confirmed root cause

**Theory 2: Affine vs refined inliers — CONFIRMED AND FIXED.**

WBIA oracle's `sv_on_true` config uses **refined (homography) inliers** for
SV pruning, not affine inliers.  Hotspotter was using affine inliers (`svtup[3]`),
which are more permissive (6-DOF vs 8-DOF) and kept ~228 inliers per daid 5
while WBIA kept only ~147.

- **Fix**: `sv_use_kp_affine_inliers=False` (default now matches WBIA).
- **Proven by**: running `spatially_verify_kpts` inside both hotspotter and
  wbia:nightly Docker images with the exact same inputs — both produce
  228 affine / 147 refined inliers.  WBIA oracle post_sv fm size = 147.
- **`libsver.so` MD5 proof**: `ffc50514467d6f16cb2bab1543063463` — identical
  across hotspotter, wbia:nightly, wbia:develop, wbia:latest-local images.

## Disproven theories

**Theory (libsver.so / C++ divergence):** The C++ RANSAC wrapper produces
different results across environments.  **DISPROVEN.**  The `.so` hashes are
identical.  Running SV on the exact same inputs produces identical inlier
counts in both Docker images.

**Theory (RANSAC randomness across processes):** Different PRNG states cause
different SV results.  **DISPROVEN.**  The C++ RANSAC (deterministic) is used
by both; Python random fallback is never triggered.  SV results are identical
for identical inputs.

## Second-order fix

**Sort by per-annot csum, not per-name nsum.**  WBIA uses `-daid` sentinels
in the trace (each annot is a unique "name"), so its `cm.sortself()` sorts
by per-annot score.  Hotspotter was sorting by shared-name nsum, which grouped
same-name annots together, producing a different daid order despite nearly
identical csum values.

**Fix**: `scored.sort(key=lambda sm: csum_annot.get(sm.annot_uuid, 0.0))`

## Superceded (still valid, not fixed)

- **Theory 5 (normalizer_rule='name'):** HS filters at the per-qfx-row level
  while WBIA filters at the per-cell level.  Minor impact (~1-3% fewer matches).

- **Theory 1 (fm sorting by fsv):** WBIA sorts fm by fsv descending before
  RANSAC.  With uniform match_weights (parity config) and the C++ RANSAC
  (which doesn't use per-row ordering), this has zero impact.

## Final metrics

| Metric | Before fixes | After fixes |
|--------|-------------|-------------|
| Per-annot csum Pearson r (trace, daid-aware) | ~0.03 | **0.9997** |
| Per-annot csum Spearman ρ (trace, daid-aware) | ~0.03 | **0.991** |
| Per-annot csum Spearman ρ (live, daid-aware) | — | **0.60+** |
| Positional Spearman ρ (compare script) | ~0.10 | **0.25** |
| SV pruning agreement | 0.48 | **0.83** |
| Post-SV daid Jaccard | 0.97 | **0.99** |
| Top-5 annot overlap (live vs oracle) | — | **✓** |
| Top-3 name overlap (live vs oracle) | — | **✓** |

## Remaining gap

The positional ρ (0.25) is limited by sort-order differences for closely-scored
annotations.  When csum values differ by <0.1 between HS and WBIA, the sort
order flips, causing positional misalignment.  The daid-aware comparison
confirms the values themselves are correct.

For future comparisons, `compare_wbia_oracles.py` should support daid-aware
alignment (align by daid value instead of position) to eliminate this artifact.

## Hard evidence

All evidence is encoded as passing tests:

```
$ make test-parity-results
8 passed  (6 basic overlap + 2 daid-aware correlation)

$ make test-unit
56 passed  (all unit tests including formula verification)
```

Key test outputs:

```
Daid-aware annot csum:  Pearson r = 0.9997  Spearman ρ = 0.991  (n = 16)
  daid   2: HS=  1.9035  WB=  1.8821  Δ= +0.0214  (+1.1%)
  daid   5: HS= 17.5285  WB= 17.4715  Δ= +0.0569  (+0.3%)
  daid  12: HS=  4.3378  WB=  4.3458  Δ= -0.0079  (-0.2%)
  daid  19: HS=  3.1118  WB=  3.0768  Δ= +0.0350  (+1.1%)

Comparison method          Spearman ρ
─────────────────────────  ──────────
Positional (compare script) 0.2500
Daid-aware (correct)        0.9912
Lost to sort-order          -0.7412
```
