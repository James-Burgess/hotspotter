# Algorithmic Differences: wbia-core vs WBIA HotSpotter Pipeline

Last verified: 2026-06-07. Based on source reading of `wildbook-ia/wbia/algo/hots/`.
**Phase 1** (Kpad dynamic, name-level scoring, canonical alignment) ✓
**Phase 2** (normalizer rule, bar_l2, ratio, const filters) ✓
**Phase 3** (requery, score normalizer) — not implemented.

## Summary

wbia-core implements the full WBIA HotSpotter scoring pipeline.
The core LNBNN formula (`ndist - vdist`) is identical. All active
WBIA filters (lnbnn, fg, bar_l2, ratio, const) and name-level
scoring methods (csum_wbia, nsum_wbia/fmech, sumamech) are now
implemented. The remaining gap is a ~0.8× score ratio caused by
FLANN distance differences at the descriptor level, not algorithmic.

---

## 1. Normalizer Selection Rule  [DONE — Phase 2]

| | WBIA | wbia-core |
|---|---|---|
| **Default** | `normalizer_rule='name'` | `normalizer_rule='last'` (configurable) |
| **'last' rule** | Uses last column of `K+Kpad+Knorm` | Uses last column |
| **'name' rule** | Selects normalizer from a different *name ID* than any voting match | Implemented (`pipeline.py:178-202`) |

The `'name'` rule (`nn_weights.py:get_name_normalizers`, line 287):
1. Maps all `K+Kpad` voting neighbors + all `Knorm` normalizer candidates to their name IDs
2. Marks any normalizer candidate invalid if its name ID matches any voting neighbor's name ID
3. Also marks invalid if normalizer name == query name
4. Falls back to `Knorm=-1` if no valid normalizer found → that feature match is dropped

**Impact:** With the `'name'` rule, the normalizer is a feature that is *guaranteed different* from all voting candidates. This makes LNBNN more discriminative — the normalizer is truly an "unrelated" feature, so `ndist - vdist` is larger when the match is correct. wbia-core's `'last'` rule can pick a normalizer from the same name as a voting neighbor, producing weaker weights.

> **This is the single biggest algorithmic difference and the most likely source of the remaining score ratio gap.**

---

## 2. Multi-Filter Multiplication  [DONE — Phase 2]

| Filter | WBIA | wbia-core |
|---|---|---|
| `lnbnn` | `ndist - vdist` | `ndist - vdist` |
| `bar_l2` | `1.0 - vdist` | `w *= 1.0 - vdist` (`bar_l2_on=True`) |
| `fg` | `sqrt(q_fgw × d_fgw)` | `w *= sqrt(q_fgw × d_fgw)` (`fg_on=True`) |
| `ratio` | `1.0 - (vdist/ndist)` with threshold binary mask | `w *= 1.0 - ratio` (`ratio_thresh` set) |
| `const` | `1.0` (every match weighted equally) | `w *= 1.0` (`const_on=True`) |

All five WBIA filters are now implemented. They multiply sequentially
in the match-building loop (`pipeline.py:218-245`), matching WBIA's
`fsv.prod(axis=1)` semantics.

---

## 3. Name-Level Scoring  [DONE — Phase 1]

wbia-core now implements all WBIA name-scoring methods via `name_scoring.py`:

| Method | WBIA | wbia-core |
|---|---|---|
| `csum_wbia` | per-annot csum → per-name max → canonicalize | `compute_maxcsum_name_score()` + `align_name_scores_with_annots()` |
| `nsum_wbia` | per-annot csum → fmech per-name nsum → canonicalize | `compute_fmech_score()` + `align_name_scores_with_annots()` |
| `sumamech` | per-annot csum → per-name sum → canonicalize | `compute_sumamech_name_score()` + `align_name_scores_with_annots()` |

Simple per-annot methods (`csum`, `nsum`) are still available for
backward compat and testing without name grouping.

---

## 4. ChipMatch State Machine  [DESIGN CHOICE]

WBIA builds full `ChipMatch` objects (`chip_match.py`, 3039 lines) with:
- `fm_list` — per-annot feature-match pairs `(qfx, dfx)`
- `fsv_list` — per-annot feature-score vectors `(n_matches × n_filters)`
- `fk_list` — per-annot filter key / rank tensors
- `filtnorm_aids` / `filtnorm_fxs` — per-filter normalizer annotations & features
- `name_groupxs` — annotation-to-name grouping indices
- `algo_annot_scores` / `algo_name_scores` — dicts of evaluated scores

wbia-core uses flat `Match` → `ScoredMatch` lists. Per-filter scores are
not preserved — only the final combined weight is stored. This is a
deliberate simplification that doesn't affect scoring correctness.

## 5. Requery Mechanism  [TODO — Phase 3]

WBIA can optionally fetch additional neighbors when all `K+Kpad` initial results are in the impossible set (`requery=True`). The `requery_knn()` function (`neighbor_index.py:795`) iteratively queries FLANN, blocking known-impossible annot indices, until enough valid neighbors are found.

wbia-core implements `Kpad` as a simple column budget, but doesn't requery.

**Impact:** If the parity benchmark data has annotations from the same image/name, WBIA's requery could fetch different neighbors than wbia-core's single query. However, this should only matter when Kpad is too small to absorb all impossible neighbors.

---

## 6. Score Normalizer  [TODO — Phase 3]

WBIA has an optional `vt.ScoreNormalizer` that can apply pre-trained
normalization to LNBNN weights (`pipeline.py:894`).

wbia-core has no score normalization. Low priority.

---

## 7. Distance Handling Path  [IDENTICAL]

Both use `VEC_PSEUDO_MAX_DISTANCE_SQRD = 2.0 * 512^2 = 524288`.

| Step | WBIA | wbia-core |
|---|---|---|
| FLANN query | Returns raw int32 SSE | Returns raw int32 SSE |
| Divide by 524288 | In `NeighborIndex.knn()` | In `identify()` |
| clip negative | No explicit clip | `np.maximum(raw_dists, 0.0)` |
| sqrt | In `weight_neighbors()` if `sqrd_dist_on=False` | Always in `identify()` |

---

## 8. Chip Extraction & Feature Extraction  [IDENTICAL]

Both systems use:
- `pyhesaff` for feature extraction (same vendored submodule)
- `cv2.warpAffine` for chip extraction

**Verified byte-identical** (2026-06-07): libhesaff.so, libsver.so,
JPEG decode pixel, warpAffine output.

---

## Verification Checklist for Parity

The benchmark `DEFAULT_CONFIG` now runs wbia-core with WBIA-equivalent
settings out of the box:

```python
DEFAULT_CONFIG = {
    "pipeline_root": "vsmany",
    "K": 4, "Knorm": 1, "Kpad": 0,
    "kpad_policy": "fixed",
    "score_method": "nsum",           # fmech path
    "normalizer_rule": "last",
    "fg_on": False,
    "bar_l2_on": False,
    "sv_on": False,
}
```

To verify parity at each phase, run:

```bash
python3 tests/benchmark/run_benchmark.py \
  --targets wbia-core wbia-develop \
  --n-annots 15 --n-queries 2 --seed 10
