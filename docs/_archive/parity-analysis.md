# Parity Analysis: wbia-core vs WBIA HotSpotter

## BREAKING FINDING (2026-06-08d): Historical gap was benchmark artifact

The previous 75% vs 91.7% top-1 gap reported below was caused by **two benchmark bugs**
in the wbia-develop target runner (`targets/wbia.py`):

1. **Wrong score field**: `normalise_wbia_result` read WBIA's pre-nsum
   `annot_score_list` (raw per-annot csum) instead of the post-nsum `score_list`
   (canonical name-aligned). WBIA was evaluated on raw csum accuracy while
   wbia-core was evaluated on nsum/fmech accuracy — different scoring methods.

2. **Missing annotation names**: `run_query()` never passed `annot_name_list`
   to WBIA's `/api/annot/json/`, so WBIA auto-generated 50 unique names for
   50 annotations. `name_groupxs` had 50 groups of size 1, making fmech
   degenerate to per-annot csum with zero cross-annot aggregation.

**After both fixes** (seed=420, 25 annots, 12 queries):
- Top-1 agrees on 11/12 queries between targets
- Scores match to within 1% (e.g. Q0: 70.72 vs 70.58)
- Remaining discrepancies are sub-1% descriptor-level distance differences

The historical benchmark data below documents the investigation path but is
**not valid for current parity assessment**.

---

## Historical Results (2026-06-08 — 51 annots, 12 queries, seed=122) ⚠️ INVALIDATED

These numbers include the benchmark artifact described above. Do not use.

| Metric | wbia-core | wbia-develop |
|---|---|---|
| Top-1 accuracy | **75.0%** (9/12) | **91.7%** (11/12) |
| MRR | 0.814 | 0.958 |
| Top-3 overlap | 0.444 | — |
| Spearman ρ (mean) | **0.735** | — |
| Spearman ρ (range) | [0.413, 0.944] | — |
| Score ratio | ~3-6× higher | ~0.2-0.3× |

wbia-core returns canonical per-individual results (collapsing same-name annotations).
WBIA develop returns all per-annotation results. The ranking granularity mismatch limits
direct Spearman comparison.

### Top-1 misses (3 of 12)

| Query | wbia-core top-1 | WBIA top-1 | Notes |
|---|---|---|---|
| Q2 | coco-annot-1948 | coco-annot-6189 | Different individual |
| Q3 | coco-annot-5493 | coco-annot-5891 | Different individual |
| Q10 | coco-annot-1387 | coco-annot-5891 | Different individual |

### Score distribution

wbia-core scores are consistently higher (3-6×) because canonical alignment
assigns all same-individual annotation scores to the top representative.

---

## Dual-agent audit (2026-06-08): New high-impact gaps found

Two parallel explore agents audited both codebases. 10 new HIGH/MEDIUM
gaps identified beyond the documented Phase 1-5 roadmap:

| # | Gap | Impact | Details |
|---|---|---|---|
| 1 | **`use_chip_extent` normalization** | HIGH | wbia-core normalizes `xy_thresh` by image max dimension; WBIA uses chip diagonal length from `ibs.get_annot_chip_dlensqrd()`. Different scale → different keypoint pairs pass SV pre-filter. |
| 2 | **FG weight computation** | HIGH | wbia-core uses centered gaussian on full image dimensions (`scoring.py:115-120`); WBIA uses CNN/RF `probchip` on cropped annotation chip at `fw_dim_size=256`. FG values are fundamentally different. |
| 3 | **Spatial verification entirely different** | HIGH | WBIA uses custom `vt.spatially_verify_kpts()` with dual affine+homography, `full_homog_checks` (exhaustive repeated sampling), feature-weighted RANSAC. wbia-core uses single-pass `cv2.findHomography(RANSAC)`. |
| 4 | **Kpad self-padding missing** | HIGH | WBIA pads K+1 when query is in DB even with `use_k_padding=False` (`pipeline.py:369`). wbia-core's `kpad=0` loses a voting column. WBIA fetches 6 cols, wbia-core fetches 5. |
| 5 | **`full_homog_checks` missing** | MEDIUM | WBIA does exhaustive repeated-sampling homography consistency check (`Config.py:284`). wbia-core does single-pass RANSAC. |
| 6 | **`weight_inliers` mechanism differs** | MEDIUM | WBIA biases RANSAC sampling toward high-FG features (`pipeline.py:1518`); wbia-core applies post-hoc `score *= (1 + 0.5 * inlier_ratio)`. Same config flag, different behavior. |
| 7 | **`sv_ori_thresh` default differs** | MEDIUM | wbia-core=`None` (disabled); WBIA=`TAU/4 ≈ 1.57` (90° filter). Different correspondence counts pass SV. |
| 8 | **Database feature filtering query-only** | MEDIUM | wbia-core filters only query features by `minscale`/`maxscale`/`fgw_thresh`; WBIA filters both query AND database before index build (`neighbor_index.py:69-95`). |
| 9 | **`can_match_sameimg` missing** | MEDIUM | WBIA excludes same-image annotations from voting (`pipeline.py:325-349`); wbia-core has no same-image concept in `AnnotatedImage` data model. |
| 10 | **SV shortlist uses wrong scoring** | MEDIUM | wbia-core builds SV shortlist from `score_method`; WBIA from `prescore_method` (`pipeline.py:1364`). Different candidates get verified. |

Additional LOW-impact findings: `sver_output_weighting` missing, `refine_method` hardcoded,
`lograt_on`/`cos_on` toggles absent (dead code in WBIA), empty feature edge case, float32 vs
float64 numerical differences.

---

## Progress history

| Date | Milestone | Mean ρ | Top-1 | Notes |
|---|---|---|---|---|
| 06-04 | OpenCV SIFT baseline | -0.15 | 0% | Full-image features, 50× scores |
| 06-05 | pyhesaff switch | 0.71 | 0% | Feature extractor parity |
| 06-06c | Chip extraction | 0.27 | 33% | 450px width, crop+resize |
| 06-06d | dim_size=700/maxwh | 0.29 | 33% | Matching WBIA chip config |
| 06-06e | warpAffine | 0.32 | 33% | Exact pixel match with WBIA |
| 06-06f | sqrt distances | 0.33 | 33% | Distance normalization fix |
| 06-07 P1 | Name-level scoring | 0.987 | 100% | fmech + canonical (15 annots, 2 queries) |
| 06-07 P2 | Filter completeness | 0.943 | 100% | bar_l2, ratio, const, normalizer_rule (15 annots, 2 queries) |
| 06-08 P3 | Config defaults | 0.943 | 100% | can_match_samename, FLANN defaults |
| 06-08 P4 | Remaining filters | — | — | rotation_invariance, scale/fgw thresholds |
| 06-08 P5 | SV completeness | — | — | shortlisting, xy/scale/ori thresholds |
| **06-08 big** | **Large-scale (51 annots)** | **0.735** | **75%** | First 12-query run with canonical naming |

## What was ruled out

| Theory | Result |
|---|---|
| FLANN KD-tree non-determinism | Exact search = identical results |
| Missing FG weights | `fg_on=False` in both systems |
| Wrong chip dimensions | Fixed: 700/maxwh matches WBIA |
| Crop+resize vs warpAffine | Fixed: now uses warpAffine |
| Wrong distance normalization | Fixed: `raw/524288`, sqrt |
| Different pyhesaff params | Both use defaults |
| probchip masking | WBIA uses raw RGB chip |
| Missing bar_l2/const/fg filters | All implemented |
| Missing ratio filter | Implemented (`ratio_thresh`) |
| Missing nsum/fmech scoring | Implemented (Phase 1b) |
| Missing canonical alignment | Implemented (Phase 1c) |
| Missing normalizer_rule='name' | Implemented (Phase 2a) |
| Missing rotation_invariance | Implemented (Phase 4a) |
| Missing SV shortlisting | Implemented (Phase 5) |

## Remaining root causes of the 75% → 92% gap

1. **Kpad self-padding** (HIGH) — wbia-core loses a voting column per feature
2. **FG weight formula difference** (HIGH) — different feature weighting changes LNBNN
3. **Descriptor-level differences** — 2.3× raw FLANN distance gap between systems persists

## Acceptance criteria

| Criterion | Current | Target |
|---|---|---|
| Top-1 accuracy (large) | 75% | ≥ 90% |
| Top-1 accuracy (small) | 100% | ≥ 90% |
| Determinism (repeat run) | ✓ | ✓ |
| Scoring pipeline | Phases 1-5 complete | — |
| Requery / ScoreNormalizer | Not implemented | Low priority |
