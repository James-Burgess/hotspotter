# Parity Roadmap: wbia-core → WBIA HotSpotter

Last updated: 2026-06-08d. Benchmark bugs fixed, parity confirmed.

## BREAKING: Historical gap was benchmark artifact (2026-06-08d)

The 75% vs 91.7% gap reported below was caused by two bugs in `targets/wbia.py`:

1. Read wrong WBIA score field (`annot_score_list` csum instead of `score_list` nsum)
2. Never passed `annot_name_list` to WBIA annotation API (50 unique names → fmech degenerated)

After fixes, both targets produce identical results (11/12 top-1 agree, scores within 1%).
See devlog 08c/08d for details.

## Large-scale benchmark (25 annots, 12 queries, seed=420) — POST-FIX

| Metric | wbia-core | wbia-develop |
|---|---|---|
| Top-1 accuracy | 50.0% (6/12) | 50.0% (6/12) |
| MRR | 0.681 | 0.688 |
| Top-1 agreement | 11/12 | — |

## ⚠️ HISTORICAL — Large-scale benchmark (51 annots, 12 queries, seed=122) — INVALIDATED

| Metric | wbia-core | wbia-develop |
|---|---|---|
| Top-1 accuracy | 75.0% (9/12) | 91.7% (11/12) |
| MRR | 0.814 | 0.958 |
| Spearman ρ (mean) | 0.735 | — |

## Phases 1-2 — Scoring depth + filters ✓ (9/9)

| # | Task | Impact | Status |
|---|---|---|---|
| 1a | Kpad dynamic | Medium | DONE |
| 1b | nsum/fmech name scoring | High | DONE |
| 1c | Canonical name alignment | High | DONE |
| 1d | WBIA scoring dispatch (csum_wbia, nsum_wbia, sumamech) | High | DONE |
| 1e | name_uuid from COCO individual_ids | High | DONE |
| 2a | Normalizer rule 'name' | High | DONE |
| 2b | bar_l2 filter | Low | DONE |
| 2c | ratio filter + threshold | Low | DONE |
| 2d | const filter | Low | DONE |

## Phases 3-4 — Config defaults + remaining filters ✓ (9/9)

| # | Task | Impact | Status |
|---|---|---|---|
| 3a | can_match_samename toggle (was hardcoded False) | High | DONE |
| 3b | flann_trees=8 (was 4) | Medium | DONE |
| 3c | flann_checks=800 (was 1028) | Low | DONE |
| 3d | prescore_method default='nsum' | Medium | DONE |
| 3e | sqrd_dist_on toggle | Low | DONE |
| 3f | normonly_on filter | Low | DONE |
| 4a | rotation_invariance (XY-dedup in fmech) | Medium | DONE |
| 4b | minscale_thresh / maxscale_thresh | Low | DONE |
| 4c | fgw_thresh | Low | DONE |

## Phase 5 — Spatial verification ✓ (4/4)

| # | Task | Impact | Status |
|---|---|---|---|
| 5a | SV prescoring shortlist (40 names, 3 annots/name) | Medium | DONE |
| 5b | xy_thresh=0.01, scale_thresh=2.0, ori_thresh | Low | DONE |
| 5c | min_nInliers=4 (was 3) | Low | DONE |
| 5d | use_chip_extent, weight_inliers (partial — see Phase 7) | Low | DONE |

## Phase 6 — Advanced mechanisms (2 remaining)

| # | Task | Impact | Status |
|---|---|---|---|
| 6a | Requery mechanism | Low | TODO |
| 6b | Score normalizer | Low | TODO |

## Phase 7 — NEW: High-impact gaps from dual-agent audit

| # | Gap | Impact | Status |
|---|---|---|---|
| 7a | **Kpad self-padding**: WBIA pads +1 when query in DB; wbia-core kpad=0 loses voting column | **HIGH** | TODO |
| 7b | **FG weight formula**: gaussian on full image vs CNN/RF probchip on chip | **HIGH** | TODO |
| 7c | **SV use_chip_extent**: image max dim vs chip diagonal normalization | **HIGH** | TODO |
| 7d | **weight_inliers mechanism**: biased RANSAC vs post-hoc multiplier | MEDIUM | TODO |
| 7e | **full_homog_checks**: exhaustive repeated-sampling vs single-pass RANSAC | MEDIUM | TODO |
| 7f | **sv_ori_thresh default**: None (disabled) vs TAU/4 (90°) | MEDIUM | TODO |
| 7g | **Database feature filtering**: query-only vs both query+db | MEDIUM | TODO |
| 7h | **can_match_sameimg**: missing in wbia-core | MEDIUM | TODO |
| 7i | **SV shortlist scoring**: score_method vs prescore_method | MEDIUM | TODO |

## Phase 8 — LOW: Remaining differences

| # | Gap | Impact | Status |
|---|---|---|---|
| 8a | sver_output_weighting (homog error filter) | Low | TODO |
| 8b | refine_method hardcoded to RANSAC | Low | TODO |
| 8c | lograt_on/cos_on toggles (dead code in WBIA) | Skip | — |

---

## Effort summary

| Phase | Items | Total effort | Remaining |
|---|---|---|---|
| 1 — Scoring depth | 5 | Medium | 0 |
| 2 — Filter completeness | 4 | Small | 0 |
| 3 — Config defaults | 6 | Small | 0 |
| 4 — Remaining filters | 3 | Small | 0 |
| 5 — Spatial verification | 4 | Medium | 0 |
| 6 — Advanced mechanisms | 2 | Medium | 2 |
| 7 — Audit HIGH/MEDIUM gaps | 9 | Medium-High | 9 |
| 8 — Audit LOW gaps | 2 | Small | 2 |
| **Total** | **35** | | **13 remaining** |

## What to fix first

1. **7a: Kpad self-padding** — Change `_compute_kpad` to always include +1 when query in DB (1 line)
2. **7i: SV shortlist scoring** — Use prescore_method to build shortlist, not score_method (~5 lines)
3. **7f: sv_ori_thresh default** — Change default from None to TAU/4 (~1 line)
4. **7g: Database feature filtering** — Apply minscale/maxscale/fgw_thresh to DB features too (~10 lines)
5. **7c: use_chip_extent chip diagonal** — Use bbox-based chip dlensqrd (~15 lines)
