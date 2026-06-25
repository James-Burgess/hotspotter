# Phase 1 Baseline

Captured before any Phase 1 changes.

**Date**: 2026-06-25
**Image**: `wbia-core:latest` (build 2026-06-24, Python 3.10)

## Unit Tests (38 tests)

All 38 pass in ~1s.

```
tests/test_config.py — 8 tests (defaults, validation, serialization)
tests/test_data.py    — 6 tests (FeatureSet shapes, AnnotatedImage, Match, ScoredMatch)
tests/test_features.py — 2 tests (extract_features color + grayscale)
tests/test_knn.py     — 2 tests (build_and_query, nearest_is_self)
tests/test_pipeline.py — 7 tests (identify shape, self/same-name exclusion, SV, correspondences, large db)
tests/test_scoring.py — 9 tests (self-match filter, LNBNN weighting, build_matches, nsum/csum scoring)
tests/test_spatial.py — 3 tests (passthrough, exact correspondences, out-of-range)
```

## Replay Tests (84 tests)

All 84 pass in ~4.5s. 12 fixture sets (giraffe_reticulated, whale_shark, zebra_grevys).
Each set: fixture_loads, wbia_scores_parsable, image_decodable, replay_rankings, self_excluded, correspondences, spatial_verification.

## Known Oracle Findings

From comparisons across nightly/latest/latest-local/develop WBIA images:

- Chips, keypoints, descriptors: bit-identical across all WBIA images.
- FLANN nearest neighbors: non-deterministic index order (root divergence point).
- Neighbor distance Pearson r ≥ 0.993 across all pairs.
- Final name score Spearman ρ: 0.955–0.991 depending on image pair.
- Feature match Jaccard: ~0.62–0.66 across image pairs.
- Parity passing mark set at ρ ≥ 0.97.
- 7 HotSpotter config permutations working; K norm=0 crashes WBIA.

## Git Status

Clean working tree. Uncommitted docs added in prior work.
