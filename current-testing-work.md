# Testing Work Remaining

Coverage gaps identified in review (2026-06-30).

## 1. Unit tests for `chip.py` — HIGH

`extract_chip()` and `_compute_affine_matrix()` have zero direct tests.

Tests to write:
- [ ] `_compute_affine_matrix` with square bbox
- [ ] `_compute_affine_matrix` with rectangular bbox
- [ ] `_compute_affine_matrix` with non-zero theta
- [ ] `extract_chip` basic crop+resize
- [ ] `extract_chip` with grayscale image
- [ ] `extract_chip` with zero-width bbox (returns 64x64 zeros)
- [ ] `extract_chip` with zero-height bbox
- [ ] `extract_chip` with `resize_dim="width"`
- [ ] `extract_chip` with `resize_dim="height"`
- [ ] `extract_chip` with `resize_dim="maxwh"` (default)
- [ ] `extract_chip` output dtype matches input dtype
- [ ] `extract_chip` output aspect ratio matches bbox

## 2. Unit tests for `trace.py` — HIGH

`TraceContext` and helpers have zero direct tests.

Tests to write:
- [ ] `_zstd_compress` / `_zstd_decompress` roundtrip
- [ ] `TraceContext._array_summary` with finite array
- [ ] `TraceContext._array_summary` with empty array
- [ ] `TraceContext._array_summary` with inf/nan values
- [ ] `TraceContext._save_array` creates .npz sidecar
- [ ] `TraceContext._save_array` inlines small arrays
- [ ] `TraceContext._to_rows` normalizes dict payload
- [ ] `TraceContext._to_rows` normalizes list-of-dicts payload
- [ ] `TraceContext.dump_stage` writes parquet file
- [ ] `TraceContext._trace_order` puts query first
- [ ] `TraceContext.trace_neighbors` column structure
- [ ] `_is_trace_enabled` true/false via env var
- [ ] `get_trace_context` returns None when disabled
- [ ] `get_trace_context` writes manifest entry
- [ ] `_trace_order` handles last-element query
- [ ] `_GLOBAL_COUNTERS` increments across instances

## 3. Unit tests for `compute_normalizer_validity` (scoring.py) — MEDIUM

Name-based normalizer validation has zero direct tests.

Tests to write:
- [ ] All normalizers valid when every annotation has unique name
- [ ] Normalizer invalid when it shares name with voting annotation
- [ ] Normalizer invalid when it shares name with query
- [ ] Normalizer invalid when out-of-range label
- [ ] All valid when `normalizer_rule` not "name" (this path not taken)

## 4. Unit tests for `_filter_query_features` (pipeline.py) — MEDIUM

Feature filtering by scale/foreground weight has zero direct tests.

Tests to write:
- [ ] No-op when all thresholds are None (default)
- [ ] minscale_thresh filters small-scale keypoints
- [ ] maxscale_thresh filters large-scale keypoints
- [ ] fgw_thresh filters low-foreground-weight features
- [ ] Combining multiple thresholds
- [ ] Filtering all features returns empty FeatureSet (should pipeline handle this?)
- [ ] Filtering preserves descriptor/keypoint alignment

## 5. Property-based tests for scoring/name_scoring — MEDIUM

Use Hypothesis or simple parameterized invariants.

Tests to write:
- [ ] `weight_neighbors_lnbnn` weights are always >= 0
- [ ] `weight_neighbors_lnbnn` weights are never > 1 (for L2-normalized inputs)
- [ ] `weight_neighbors_lnbnn` weights are non-increasing per row (columns are sorted by distance)
- [ ] `weight_neighbors_lnbnn` weights zero when vdist > ndist (LNBNN invariant)
- [ ] `compute_fmech_score` nsum <= total csum for shared-name annotations
- [ ] `baseline_filter` always marks self as invalid
- [ ] `build_matches` never produces duplicates for the same (qfx, dfx) pair
- [ ] `score_matches` scores are non-negative

## 6. SiftConfig wiring — LOW

`_to_hesaff_kwargs` and SiftConfig scale parameter interaction untested.

Tests to write:
- [ ] Default SiftConfig produces expected hesaff kwargs
- [ ] Custom scale values flow to hesaff kwargs
- [ ] ori_hist_threshold flows to hesaff kwargs

## 7. Error/edge case tests — LOW

Pipeline robustness under unusual conditions.

Tests to write:
- [ ] identify() with empty database (should return [])
- [ ] identify() with single-annotation database
- [ ] identify() when all features are filtered by minscale/maxscale/fgw
- [ ] identify() when `knn > num_annotations_in_database`
- [ ] identify() with all annotations sharing the same name
- [ ] identify() with knn_backend="exact" vs "faiss" producing consistent results

## Verification

Run `make test-unit` after each batch of new tests.
