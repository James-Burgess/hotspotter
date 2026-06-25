# Hotspotter Parity Discrepancies

Known gaps between `hotspotter` library output and WBIA oracle output.
Findings from actual parquet-to-parquet comparison (2026-06-25).

## Run The Comparison

```bash
# One-command parity check (requires hotspotter:latest Docker image)
make test-parity

# Point to a specific oracle
make test-parity ORACLE=../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-173226

# Or manually:
python3 scripts/compare_to_wbia.py \
    ../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-173226 \
    --passing-rho 0.97
```

## Oracle Recording

```bash
# Record all WBIA images (nightly, latest, develop)
python3 ../scripts/record_wbia_oracle.py

# Naming convention: {config_label}_{query_index:06d}.parquet
# Both WBIA monkeypatch and hotspotter trace use the same scheme
# with trace_manifest.json at the run root
```

## Current Oracle Baseline (2026-06-25)

Latest nightly oracle: `wildme-wbia-nightly-20260625-173226`

9 configs × 3 queries = 27 entries. Knorm0 excluded (crashes WBIA with
divide-by-zero).  sv_on_n20 / kpad_fixed_0 are missing from WBIA's
annotations/chips traces (WBIA trace recorder skip).

## Stage Coverage

All 10 hotspotter HotSpotter trace stages match WBIA. All stages now write
**19 rows per file** (matching WBIA's per-annotation structure). AID alignment
between the two systems is perfect (hotspotter 0-based AIDs map 1:1 to WBIA
1-based AIDs).

| Stage | WBIA | Hotspotter | Notes |
|---|---|---|---|
| `chips` | 19 rows/file | 19 rows/file | AIDs aligned; chip_fpath differs |
| `annotations` | 19 rows/file | 19 rows/file | UUIDs/bboxes run-specific |
| `features_keypoints` | 19 rows/file | 19 rows/file | 19/19 kpt counts identical |
| `features_descriptors` | 19 rows/file | 19 rows/file | 36,423/36,423 descriptors identical |
| `nearest_neighbors` | [N,5] | [N,5] | Same K,Kpad,Knorm; 73% ID match |
| `baseline_neighbor_filter` | [N,4] bool | [N,4] bool | — |
| `neighbor_weights` | [N] float | [N] float | LNBNN formula matches |
| `chipmatches_pre_sv` | 1 row/file | 1 row/file | UUID vs aid naming |
| `chipmatches_post_sv` | 1 row/file | 1 row/file | UUID vs aid naming |
| `final_scores` | 1 row/file | 1 row/file | UUID vs aid naming |

## Measured Metrics (Hotspotter vs WBIA Nightly)

Oracle: `wildme-wbia-nightly-20260625-173226`.
21 matched files across 7 working configs (sv_on_true, sv_on_false, K2, K6,
score_csum, pre_csum, Knorm2). `kpad_fixed_0` files partially missing in WBIA.

| Metric | Mean | Count | Range | Notes |
|---|---|---|---|---|
| Neighbor dist Pearson r | 0.00 | 21 | — | **Comparer bug**: npy path broken |
| Actual neighbor dist r | **0.9789** | 1538 | — | Loaded npy manually, per-row r |
| Neighbor ID exact match | **72.98%** | 7690 pairs | 90.4% (col 0) – 57.7% (col 4) | Col 0 match drops per column |
| Descriptor cosine similarity | 0.00 | 399 | — | **Comparer bug**: npy path broken |
| All 19 annotation descriptors | **bit-identical** | 36,423 | — | PyHesaff output matches exactly |
| Final annot score Spearman ρ | 0.1136 | 9 | −0.3003 – 0.7957 | |
| Final name score ρ | **0.3031** | 9 | −0.0258 – 0.5335 | |
| Feature match Jaccard | 0.0993 | 21 | 0.0000 – 0.2221 | |
| SV pruning agreement | 0.4762 | 21 | 0.0000 – 1.0000 | |

Parity threshold is ρ ≥ 0.97. Current state: **FAIL (ρ = 0.3031)**.

Key finding: features are **100% identical** (descriptors, keypoints, chip
pixels all match). Neighbor distances correlate at r=0.98. 73% of neighbor
IDs are identical. The remaining gap comes from:
1. The 27% non-matching FLANN neighbors (different pyflann/numpy versions
   across Docker images produce different KD-tree structures)
2. Scoring amplification — small neighbor differences cascade through
   LNBNN → csum → name aggregation → SV
3. SV pruning agreement at 47.6% — half of annotations pruned differently

## Completed Parity Fixes

| Area | Change | Result |
|---|---|---|
| `knorm` | `HotSpotterConfig.knorm` (`ge=1`), used by `identify()` | Knorm=2 tested; Knorm=0 unsupported |
| Kpad | `_compute_kpad` respects `can_match_samename`; dynamic policy | Kpad=0 matches WBIA oracle |
| FLANN index | Query excluded from index (matching WBIA `qreq_.daids`) | No self-match penalty |
| Descriptor stacking | Database built in batch order (not queries-first) | 73% neighbor ID match (was 7.2%) |
| Chip extraction | Pass raw bbox to affine (negative coords via BORDER_CONSTANT) | 19/19 kpt counts match WBIA |
| Trace row counts | All stages write 19 rows via `trace_chips_and_features` batch | Row counts match WBIA |
| Trace chips schema | `chip_fpath`, `chip_size` columns (WBIA-compatible) | Schema aligned |
| Spatial verification | `vtool.spatially_verify_kpts()` replaces `cv2.findHomography` | Same vtool family as WBIA |
| `fm_list` trace | `Nx2` `[qfx, dfx]` arrays as `.npy` sidecars with `values` | Jaccard now computable |
| Docker deps | Vendored `wbia-vtool 4.0.3` preserved; runtime deps explicit | vtool.sver imports |
| Trace query index | `trace_query_index` decoupled from database index | Files match when queries non-sequential |
| AGENTS.md | Removed incorrect "query included in FLANN" directive | Reflects WBIA reality |

## Root Causes (Ranked by Impact)

### 1. FLANN Non-Determinism (Docker Image Variance) — HIGH

Both systems use identical features and identical `random_seed=42`, but
different pyflann/numpy versions across Docker images produce different
KD-tree structures.  73% of neighbor IDs match; 27% diverge.

**Impact**: 27% different neighbors → different feature matches → different
LNBNN weights → different scores.

**Fix path**: Align pyflann/numpy versions between hotspotter and WBIA
Docker images, or accept this as the FLANN noise floor (~0.99 ρ for
WBIA-vs-WBIA is the theoretical ceiling).

### 2. Comparer npy Path Resolution — HIGH

The `compare_wbia_oracles.py` script can't load hotspotter `.npy` sidecars
because the metadata paths point to `/artifacts/wbia-oracle/...` (inside
the Docker container), not the actual host path `/tmp/hotspotter-trace-...`.
This makes `neighbor_dist_pearson_r` and `descriptor_cosine` always report
0.00 even though the actual data shows r=0.98 and descriptors are identical.

**Impact**: Rich metrics are misleading; human analysis requires manual npy
loading to confirm structural correctness.

**Fix path**: Either write paths relative to trace root, or teach the
comparer to resolve paths by searching the trace directory.

### 3. Spatial Verification Semantics — MEDIUM

Hotspotter calls `vtool.spatially_verify_kpts()` but SV pruning agreement
is 0.4762 — half of annotations pruned differently from WBIA.

**Impact**: Different pruned sets → different post-SV scores → rank-order
drift even when pre-SV matches are identical.

**Fix path**: Align match weights, dlen/extent inputs, refine method,
shortlist ordering, and post-SV score update to match WBIA.

### 4. Scoring Amplification — MEDIUM

With 73% matching neighbor IDs, final name ρ is only 0.30. The LNBNN
weighting → csum → name-level aggregation chain amplifies small neighbor
differences. The scoring formulas (fmech/nsum, max-csum, canonical
alignment) are correct, but they operate on different input match sets.

**Impact**: Even structurally correct scoring produces different rankings
from different neighbor assignments.

**Fix path**: Reduce neighbor divergence first (fix #1), then check whether
scoring produces matching output for matching neighbors.

### 5. Scoring Method Names — LOW

WBIA `score_method="csum"` means max-csum name scoring. Hotspotter names
this `"csum_wbia"`. The `compare_to_wbia.py` script maps names correctly,
but manual runs can misconfigure.

**Fix path**: Align hotspotter names to WBIA semantics, or keep mapping shim.

## Config Table

| Config Label | WBIA query_config_dict | Hotspotter equivalent |
|---|---|---|
| `sv_on_true` | `sv_on=True` | `sv_on=True, kpad_policy=dynamic, knorm=1` |
| `sv_on_false` | `sv_on=False` | `sv_on=False, kpad_policy=dynamic, knorm=1` |
| `sv_on_n20` | `sv_on=True, n=20` | `sv_on=True, num_return=20, kpad_policy=dynamic, knorm=1` |
| `K2` | `sv_on=True, K=2` | `sv_on=True, knn=2, kpad_policy=dynamic, knorm=1` |
| `K6` | `sv_on=True, K=6` | `sv_on=True, knn=6, kpad_policy=dynamic, knorm=1` |
| `score_csum` | `sv_on=True, score_method=csum` | `sv_on=True, score_method=csum_wbia, kpad_policy=dynamic, knorm=1` |
| `pre_csum` | `sv_on=True, prescore_method=csum` | `sv_on=True, prescore_method=csum_wbia, kpad_policy=dynamic, knorm=1` |
| `Knorm2` | `sv_on=True, Knorm=2` | `sv_on=True, knorm=2, kpad_policy=dynamic` |
| `kpad_fixed_0` | `sv_on=True, Kpad=0` | `sv_on=True, kpad_policy=fixed, kpad=0, knorm=1` |

All configs carry `pipeline_root="vsmany"` and `fg_on=False`.

## Known Crash Configs

- **Knorm=0**: Crashes both WBIA and hotspotter. Excluded from parity.
- **sv_on_n20 / kpad_fixed_0**: WBIA trace recorder skips annotations/chips
  stages for these configs (non-blocking, files partially missing).

## Fixture Reference

| Field | Value |
|---|---|
| WBIA reference oracle | `../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-173226/` |
| Test images | 19 COCO zebra JPEGs at `../pipeline/tests/assets/images/` |
| Reference batch | `../pipeline/tests/reference_batch.json` |
| Configs | 9 valid (8 working in WBIA), 1 crash (Knorm0) |
| Parquet stages | 10 hotspotter stages, 18 WBIA stages |
| Comparison script | `scripts/compare_to_wbia.py` → `../scripts/compare_wbia_oracles.py` |
| Parity gate | Final name score Spearman ρ ≥ 0.97 |

## Phase 2 Priority Queue

1. **FLANN/pyflann version alignment** — match WBIA Docker image's pyflann/numpy
   versions in hotspotter's Dockerfile to eliminate remaining 27% neighbor
   divergence.
2. **Comparer npy path resolution** — fix sidecar paths or teach the comparer
   to resolve them, so rich metrics report real values.
3. **Spatial verification semantics** — align vtool inputs to raise SV
   agreement from 47.6%.
4. **Scoring pipeline audit** — verify fmech/nsum/csum produce identical
   output given identical neighbors (unit test with shared reference data).
5. **Automated pytest parity test** — once metrics are stable, add CI gate.
