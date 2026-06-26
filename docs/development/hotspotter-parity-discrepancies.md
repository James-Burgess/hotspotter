# Hotspotter Parity Discrepancies

Known gaps between `hotspotter` library output and WBIA oracle output.
Findings from actual parquet-to-parquet comparison (latest run: 2026-06-26).

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

## Current Oracle Baseline (2026-06-26)

Latest nightly oracle: `wildme-wbia-nightly-20260625-173226`

Latest hotspotter trace checked:
`../artifacts/hotspotter-debug-trace/full-oracle-nsum-singleton-20260626-102528/`

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
Hotspotter trace: `full-oracle-nsum-singleton-20260626-102528`.
21 matched files across 7 working configs (sv_on_true, sv_on_false, K2, K6,
score_csum, pre_csum, Knorm2). `kpad_fixed_0` files partially missing in WBIA.

| Metric | Mean | Count | Range | Notes |
|---|---|---|---|---|
| Neighbor dist Pearson r | **0.9927** | 21 | 0.9893 – 0.9956 | Distances nearly identical |
| Descriptor cosine similarity | **1.0000** | 399 | 1.0000 – 1.0000 | Descriptors bit-identical |
| Daid Jaccard pre-SV | **1.0000** | 21 | 1.0000 – 1.0000 | Candidate annotation sets match |
| Daid Jaccard post-SV | 0.9577 | 21 | 0.8889 – 1.0000 | SV still prunes differently |
| Final annot score Spearman ρ | −0.1401 | 10 | −0.5893 – 0.7379 | |
| Final name score ρ | **−0.1257** | 10 | −0.4180 – 0.2508 | |
| Feature match Jaccard | 0.1113 | 21 | 0.0000 – 0.7812 | |
| SV pruning agreement | 0.4762 | 21 | 0.0000 – 1.0000 | |

Parity threshold is ρ ≥ 0.97. Current state: **FAIL (ρ = −0.1257)**.

Key finding: features are **100% identical** (descriptors and keypoints match).
Chip dimensions now match WBIA exactly in parquet traces: 21 common chip files
compared, **0 `chip_size` mismatches**. Examples: aid 1 `[700, 401]`, aid 2
`[700, 427]`, aid 6 `[700, 538]`, aid 17 `[700, 615]` in both systems.
Neighbor distances correlate at r≈0.993. The remaining gap comes from:
1. The ~25–30% non-matching FLANN neighbors (different pyflann/numpy versions
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
| Chip dimensions | Trace `chip_size` uses WBIA `[width, height]`; latest run has 0/21 mismatched chip files | Dimensions aligned |
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

### 2. Spatial Verification Semantics — MEDIUM

Hotspotter calls `vtool.spatially_verify_kpts()` but SV pruning agreement
is 0.4762 — half of annotations pruned differently from WBIA.

**Impact**: Different pruned sets → different post-SV scores → rank-order
drift even when pre-SV matches are identical.

**Fix path**: Align match weights, dlen/extent inputs, refine method,
shortlist ordering, and post-SV score update to match WBIA.

### 3. Scoring Amplification — MEDIUM

With ~70–75% matching neighbor IDs, final name ρ is still failing. The LNBNN
weighting → csum → name-level aggregation chain amplifies small neighbor
differences. The scoring formulas (fmech/nsum, max-csum, canonical
alignment) are correct, but they operate on different input match sets.

**Impact**: Even structurally correct scoring produces different rankings
from different neighbor assignments.

**Fix path**: Reduce neighbor divergence first (fix #1), then check whether
scoring produces matching output for matching neighbors.

### 4. Scoring Method Names — LOW

WBIA default scoring is `score_method="nsum"`; `score_method="csum"` remains
supported for the csum configs. Hotspotter traces now emit `nsum` for default
configs and `csum` for `score_csum`.

**Fix path**: Align hotspotter names to WBIA semantics, or keep mapping shim.

## Config Table

| Config Label | WBIA query_config_dict | Hotspotter equivalent |
|---|---|---|
| `sv_on_true` | `sv_on=True` | `sv_on=True, kpad_policy=dynamic, knorm=1` |
| `sv_on_false` | `sv_on=False` | `sv_on=False, kpad_policy=dynamic, knorm=1` |
| `sv_on_n20` | `sv_on=True, n=20` | `sv_on=True, num_return=20, kpad_policy=dynamic, knorm=1` |
| `K2` | `sv_on=True, K=2` | `sv_on=True, knn=2, kpad_policy=dynamic, knorm=1` |
| `K6` | `sv_on=True, K=6` | `sv_on=True, knn=6, kpad_policy=dynamic, knorm=1` |
| `score_csum` | `sv_on=True, score_method=csum` | `sv_on=True, score_method=csum, kpad_policy=dynamic, knorm=1` |
| `pre_csum` | `sv_on=True, prescore_method=csum` | `sv_on=True, prescore_method=csum, kpad_policy=dynamic, knorm=1` |
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
2. **Spatial verification semantics** — align vtool inputs to raise SV
   agreement from 47.6%.
3. **Scoring pipeline audit** — verify fmech/nsum/csum produce identical
   output given identical neighbors (unit test with shared reference data).
4. **Automated pytest parity test** — once metrics are stable, add CI gate.
