# WBIA Oracle Testing Framework

Records real WBIA HotSpotter pipeline runs to parquet checkpoints, compares any two
runs stage-by-stage with semantic agreement metrics, and enforces a **parity passing
mark of final name score Spearman ρ ≥ 0.97**.

## Workflow

```
record (once per WBIA image)
   ├── produces artifacts/wbia-oracle/<run-id>/
   │
compare (any two runs)
   ├── terminal summary
   ├── HTML report with per-stage detail
   └── parity gate: exit 2 if mean final name score ρ < 0.97
```

## Recording

The oracle recorder runs **inside** the WBIA container as a standalone Python
process. WBIA's async web job engine can execute queued jobs outside the
monkeypatched web process — running in-process avoids that.

```bash
python3 scripts/record_wbia_oracle.py
```

By default this records `nightly`, `latest`, and `develop`. To record specific images:

```bash
python3 scripts/record_wbia_oracle.py --images wildme/wbia:nightly wildme/wbia:latest-local
```

### Inputs

| Asset | Path |
|---|---|
| Test images (19 COCO JPEGs) | `pipeline/tests/assets/images/` |
| Annotation bboxes + metadata | `pipeline/tests/reference_batch.json` |
| WBIA compose | `docker-compose.ml.yml` |
| WBIA patches | `patches/patch_wbia_schema.py`, `patches/wbia_parquet_trace.py`, `patches/wbia_record_oracle_incontainer.py` |

## Query Configurations

Every WBIA image is run against **7 HotSpotter parameter permutations** (8th known
to crash — see below). Each config × 3 query annotations = 21 identify calls per
image.

| Config | `query_config_dict` | Tests |
|---|---|---|
| `sv_on_true` | `sv_on=True` | Baseline (K=4, Knorm=1, nsum scoring) |
| `sv_on_false` | `sv_on=False` | No spatial verification |
| `sv_on_n20` | `sv_on=True, n=20` | Troutspotter override |
| `K2` | `sv_on=True, K=2` | Fewer neighbors per descriptor |
| `K6` | `sv_on=True, K=6` | More neighbors per descriptor |
| `score_csum` | `sv_on=True, score_method=csum` | Max-csum name scoring |
| `pre_csum` | `sv_on=True, prescore_method=csum` | Max-csum pre-SV scoring |
| ~~`Knorm0`~~ | ~~`sv_on=True, Knorm=0`~~ | **Crashes WBIA** — divide-by-zero in `build_chipmatches` |

All configs carry `pipeline_root="vsmany"` and `fg_on=False` (foreground feature
matching disabled — CNN plugin unavailable).

## Output Structure

```
artifacts/wbia-oracle/<run-id>/
  trace_manifest.json           # Config labels + query indices per trace entry
  manifest.json                 # Run metadata, image/annot UUID mapping
  manifest.start.json           # Pre-run snapshot
  job-results/                  # Per-config per-query result JSONs
  oracle_dataset/               # Annot ID ↔ WBIA rowid mapping
  query_request/                # qreq payload per identify call
  annotations/                  # Per-query annot metadata dump
  chips/                        # Chip images (.npy sidecars)
  features_keypoints/           # PyHesaff keypoints (.npy)
  features_descriptors/         # PyHesaff SIFT descriptors (.npy)
  nearest_neighbors/            # FLANN KNN index + distances (.npy)
  baseline_neighbor_filter/     # Self-image filter masks
  neighbor_weights/             # LNBNN filter weights
  chipmatches_pre_sv/           # ChipMatch objects before SV
  chipmatches_post_sv/          # ChipMatch objects after SV
  final_scores/                 # Scored annotations + names + fm_list (.npy)
  oracle_query_result/          # Summary JSON per query
```

**File naming**: `{config_label}_{query_index:06d}.parquet` (e.g. `sv_on_true_000000.parquet`).
Hotspotter traces use the same convention so the comparison tool matches files by name.

Complete runs contain `oracle_dataset/`, `features_descriptors/`, `nearest_neighbors/`,
and `final_scores/`. Runs containing only `trace_start/` are failed web-engine
attempts and should be ignored.

Array columns (`*_array`) store JSON metadata with a `.npy` sidecar path. Large arrays
(>64 elements) are read from the sidecar; small arrays are inlined as `"values"`.

## Comparing Hotspotter Against WBIA

```bash
python3 scripts/compare_to_wbia.py \
  ../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-144646 \
  --passing-rho 0.97
```

This runs `hotspotter.identify()` with all 7 canonical configs against the same
test images, writes parquet traces, and passes both oracle dumps to
`compare_wbia_oracles.py`. Hotspotter parity configs currently force
`fg_on=False`, `kpad_policy="dynamic"`, and `knorm=1`.

## Comparing Two Oracle Runs (WBIA vs WBIA)

```bash
python3 scripts/compare_wbia_oracles.py \
  artifacts/wbia-oracle/wildme-wbia-nightly-20260625-105210 \
  artifacts/wbia-oracle/wildme-wbia-latest-20260625-105303
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--out <path>` | auto | HTML report path |
| `--text-out <path>` | — | Write terminal summary to file |
| `--atol` | 1e-9 | Absolute tolerance for numeric comparison |
| `--rtol` | 1e-9 | Relative tolerance for numeric comparison |
| `--fail-on-diff` | off | Exit 1 when any stage has bit-level differences |
| `--passing-rho` | **0.97** | Minimum final name score Spearman ρ for parity PASS |

### Exit Codes

| Code | Meaning |
|---|---|
| 0 | Parity passed (ρ ≥ `--passing-rho`) |
| 1 | `--fail-on-diff` set and bit-level differences found |
| 2 | Parity FAIL — mean final name score ρ below threshold |

### Terminal Output (Hotspotter vs WBIA Nightly, 2026-06-25)

Latest run after Phase 2 fixes (query-excluded FLANN, batch-order database,
negative-bbox chip fix, knorm/kpad configs, 19-row traces):

```
Stages compared: 18    Stages with differences: 18
Scalar cell differences: 6692    Array differences: 1036
All 9 configs run; 21 files matched per stage

Rich Metrics
Metric                               Mean    Count    Range
---------------------------------------------------------------
Neighbor dist Pearson r             0.0000      21    npy path bug in comparer
Actual neighbor dist Pearson r      0.9789    1538    per-row from npy files
Neighbor ID exact match             0.7298    7690    col0=0.904 col1=0.812 col2=0.713 col3=0.642 col4=0.577
Descriptor cosine similarity        0.0000     399    npy path bug in comparer
Actual descriptors                  identical 36,423  all 19 annots bit-identical
Final annot score Spearman ρ        0.1136       9    −0.3003–0.7957
Final name score Spearman ρ         0.3031       9    −0.0258–0.5335
Feature match Jaccard               0.0993      21    0.0000–0.2221
SV pruning agreement                0.4762      21    0.0000–1.0000

Parity check: mean final name score Spearman ρ = 0.3031 (threshold 0.970)  FAIL
```

Key findings since baseline: features are **100% identical** (36,423
descriptors match). Neighbor arrays have matching [N,5] shapes. Neighbor
IDs match at 73% (was 7.2% before descriptor-order fix). Neighbor
distances correlate at r=0.98 (comparer reports 0.00 due to npy path
resolution bug).

The ρ = 0.30 reflects remaining FLANN non-determinism (27% neighbor
divergence from different pyflann/numpy Docker image versions) and SV
semantics (47.6% agreement). See
`docs/development/hotspotter-parity-discrepancies.md` for full analysis.

### HTML Report

The HTML report includes:

- **Parity banner** — PASS/FAIL with the ρ value and threshold, visible immediately.
- **Analysis section** — human-readable narrative tracing where divergence starts
  (FLANN), how it propagates (neighbors → chipmatches → SV → scores), what the
  metrics mean, and whether the two runs are statistically equivalent.
- **Rich Metrics table** — semantic agreement metrics with mean, count, min/max.
- **Per-stage detail** — every parquet file with row counts, scalar diffs, array
  diffs, and individual array mismatch details.

Reports are written to `artifacts/wbia-oracle/comparisons/<run_a>__vs__<run_b>.html`.

## Rich Metrics Reference

| Metric | What it measures | Par with ρ≥0.97? |
|---|---|---|
| Neighbor dist Pearson r | Correlation of FLANN distance vectors per query | ~0.993 always |
| Neighbor ID exact match | Fraction of identical neighbor assignments | 0.73 (hotspotter vs WBIA) |
| Descriptor cosine similarity | Pairwise cosine between same-annot descriptors | 1.0 (bit-identical verified) |
| Daid Jaccard pre_sv | Set overlap of candidate annotation lists | 1.0 always |
| Daid Jaccard post_sv | Set overlap after spatial verification | 0.945–1.0 |
| Final annot score Spearman ρ | Rank correlation of per-annotation scores | Parity target |
| Final name score Spearman ρ | Rank correlation of per-individual scores | **≥0.97 pass** |
| Feature match Jaccard | Overlap of feature correspondence pairs | 0.0993 (hotspotter vs WBIA) |
| SV pruning agreement | Do both runs prune the same annotations? | 0.4762 (hotspotter vs WBIA) |

## What We Know About WBIA Determinism

All findings from 4 WBIA images × 7 configs × 21 pairwise comparisons (2026-06-25):

### Fully deterministic (ρ=1.0, bit-identical)

- **Chips**: identical pixel output from same image + bbox.
- **Feature keypoints**: identical PyHesaff output.
- **Feature descriptors**: identical SIFT vectors (128-dim uint8).
- **Baseline neighbor filter**: same self-image exclusion mask.
- **Neighbor weights**: same LNBNN weight computation.
- **Query request configs**: identical effective parameters.
- **Annotation metadata**: identical across same image/bbox input.

### Root divergence point

- **FLANN nearest neighbors**: non-deterministic index build order.
  - Neighbor distance vectors are near-identical (Pearson r ≥ 0.993).
  - Neighbor index assignments diverge (nearly all query descriptors get different
    FLANN-assigned neighbors).
  - This is the **only** source of non-determinism in the entire pipeline.

### How divergence cascades

FLANN divergence → different neighbor assignments → different feature matches
(Jaccard ~0.65) → different pre-SV scores → spatial verification prunes differently
(though pruned set is the same for same-build images) → final scores differ
(ρ 0.93–0.99 depending on image pair).

### KNOWN Failure

- **Knorm=0** crashes WBIA (`IndexError: cannot do a non-empty take from an empty axes`
  in `build_chipmatches > get_sparse_matchinfo_nonagg`). This parameter is not
  viable in production.

### Image build differences

The `develop` image is built via `develop/Dockerfile` which starts FROM the
standard `latest` image, copies the local wildbook-ia source over, and re-runs
`pip install` on all requirements. The dependency re-resolution introduces small
version variance in numpy/scipy that causes slightly larger FLANN/score drift
compared to the provision image's fixed dependency set.

| Pair | Name ρ |
|---|---|
| Same image, re-run | 0.991 (FLANN ceiling) |
| Same date, different images | 0.969–0.986 |
| Builds 1 year apart | 0.955–0.974 |

## Fixture Reference

| Field | Value |
|---|---|
| Test images | 19 COCO zebra JPEGs |
| Annotations | 3 queries + 16 database |
| Species | `zebra_plains` |
| Bbox source | `pipeline/tests/reference_batch.json` (COCO format) |
| WBIA images tested | nightly (2026-05-09), latest (Docker Hub 2025-06-05), latest-local (2026-06-24), develop (2026-06-24) |
