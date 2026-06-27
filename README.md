# Hotspotter

Stateless HotSpotter wildlife re-identification pipeline — extracted from `wildbook-ia`, distributed as a pure-Python pip package.

## Executive summary

Hotspotter is the **pattern-matching engine** that powers Wildbook's
*Who is this animal?* question. You give it a crop of an animal (a query
"chip") and a collection of known animals (a database of chips). It
returns a ranked list of the most likely matches, each with a confidence
score.

**How it works in four steps:**

1. **Describe** — extract SIFT keypoints and descriptors from every chip
   using Hessian-affine feature detection.
2. **Match** — for each feature on the query, find its K nearest neighbours
   across the entire database (global FLANN index or exact L2 KNN).
3. **Score** — apply LNBNN weighting and per-annotation scoring to turn
   raw feature matches into annotation-level scores.
4. **Verify** — run RANSAC homography spatial verification to reject
   matches that don't form a consistent geometric relationship.

**What it replaces** — the Hotspotter algorithm core from Wildbook's
`wildbook-ia` monorepo. Extraction verified to WBIA parity (see
`deeseek-wbia-parity.md`). Top-1 identification accuracy: 87% on the GZGC
zebra COCO benchmark.

**What it does not replace** — detection (MegaDetector/YOLO), species
classification, embedding models, the IBEIS database, the job queue, or
the web API. Those live in `wildlife-id` and `ml-service`.



## Module map

| `wildbook-ia` | `hotspotter` |
|---|---|
| `wbia.algo.hots.chip_match` (3039 lines) | `hotspotter.data` (dataclasses) |
| `wbia.algo.hots.pipeline` (1655 lines) | `hotspotter.pipeline.identify` (one function) |
| `wbia.algo.hots.nn_weights` (669 lines) | Inlined filter chain in `pipeline.py` |
| `wbia.algo.hots.name_scoring` (404 lines) | `hotspotter.name_scoring` |
| `wbia.algo.hots.scoring` (177 lines) | `hotspotter.scoring` |
| `wbia.algo.hots.neighbor_index` (1118 lines) | `hotspotter.knn` |
| `wbia.algo/Config.py` (nested config) | `hotspotter.config` (Pydantic v2) |

What it does **not** replace: detection (YOLO/MegaDetector), classification, embedding extraction, the `IBEISController`, the depcache, the ZMQ job engine, or the `wbia_*` PostgreSQL schemas. Those belong to `ml-service` and `wildlife-id`.

## Design

- **Stateless.** Pure functions over numpy arrays. No DB, no network, no `IBEISController`.
- **Deterministic.** Same `(image, config)` → same features, every time.
- **Package-first.** Target boundary is a reusable `hotspotter` library; sidecar/API code is transitional.
- **Single global FLANN index** over database descriptors (query excluded), matching WBIA's `NeighborIndex` behaviour.
- **Configurable filter chain.** LNBNN, bar_l2, ratio, FG, const, and normonly filters combine multiplicatively.
- **Name-level scoring.** fmech (nsum), max-per-name (csum_wbia), per-name sum (sumamech), with canonical alignment.
- **Spatial verification.** RANSAC homography with prescore shortlisting, xy/scale/ori threshold filtering.

## Architecture

```
wbia-core/
├── Dockerfile                         # Source-compiled submodule build
├── Makefile                           # test, build, server, shell shortcuts
├── pyproject.toml                     # numpy, pydantic, opencv, flask, pyflann
├── sidecar/
│   └── api.py                         # Flask — POST /api/v1/identify/
├── scripts/entrypoints/
│   ├── server-entrypoint.sh           # gunicorn entrypoint
│   └── test-entrypoint.sh             # parity test entrypoint
├── src/hotspotter/
│   ├── pipeline.py                    # identify() — full LNBNN + filters + SV
│   ├── scoring.py                     # per_feature_fg, score_matches
│   ├── name_scoring.py                # compute_fmech_score, align_name_scores_with_annots
│   ├── spatial.py                     # spatial_verify, make_sver_shortlist
│   ├── knn.py                         # build_global_index, query_index, exact_knn
│   ├── features.py                    # extract_features() — pyhesaff only
│   ├── data.py                        # FeatureSet, AnnotatedImage, Match, ScoredMatch
│   ├── config.py                      # HotSpotterConfig, IdentificationConfig (Pydantic v2)
│   └── debug_log.py                   # Pipeline debug tracing (WBIA_CORE_DEBUG=1)
├── tests/
│   ├── benchmark/                     # COCO multi-target regression suite
│   ├── replay/                        # Recorded WBIA fixture replay tests
│   └── test_*.py                      # Unit + integration tests (38 total)
├── docs/
│   └── development/                   # Plans, contracts, devlog, parity analysis
├── wbia-utool/                        # git submodule — pure Python utilities
├── wbia-vtool/                        # git submodule — vision tools (libsver.so)
└── wbia-tpl-pyhesaff/                 # git submodule — Hessian-affine SIFT (libhesaff.so)
```

## Submodules

Three WildMe packages are vendored as git submodules for source compilation:

| Submodule | Path | Purpose |
|---|---|---|
| `wbia-utool` | `wbia-core/wbia-utool/` | Utility library (pure Python) |
| `wbia-vtool` | `wbia-core/wbia-vtool/` | Vision tools + spatial verification (`libsver.so`) |
| `wbia-tpl-pyhesaff` | `wbia-core/wbia-tpl-pyhesaff/` | Hessian-affine SIFT feature extraction (`libhesaff.so`) |

They must be compiled from source against the target system's OpenCV. The Dockerfile handles this in dependency order: `wbia-utool → wbia-vtool → wbia-tpl-pyhesaff → wbia-core`.


## Configuration

All pipeline parameters live in `HotSpotterConfig` (Pydantic v2 model). Override via
`HotSpotterConfig(knn_backend="flann", sv_on=False, ...)` or JSON `--config` in
`run_fixture.py`.

### KNN / neighbour columns

| Parameter | Default | Description |
|---|---|---|
| `knn` | 4 | Number of voting neighbour columns (`K`) |
| `knorm` | 1 | Normalizer column count. Adds `knorm` farthest neighbours after `K+Kpad`. WBIA: 1. |
| `kpad` | 1 | Extra columns between voting and normalizer to absorb self-/same-name matches |
| `kpad_policy` | `"fixed"` | `"fixed"` uses `kpad` as-is; `"dynamic"` counts impossible aids at runtime |
| `knn_backend` | `"exact"` | **`"exact"`** (numpy L2, deterministic), **`"flann"`** (pyflann kdtree, approximate), **`"faiss"`** (IndexFlatL2, deterministic). Use `"flann"` for WBIA parity; `"exact"` for production. |

### FLANN parameters (only when `knn_backend="flann"`)

| Parameter | Default | WBIA default | Description |
|---|---|---|---|
| `flann_algorithm` | `"kdtree"` | `"kdtree"` | FLANN index algorithm |
| `flann_trees` | 4 | 4 | Number of parallel kd-trees |
| `flann_checks` | 32 | 32 | Leaf nodes checked per search (thoroughness) |
| `flann_random_seed` | -1 | -1 | Seed for kd-tree construction. -1 = random per run. |
| `flann_cores` | 1 | 1 | Thread count. 0 = all available. |

> **FLANN is non-deterministic by design** — even `random_seed=42` + single-threaded
> produces different KNN results across runs. WBIA's `seed=-1` makes every WBIA
> invocation produce a different kd-tree forest. Use `knn_backend="exact"` for
> reproducible results; use `knn_backend="flann"` + WB-matched params for WBIA
> parity comparison.

### Scoring

| Parameter | Default | Description |
|---|---|---|
| `score_method` | `"nsum"` | Per-annot scoring: `"nsum"` (mean), `"csum"` (sum), `"nsum_wbia"` (fmech), `"csum_wbia"` (max-per-name), `"sumamech"` |
| `prescore_method` | `"nsum"` | Scoring method for pre-SV shortlisting |
| `normalizer_rule` | `"last"` | Normalizer source: `"last"` (farthest neighbour), `"name"` (different-name validation) |
| `sqrd_dist_on` | `False` | Keep distances in squared-norm space (no sqrt). WBIA: False. |
| `normonly_on` | `False` | Replace voting distances with normalizer distance. Debug only. |
| `lnbnn_ratio` | 1.0 | LNBNN weight ratio multiplier |

### Feature-match filters

| Parameter | Default | Description |
|---|---|---|
| `fg_on` | `True` | Foreground weighting (multiplies match weights by FG confidence) |
| `bar_l2_on` | `False` | `1.0 - vdist` filter (more distant = lower score) |
| `const_on` | `False` | Constant-weight filter (all matches weight 1.0) |
| `ratio_thresh` | `None` | Lowe's ratio test threshold (e.g. 0.8). `None` = disabled. |

### Feature pre-filtering

| Parameter | Default | Description |
|---|---|---|
| `minscale_thresh` | `None` | Minimum keypoint scale before KNN query |
| `maxscale_thresh` | `None` | Maximum keypoint scale before KNN query |
| `fgw_thresh` | `None` | Minimum foreground weight (0.0–1.0) before KNN query |

### Match eligibility

| Parameter | Default | Description |
|---|---|---|
| `can_match_samename` | `True` | Allow matches to annotations sharing the query's name. WBIA: True. |
| `can_match_sameimg` | `False` | Allow matches to annotations in the same image (contact). WBIA: False. |
| `rotation_invariance` | `False` | XY-dedup in fmech scoring. Prevents rotated duplicate features from double-voting per name. |

### Spatial verification (SV)

| Parameter | Default | Description |
|---|---|---|
| `sv_on` | `True` | Enable RANSAC homography spatial verification |
| `sv_verify_all` | `True` | Send every scored annot through SV (matches WBIA behaviour). When `False`, shortlists top-N names. |
| `sv_n_name_shortlist` | 40 | Max names in SV shortlist (only when `sv_verify_all=False`) |
| `sv_n_annot_per_name` | 999 | Max annots per name in SV shortlist. 999 = effectively all. WBIA literal default is 3. |
| `sv_xy_thresh` | 0.01 | Max pixel-normalized XY error for inlier |
| `sv_scale_thresh` | 2.0 | Max scale ratio delta for inlier |
| `sv_ori_thresh` | 1.5708 | Max orientation delta in radians (WBIA: TAU/4) |
| `sv_use_chip_extent` | `True` | Compute `dlen_sqrd2` from chip dimensions (W²+H²) like WBIA |
| `sv_weight_inliers` | `True` | Bias RANSAC sampling toward high-FG features (WBIA `weight_inliers`). Does not multiply scores. |
| `sv_use_kp_affine_inliers` | — | **Deprecated.** Survival is gated by `sver is None` (affine < 7); scoring always uses homography-refined inliers. |
| `sv_sver_output_weighting` | `False` | Append homography-error weight per inlier and re-score. WBIA `sver_output_weighting`, defaults False. |

### Output

| Parameter | Default | Description |
|---|---|---|
| `num_return` | 10 | Maximum scored matches returned per query |

## API

The current high-level API is useful for parity work, but it is broader than the target `hotspotter` package boundary. Future cleanup should keep reusable feature/scoring primitives here and move service/index orchestration to `wildlife-id`.

### `identify(query_index, database, config) → list[ScoredMatch]`

Run the full identification pipeline for one query against a database of annotated images. Builds a single global FLANN index, applies LNBNN scoring with configurable filters, runs name-level aggregation, and optionally spatial verification.

### Data containers

| Class | Purpose |
|---|---|
| `FeatureSet` | Keypoints [N,6] + descriptors [N,128] for one image |
| `AnnotatedImage` | Annotation with features, image, bbox, name_uuid |
| `Match` | Single query-feature → database-feature correspondence |
| `ScoredMatch` | Per-annotation result with score, correspondences, SV inliers |

### Scoring functions

| Function | WBIA equivalent |
|---|---|
| `score_matches(matches, db, method)` | `scoring.score_chipmatch_list` |
| `compute_fmech_score(matches_by_name)` | `name_scoring.compute_fmech_score` |
| `align_name_scores_with_annots(...)` | `name_scoring.align_name_scores_with_annots` |
| `spatial_verify(matches, query_kp, db)` | `pipeline.spatial_verification` |
| `make_sver_shortlist(scored, n, m)` | `scoring.make_chipmatch_shortlists` |



See `docs/development/`:
- `deepseek-wbia-parity.md` - current notes about the work of achiving parity.
- `hotspotter-transition.md` — current boundary checklist for the `hotspotter` rename and `wildlife-id` split
- `parity-analysis.md` — investigation history + benchmark artifact analysis
- `parity-roadmap.md` — implementation plan with verification results
- `wbia-pipeline-differences.md` — per-section WBIA comparison
- `devlog.md` — full development log with all entries
- `executive-summary.html` — self-contained HTML dashboard
