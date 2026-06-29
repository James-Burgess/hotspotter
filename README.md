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
| `wbia.algo.hots.pipeline` (1655 lines) | `hotspotter.pipeline.identify` |
| `wbia.algo.hots.nn_weights` (669 lines) | `hotspotter.scoring` |
| `wbia.algo.hots.name_scoring` (404 lines) | `hotspotter.name_scoring` |
| `wbia.algo/Config.py` (nested config) | `hotspotter.config` (Pydantic v2) |
| `wbia.vtool.spatial_verification` | `hotspotter._vendor.sver` (stripped, utool-free) |
| `pyhesaff._pyhesaff` (1054 lines) | `hotspotter._vendor.pyhesaff` (stripped, 140 lines) |


## Design

- **Stateless.** Pure functions over numpy arrays. No DB, no network, no `IBEISController`.
- **Deterministic.** Same `(image, config)` → same features, every time.
- **Package-first.** Target boundary is a reusable `hotspotter` library; sidecar/API code is transitional.
- **Single global index** over database descriptors (query excluded), matching WBIA's `NeighborIndex` behaviour.
- **Configurable filter chain.** LNBNN, bar_l2, ratio, FG, and normonly filters combine multiplicatively.
- **Name-level scoring.** fmech (nsum), max-per-name (csum_wbia), per-name sum (sumamech), with canonical alignment.
- **Spatial verification.** RANSAC homography with prescore shortlisting, xy/scale/ori threshold filtering.
- **Zero utool dependency.** All vendored C++ extensions (sver, pyhesaff) are compiled directly; utool/ubelt/six eliminated from the runtime path.

## Architecture

```
wbia-core/
├── Dockerfile                         # Multi-stage: build (cmake/g++) → run (ubuntu:22.04)
├── pyproject.toml                     # numpy, pydantic, opencv, scipy, faiss-cpu
├── src/hotspotter/
│   ├── pipeline.py                    # identify() — orchestrates the full pipeline
│   ├── scoring.py                     # LNBNN weights, FG weights, match building
│   ├── name_scoring.py                # fmech/nsum, csum, canonical alignment
│   ├── spatial.py                     # spatial_verify, make_sver_shortlist
│   ├── knn.py                         # build_global_index, query_index, exact_knn
│   ├── features.py                    # extract_features() — vendored pyhesaff
│   ├── chip.py                        # Chip extraction (mask, resize)
│   ├── data.py                        # FeatureSet, AnnotatedImage, Match, ScoredMatch
│   ├── config.py                      # HotSpotterConfig, IdentificationConfig (Pydantic v2)
│   ├── debug_log.py                   # Pipeline debug tracing (WBIA_CORE_DEBUG=1)
│   ├── trace.py                       # Parquet trace writer for parity comparison
│   └── _vendor/                       # Stripped, utool-free vendored C++ extensions
│       ├── sver/                      # Spatial verification (from wbia-vtool, 6 files)
│       │   ├── _spatial_verification.py
│       │   ├── _sver_c_wrapper.py
│       │   ├── _keypoint.py, _linalg.py, _distance.py, _util_math.py
│       │   └── _sver_cpp/             # sver.cpp + compiled libsver.so
│       └── pyhesaff/                  # Hessian-affine SIFT (3 files + libhesaff.so)
│           ├── _pyhesaff.py
│           ├── _ctypes_interface.py
│           └── lib/                   # compiled libhesaff.so
├── tests/
│   ├── assets/                        # Committed golden + silver trace fixtures
│   ├── test_*.py                      # Unit + integration tests
│   ├── benchmark/                     # COCO multi-target regression suite
│   └── replay/                        # Recorded WBIA fixture replay tests
├── scripts/                           # run_fixture.py, evaluate_groundtruth.py, compare_to_wbia.py, etc.
├── wbia-tpl-pyhesaff/                 # git submodule — C++ Hessian-affine SIFT source
└── wbia-tpl-pyflann/                  # git submodule — C++ FLANN approximate KNN library
```

## Build

Multi-stage Docker. The build stage compiles two C++ extensions; the run
stage copies only the `.so` files and Python venv into a slim
`ubuntu:22.04` image.

| Extension | Build method | Output |
|---|---|---|
| `libsver.so` | `g++ -shared -fPIC -O2 -fopenmp sver.cpp -lopencv_core` | `_vendor/sver/_sver_cpp/libsver.so` |
| `libhesaff.so` | `cmake` (multi-file project from `wbia-tpl-pyhesaff`) | `_vendor/pyhesaff/lib/libhesaff.so` |
| `libflann.so` | `cmake` / scikit-build (from `wbia-tpl-pyflann`) | `pyflann/lib/libflann.so` |

**Image size**: 2.15 GB (down from 9.08 GB pre-extraction). No CUDA,
no cmake, no `-dev` headers in the runtime image.

```bash
make build        # docker build
make test-unit    # run tests in container
make shell        # interactive shell
```


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
| `sv_sver_output_weighting` | `False` | Append homography-error weight per inlier and re-score. WBIA `sver_output_weighting`, defaults False. |

### Output

| Parameter | Default | Description |
|---|---|---|
| `num_return` | 10 | Maximum scored matches returned per query |

### SIFT extraction

| Parameter | Default | Description |
|---|---|---|
| `scale` | `[1.0, 4.0, 8.0]` | Hessian-affine detection scales |
| `ori_hist_bins` | 36 | Orientation histogram bins |
| `ori_hist_threshold` | 0.8 | Orientation peak threshold (0.0–1.0) |


## Config presets

Common configurations for different use cases:

| Preset | Key overrides | Use case |
|---|---|---|
| **Default** | `score_method="nsum"`, `sv_on=True`, `knn_backend="exact"` | Production, deterministic |
| **WBIA parity** | `score_method="nsum_wbia"`, `knn_backend="flann"`, `flann_trees=8`, `flann_random_seed=42` | Bit-faithful WBIA comparison |
| **No SV** | `sv_on=False` | Fast pre-SV scoring only |
| **csum** | `score_method="csum"` | Cumulative sum scoring |
| **Max SV** | `sv_verify_all=True`, `sv_n_annot_per_name=999` | Verify every candidate (WBIA default) |
| **Shortlist SV** | `sv_verify_all=False`, `sv_n_annot_per_name=3` | WBIA literal Config.py default |

Example:
```python
from hotspotter.config import IdentificationConfig
config = IdentificationConfig(
    hotspotter={
        "score_method": "nsum_wbia",
        "knn_backend": "flann",
        "flann_trees": 8,
    }
)
```


## API

### `identify(query_index, database, config) → list[ScoredMatch]`

Run the full identification pipeline for one query against a database of annotated images. Builds a single global index, applies LNBNN scoring with configurable filters, runs name-level aggregation, and optionally spatial verification.

### Data containers

| Class | Purpose |
|---|---|
| `FeatureSet` | Keypoints [N,6] + descriptors [N,128] for one image |
| `AnnotatedImage` | Annotation with features, image, bbox, name_uuid |
| `Match` | Single query-feature → database-feature correspondence |
| `ScoredMatch` | Per-annotation result with score, correspondences, SV inliers |


## Testing

Three-layer test net:

| Layer | Test file | What it checks |
|---|---|---|
| **Unit** | `test_scoring.py`, `test_knn.py`, `test_spatial.py`, ... | Synthetic data, fast, isolated functions |
| **Golden replay** | `test_deterministic_replay.py` | HS-vs-HS bit-exact pre-SV stages against committed golden trace |
| **Silver parity** | `test_wbia_silver_parity.py` | HS-vs-WBIA final_scores decision parity (Top-1 daid agreement) |

```bash
make test-unit    # all unit tests
make test-replay  # golden replay
make test-parity  # silver parity
```


See `docs/development/`:
- `deepseek-wbia-parity.md` - current notes about the work of achiving parity.
- `dependency-stripping-plan.md` — how vtool/pyhesaff/utool were extracted and stripped
- `hotspotter-transition.md` — current boundary checklist for the `hotspotter` rename and `wildlife-id` split
- `parity-analysis.md` — investigation history + benchmark artifact analysis
- `parity-roadmap.md` — implementation plan with verification results
- `wbia-pipeline-differences.md` — per-section WBIA comparison
- `devlog.md` — full development log with all entries
- `executive-summary.html` — self-contained HTML dashboard
