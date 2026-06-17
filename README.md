# wbia-core

Stateless HotSpotter wildlife re-identification pipeline — extracted from `wildbook-ia`, distributed as a pure-Python pip package.

## What it does

`wbia-core` provides the algorithmic primitives for spot-pattern animal identification. Given a query image and a database of annotated images, it extracts Hessian-affine SIFT features, performs k-nearest-neighbor search, applies LNBNN (Local Naive Bayes Nearest Neighbor) scoring with configurable filters, runs name-level aggregation, and optionally applies spatial verification via RANSAC homography.

```python
from wbia_core import identify, IdentificationConfig, HotSpotterConfig

config = IdentificationConfig(
    hotspotter=HotSpotterConfig(knn=4, sv_on=False)
)
results = identify(query_index=0, database=annotated_images, config=config)
# → list[ScoredMatch] sorted by score descending
```

## What it replaces

| `wildbook-ia` | `wbia-core` |
|---|---|
| `wbia.algo.hots.chip_match` (3039 lines) | `wbia_core.data` (dataclasses) |
| `wbia.algo.hots.pipeline` (1655 lines) | `wbia_core.pipeline.identify` (one function) |
| `wbia.algo.hots.nn_weights` (669 lines) | Inlined filter chain in `pipeline.py` |
| `wbia.algo.hots.name_scoring` (404 lines) | `wbia_core.name_scoring` |
| `wbia.algo.hots.scoring` (177 lines) | `wbia_core.scoring` |
| `wbia.algo.hots.neighbor_index` (1118 lines) | `wbia_core.knn` |
| `wbia.algo/Config.py` (nested config) | `wbia_core.config` (Pydantic v2) |

What it does **not** replace: detection (YOLO/MegaDetector), classification, embedding extraction, the `IBEISController`, the depcache, the ZMQ job engine, or the `wbia_*` PostgreSQL schemas. Those belong to `ml-service` and `wildlife-id`.

## Design

- **Stateless.** Pure functions over numpy arrays. No DB, no network, no `IBEISController`.
- **Deterministic.** Same `(image, config)` → same features, every time.
- **Single global FLANN index** over all database descriptors (including the query), matching WBIA's `NeighborIndex` behaviour.
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
├── src/wbia_core/
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

```bash
git clone --recursive git@github.com:WildMeOrg/wildbook-infra.git
```

## Build & run

```bash
cd wbia-core
docker build -t wbia-core:latest .       # ~3 min

# Server
make server
curl http://localhost:5000/api/health/

# Identify
curl -X POST http://localhost:5000/api/v1/identify/ \
  -H "Content-Type: application/json" -d @request.json

# Unit tests
make test-unit                           # 38 tests, <3s

# Benchmark (from host)
python3 tests/benchmark/run_benchmark.py \
  --targets wbia-core wbia-develop \
  --n-annots 15 --n-queries 2 --seed 10
```

## Configuration

All pipeline parameters live in `HotSpotterConfig` (Pydantic v2 model):

| Parameter | Default | Description |
|---|---|---|
| `knn` | 4 | Number of voting neighbors |
| `kpad` / `kpad_policy` | 0 / `"fixed"` | Extra columns for self-filtering |
| `score_method` | `"csum"` | `csum`, `nsum`, `csum_wbia`, `nsum_wbia`, `sumamech` |
| `normalizer_rule` | `"last"` | `"last"` or `"name"` (different-name validation) |
| `fg_on` | `True` | Foreground weighting |
| `bar_l2_on` | `False` | `1.0 - vdist` filter |
| `ratio_thresh` | `None` | Lowe's ratio test threshold |
| `sv_on` | `True` | Spatial verification |
| `can_match_samename` | `True` | Allow matches to same-name annotations |
| `rotation_invariance` | `False` | XY-dedup in fmech scoring |
| `flann_trees` / `flann_checks` | 8 / 800 | FLANN index parameters |

See `wbia_core/config.py` for the full list.

## API

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

## Parity status

Algorithmically equivalent to WBIA HotSpotter's `vsmany` pipeline as of
2026-06-08d. Two benchmark bugs in the wbia-develop target runner
(`targets/wbia.py`) were discovered and fixed: reading the wrong
score field and missing annotation name assignment. After fixes,
top-1 agrees on 11/12 queries and scores match within 1%.

See `docs/development/`:
- `parity-analysis.md` — investigation history + benchmark artifact analysis
- `parity-roadmap.md` — implementation plan with verification results
- `wbia-pipeline-differences.md` — per-section WBIA comparison
- `devlog.md` — full development log with all entries
- `executive-summary.html` — self-contained HTML dashboard
