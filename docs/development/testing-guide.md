# Hotspotter Testing Guide

## Overview

Hotspotter has four testing layers:

| Layer | Scope | Speed | Command |
|---|---|---|---|
| **Unit** | Config, data, KNN, pipeline, scoring, spatial | < 2s | `make test-unit` |
| **Replay** | Recorded WBIA fixtures, parity verification | ~5s | `make test-replay` |
| **Parity** | Full pipeline vs WBIA oracle parquet comparison | ~2 min | `make test-parity` |
| **Benchmark** | COCO wildlife dataset, multi-target regression | 5–60 min | `python tests/benchmark/run_benchmark.py` |

All tests run inside the Docker image (`hotspotter:latest`).

---

## Layer 1: Unit Tests (42 tests)

Tests individual functions against synthetic inputs. No network, no WBIA.

**Location:** `tests/`

```
tests/
├── test_config.py      # Pydantic config models (8 tests)
├── test_data.py        # FeatureSet, AnnotatedImage, ScoredMatch (6 tests)
├── test_features.py    # extract_features() with pyhesaff (2 tests)
├── test_knn.py         # FLANN k-NN matching (2 tests)
├── test_pipeline.py    # Full identify() pipeline, knorm, dynamic Kpad (10 tests)
├── test_scoring.py     # LNBNN scoring (9 tests)
└── test_spatial.py     # Spatial verification (3 tests)
```

### Run

```bash
# Via Makefile (recommended)
make test-unit

# Direct Docker
docker run --rm --entrypoint bash hotspotter:latest -c \
  "pip install pytest -q && python -m pytest tests/ -q --ignore=tests/benchmark --ignore=tests/replay"
```

---

## Layer 2: Replay Tests (84 tests)

Compare hotspotter against recorded WBIA identification NPZ fixtures.

**Location:** `tests/replay/`

```bash
make test-replay
```

12 fixture sets across 3 species (giraffe_reticulated, whale_shark, zebra_grevys).
Each set: fixture loading, ranking, self-exclusion, correspondences, spatial
verification.

---

## Layer 3: Parity Tests

Runs hotspotter against the same test images as the WBIA oracle, writes parquet
checkpoints, and compares them stage-by-stage using `compare_wbia_oracles.py`.

```bash
# Default: compare against nightly oracle
make test-parity

# Specify oracle directory and threshold
make test-parity ORACLE=../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-144646 PARITY_RHO=0.97

# Direct script
python3 scripts/compare_to_wbia.py \
    ../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-144646 \
    --passing-rho 0.97
```

9 configs × 3 queries each. Parquet traces written for 10 stages: chips,
annotations, features_keypoints, features_descriptors, nearest_neighbors,
baseline_neighbor_filter, neighbor_weights, chipmatches_pre_sv,
chipmatches_post_sv, final_scores.

Exit code 0 = PASS (ρ ≥ 0.97), 2 = parity FAIL, 1 = fatal error. Current
hotspotter-vs-WBIA parity is expected to fail (ρ = 0.3031) while Phase 2
algorithm gaps are being closed. Status: features 100% identical, neighbor
IDs 73% match, remaining gap in scoring amplification and SV semantics.

---

## Layer 4: COCO Benchmark Tests

Multi-target regression suite using real-world wildlife COCO dataset.

**Location:** `tests/benchmark/`

```bash
# Quick smoke test — 5 annots, 2 queries
python tests/benchmark/run_benchmark.py --n-annots 5 --n-queries 2

# Full comparison — all targets
python tests/benchmark/run_benchmark.py \
    --n-annots 10 --n-queries 3 \
    --targets hotspotter wbia-latest wbia-nightly wbia-develop
```

### Targets

| Key | Image | Description |
|---|---|---|
| `hotspotter` | `hotspotter:latest` | Hotspotter library (no service) |
| `wbia-latest` | `wildme/wbia:latest` | Full WBIA pipeline (latest stable) |
| `wbia-nightly` | `wildme/wbia:nightly` | Full WBIA pipeline (nightly build) |
| `wbia-develop` | `wildme/wbia:develop` | Full WBIA pipeline (dev branch) |

---

## Building

```bash
make build          # docker build -t hotspotter:latest .
```

The Docker image source-builds three vendored submodules (`wbia-utool`,
`wbia-vtool`, `wbia-tpl-pyhesaff`) in dependency order, then installs the
`hotspotter` package with pandas + pyarrow for trace support. No Flask or
service dependencies.

---

## Test Commands Summary

| Command | Tests | Runtime |
|---|---|---|
| `make test-unit` | 42 unit tests | < 2s |
| `make test-replay` | 84 replay tests | ~5s |
| `make test-parity` | Parity vs WBIA oracle | ~2 min |
| `make test-all` | All pytest tests | ~10s |
| `make test-benchmark` | Benchmark pytest tests | ~1 min |
| `make shell` | Interactive shell in container | — |

## Running Scripts

```bash
# Run fixture pipeline (loads images, extracts chips, runs identify)
docker run --rm \
    -v ../pipeline/tests:/app/pipeline/tests \
    hotspotter:latest \
    python scripts/run_fixture.py pipeline/tests/reference_batch.json

# With config override
docker run --rm \
    -v ../pipeline/tests:/app/pipeline/tests \
    hotspotter:latest \
    python scripts/run_fixture.py pipeline/tests/reference_batch.json \
    --config '{"sv_on": true, "knn": 6}'

# With parquet tracing
docker run --rm \
    -v ../pipeline/tests:/app/pipeline/tests \
    -v /tmp/traces:/app/traces \
    -e HOTSPOTTER_TRACE_DIR=/app/traces \
    -e HOTSPOTTER_TRACE_CONFIG_LABEL=sv_on_true \
    hotspotter:latest \
    python scripts/run_fixture.py pipeline/tests/reference_batch.json
```
