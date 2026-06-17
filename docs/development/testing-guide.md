# wbia-core Testing Guide

## Overview

wbia-core has four testing layers:

| Layer | Scope | Speed | CI | Command |
|---|---|---|---|---|
| **Unit** | Pure functions, config, data containers | < 2s | Always | `pytest tests/ --ignore=tests/benchmark --ignore=tests/replay` |
| **Benchmark** | COCO wildlife dataset, multi-target regression | 5–60 min | On demand | `python tests/benchmark/run_benchmark.py` |
| **Replay** | Recorded WBIA fixtures, parity verification | ~30s | With WBIA | `pytest tests/replay/ -m replay` |
| **Server** | Flask sidecar health + identify endpoints | < 1s | Always | `pytest tests/benchmark/sidecar/test_sidecar.py` |

---

## Layer 1: Unit Tests

Tests individual algorithmic functions against synthetic inputs. No Docker, no network, no WBIA.

**Location:** `tests/`

```
tests/
├── test_features.py    # extract_features() with pyhesaff
├── test_config.py      # Pydantic config models (8 tests)
├── test_data.py        # FeatureSet, AnnotatedImage, ScoredMatch (7 tests)
├── test_knn.py         # FLANN k-NN matching (2 tests)
├── test_pipeline.py    # Full identify() pipeline (12 tests)
├── test_scoring.py     # LNBNN scoring (7 tests)
└── test_spatial.py     # Spatial verification (3 tests)
```

### Run

```bash
# Inside Docker container
docker run --rm --entrypoint bash wbia-core:latest -c \
  "pip install pytest -q && python -m pytest tests/ --ignore=tests/benchmark --ignore=tests/replay -v"

# Or on host with venv
cd wbia-core
pip install -e ".[dev]"
pytest tests/ --ignore=tests/benchmark --ignore=tests/replay -v
```

**Results:** 38 tests in < 2 seconds.

### What they don't cover

- Feature extraction correctness (requires pyhesaff + real images)
- WBIA parity (requires WBIA comparison)
- Real-world performance (synthetic descriptors are tiny)

---

## Layer 2: COCO Benchmark Tests

Multi-target regression suite using a real-world wildlife COCO dataset (4,948 images, 6,925 annotations, giraffe+zebra). Runs subsets through multiple identification backends and compares rankings.

**Location:** `tests/benchmark/`

```
benchmark/
├── run_benchmark.py    # CLI driver
├── runner.py           # Orchestrator — starts targets, runs queries, saves results
├── compare.py          # Cross-target comparator (Spearman ρ, top-k overlap, score delta)
├── analyze.py          # Result analysis CLI (report, fixtures, check)
├── coco/
│   ├── loader.py       # COCO dataset loader with deterministic subset selection
│   └── test_loader.py  # Loader unit tests (7 tests)
├── targets/
│   ├── base.py         # TargetConfig, TargetRunner, QueryResult
│   ├── core.py         # CoreTargetRunner — single-shot POST to wbia-core sidecar
│   ├── wbia.py         # WbiaTargetRunner — multi-step WBIA REST flow
│   └── test_runners.py # Runner tests (3 tests, require Docker)
├── sidecar/
│   └── test_sidecar.py # Flask app endpoint tests (2 tests)
├── test_runner.py      # Benchmark runner integration tests (2 tests)
└── conftest.py         # Shared fixtures
```

### Targets

| Key | Image | Description |
|---|---|---|
| `wbia-core` | `wbia-core:latest` | Single-shot identify via Flask sidecar |
| `wbia-latest` | `wildme/wbia:latest` | Full WBIA pipeline (latest stable) |
| `wbia-nightly` | `wildme/wbia:nightly` | Full WBIA pipeline (nightly build) |
| `wbia-develop` | `wildme/wbia:develop` | Full WBIA pipeline (dev branch) |

### CLI

The benchmark `DEFAULT_CONFIG` in `run_benchmark.py` matches WBIA's
default vsmany pipeline (K=4, Knorm=1, Kpad=0, score_method=nsum/fmech,
fg_on/bar_l2_on/sv_on=False). Config is passed to both wbia-core
(via sidecar → `HotSpotterConfig`) and WBIA (via `query_config_dict`).

```bash
# Quick smoke test — 5 annots, 2 queries, wbia-core only (no comparison)
python tests/benchmark/run_benchmark.py --n-annots 5 --n-queries 2

# Reference-based comparison (fast — no WBIA startup)
python tests/benchmark/run_benchmark.py \
    --n-annots 10 --n-queries 3 \
    --reference tests/benchmark/reference/wbia-latest-10/

# Full comparison — all targets (slow — requires WBIA startup)
python tests/benchmark/run_benchmark.py \
    --n-annots 10 --n-queries 3 \
    --targets wbia-core wbia-latest wbia-nightly wbia-develop

# Large-scale with reference
python tests/benchmark/run_benchmark.py \
    --n-annots 100 --n-queries 10 \
    --reference tests/benchmark/reference/wbia-latest-10/

# Species filter
python tests/benchmark/run_benchmark.py --species zebra_plains --reference tests/benchmark/reference/wbia-latest-10/
```

### Reference results

WBIA is deterministic across versions (ρ=1.0). Results only need to be recorded once:

```
tests/benchmark/reference/
└── wbia-latest-10/       # 10 annots, 3 queries, seed=42
    ├── manifest.json
    ├── query_000/
    │   ├── request.json
    │   └── response.json
    ├── query_001/
    └── query_002/
```

To create a new reference:
```bash
# Record WBIA results (one-time)
python tests/benchmark/run_benchmark.py \
    --n-annots 10 --n-queries 3 \
    --targets wbia-latest \
    --results-dir test-run-results-wbia-ref

# Save as reference
cp -r test-run-results-wbia-ref/target-wbia-latest/ \
    tests/benchmark/reference/wbia-latest-10/
```

All future runs use `--reference` for instant comparison — no WBIA container startup needed.

### Analysis

```bash
# Full report on a completed run
python tests/benchmark/analyze.py report test-run-results-20260606T101918/

# Replay fixtures through sidecar
python tests/benchmark/analyze.py fixtures --fixture-dir tests/replay/testdata/fixtures/

# Cross-check fixtures with benchmark
python tests/benchmark/analyze.py check test-run-results-20260606T101918/
```

### Comparison Metrics

| Metric | Description |
|---|---|
| **Top-1 identical** | All targets agree on #1 match |
| **All rankings match** | Complete ordering agreement across targets |
| **Max score delta** | Largest score difference between targets |
| **Spearman ρ** | Rank correlation coefficient (≥ 0.95 = strong agreement) |
| **Top-3 overlap** | Fraction of top-3 annotations shared between targets |
| **Score distribution** | μ, σ, range per target per query |

### Test data

The COCO dataset lives at `tests/test-dataset/`:

```
tests/test-dataset/
├── annotations/
│   ├── instances_train2020.json
│   ├── instances_val2020.json
│   └── instances_test2020.json
└── images/
    └── train2020/        # 4,948 wildlife images
```

### Running all benchmark tests

```bash
cd wbia-core
# Requires Docker + COCO dataset
pytest tests/benchmark/ -v
# 34 tests total — requires wbia-core:latest image to be built
```

---

## Layer 3: Docker-Based Tests

All tests that require the compiled Docker image (pyhesaff, FLANN, OpenCV).

### Building the test image

```bash
cd wbia-core
docker build -t wbia-core:latest .
```

### Running tests inside the container

```bash
# Full test suite (excluding benchmark + replay)
docker run --rm --entrypoint bash wbia-core:latest -c \
  "pip install pytest -q && python -m pytest tests/ --ignore=tests/benchmark --ignore=tests/replay -v"

# Feature extraction test
docker run --rm --entrypoint bash wbia-core:latest -c \
  "pip install pytest -q && python -m pytest tests/test_features.py -v"

# Server smoke test
docker run -d --name test-server -p 5001:5000 wbia-core:latest
sleep 3
curl http://localhost:5001/api/health/
docker kill test-server
```

### Server entrypoint test

The server entrypoint boots gunicorn. Test with:

```bash
docker run --rm --entrypoint scripts/entrypoints/server-entrypoint.sh -d -p 5001:5000 wbia-core:latest
```

---

## Layer 4: Replay/Fixture Tests

Compare wbia-core against recorded WBIA identification results.

**Location:** `tests/replay/`

```
replay/
├── record_fixtures.py   # Generates synthetic images → WBIA → NPZ fixtures
├── test_replay.py       # Parametrized pytest: replay fixture through wbia-core
├── parity_test.py       # Standalone parity test (runs in Docker)
├── compare_knn.py       # FLANN output comparison
├── compare_knn_wbia.py  # FLANN vs WBIA comparison
├── compare_features.py  # Feature extraction comparison
├── patch_wbia_schema.py # WBIA schema patcher for SQLite mode
├── conftest.py          # Docker compose lifecycle
└── testdata/
    └── fixtures/        # Recorded NPZ fixtures
```

### Recording fixtures

```bash
# Start WBIA
cd tests/replay && docker compose up -d

# Record fixtures (waits for WBIA heartbeat)
python tests/replay/record_fixtures.py

# Run replay tests
pytest tests/replay/ -m replay -v
```

### Replay test flow

1. **Record**: Generate synthetic spot-pattern images → send to WBIA → record NPZ fixtures
2. **Replay**: Load fixture → extract features with pyhesaff → run identify() → compare rankings
3. **Compare**: Top-N overlap, Spearman ρ, score delta vs WBIA's recorded output

### Fixture format

```python
# Each NPZ contains:
{
    "annot_uuids": ["uuid1", "uuid2", ...],     # Annotation UUIDs (query first)
    "name_uuids": ["name_uuid", None, ...],      # Name UUIDs
    "image_bytes": [b"...", b"...", ...],         # PNG-encoded images
    "bboxes": [[x,y,w,h], ...],                   # Bounding boxes
    "species": "zebra_grevys",                    # Species string
    "raw_result": {...},                          # Full WBIA job result JSON
}
```

### Parity test entrypoint

```bash
# Separate entrypoint for standalone parity testing
docker run --rm --entrypoint scripts/entrypoints/test-entrypoint.sh wbia-core:latest
```

---

## Test Quick Reference

```bash
cd wbia-core

# Everything you need for clean test infrastructure
docker build -t wbia-core:latest .                    # Build image once

# --- Run tests ---

# 1. Unit + integration (fast, no Docker needed if pyhesaff installed)
pytest tests/ --ignore=tests/benchmark --ignore=tests/replay -v

# 2. Unit + integration (in Docker, guarantees pyhesaff)
docker run --rm --entrypoint bash wbia-core:latest -c \
  "pip install pytest -q && pytest tests/ --ignore=tests/benchmark --ignore=tests/replay -v"

# 3. Sidecar endpoint tests
docker run --rm --entrypoint bash wbia-core:latest -c \
  "pip install pytest -q && pytest tests/benchmark/sidecar/test_sidecar.py -v"

# 4. COCO benchmark (small)
python tests/benchmark/run_benchmark.py --n-annots 5 --n-queries 2

# 5. COCO benchmark (full comparison, all targets)
python tests/benchmark/run_benchmark.py \
    --n-annots 10 --n-queries 3 \
    --targets wbia-core wbia-latest wbia-nightly wbia-develop

# 6. Analyze results
python tests/benchmark/analyze.py report test-run-results-*

# 7. Replay/fixture tests (requires WBIA + recorded fixtures)
cd tests/replay && docker compose up -d && python record_fixtures.py
cd ../.. && pytest tests/replay/ -m replay -v

# 8. All benchmark + unit tests combined
pytest tests/ --ignore=tests/replay -v
```

## Test Commands Summary Table

| Command | What it tests | Runtime | Prerequisites |
|---|---|---|---|
| `pytest tests/ --ignore=tests/benchmark --ignore=tests/replay` | Unit + pipeline + scoring | < 2s | pyhesaff (or Docker) |
| `pytest tests/benchmark/sidecar/test_sidecar.py` | Flask app endpoints | < 1s | Docker |
| `python tests/benchmark/run_benchmark.py --n-annots 5 --n-queries 2` | COCO smoke test | ~1 min | Docker + COCO dataset |
| `python tests/benchmark/run_benchmark.py --n-annots 200 --n-queries 20 --targets wbia-core wbia-latest` | Large-scale regression | 30–60 min | Docker + COCO dataset + WBIA images |
| `pytest tests/replay/ -m replay` | WBIA fixture replay | ~30s | WBIA container + recorded fixtures |
| `python tests/benchmark/analyze.py report <dir>` | Result analysis | < 5s | Completed benchmark run |
