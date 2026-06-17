# Benchmark Tests

Parity testing infrastructure — ensures the `wbia-core` stateless pipeline produces equivalent rankings to the legacy WBIA container.

## Architecture

Three phases run sequentially:

### 1. Data Loading (`coco/loader.py`)

Reads COCO-format dataset from `tests/test-dataset/` (zebra/giraffe images). `CocoLoader.select_subset()` randomly samples annotations controlled by `--seed` for determinism. Returns a `CocoSubset` with annotations, query indices, and image bytes.

### 2. Query Execution (`runner.py` + `targets/*.py`)

`run_benchmark()` orchestrates Docker containers for each target backend, then runs queries:

- **`CoreTargetRunner`** (`targets/core.py`) — Starts `wbia-core:latest`, single `POST /api/v1/identify/`.
- **`WbiaTargetRunner`** (`targets/wbia.py`) — Starts `wildme/wbia:*`, multi-step REST flow with image server.

Each query's request/response is saved to `test-results/test-run-results-<timestamp>/target-<name>/query_NNN/`.

### 3. Comparison (`compare.py` + `analyze.py`)

Computes per-query and global metrics:

| Metric | Description |
|--------|-------------|
| Top-1 identical | All backends agree on single best match |
| Rankings match | Full score ordering is identical |
| Max score delta | Largest score difference for any shared annotation |
| Spearman rho | Pairwise rank correlation between backends |
| Top-3 overlap | Fraction of top-3 that appears in the other's top-3 |
| Top-1/3/5 accuracy | Ground-truth individual appears in top-K |
| MRR | Mean Reciprocal Rank of first correct match |

Results are written as `summary.json`. View via CLI (`analyze.py report`) or web dashboard (`web_view.py` on port 8080).

## File map

```
tests/benchmark/
├── run_benchmark.py       # Main CLI entry point
├── runner.py              # Orchestrator: Docker lifecycle, query loop, result I/O
├── compare.py             # Core comparison engine (Spearman, accuracy, overlap)
├── analyze.py             # Standalone analysis CLI: report, fixtures, check
├── web_view.py            # Bottle web dashboard for browsing results
├── test_runner.py         # Test suite: mock targets, _strip_images, error handling
├── conftest.py            # Adds benchmark/ to sys.path for imports
│
├── coco/
│   ├── loader.py          # COCO JSON + image reader, CocoSubset sampler
│   └── test_loader.py     # Tests: annotation count (6925), determinism, filtering
│
├── targets/
│   ├── base.py            # TargetConfig, QueryResult dataclasses, TargetRunner ABC
│   ├── core.py            # CoreTargetRunner: wbia-core via single POST
│   ├── wbia.py            # WbiaTargetRunner: WBIA multi-step REST flow
│   └── test_runners.py    # WBIA result normalisation unit tests
│
├── sidecar/
│   └── test_sidecar.py    # Flask sidecar integration test
│
└── .chip_cache/           # Cached annotation chips for web_view.py
```

## Usage

### Full benchmark (wbia-core vs WBIA, needs both Docker images)

```bash
python3 tests/benchmark/run_benchmark.py \
  --n-annots 100 --n-queries 20 \
  --species zebra_plains --seed 42
```

### Run pytest tests

```bash
make test-benchmark              # COCO loader + runner tests (needs --volume mount for test-dataset)
make test-benchmark-runner       # Mock target tests (needs Docker socket + host network)
```

### Analyse results

```bash
# CLI report
python3 tests/benchmark/analyze.py report test-results/<run-dir>/
python3 tests/benchmark/analyze.py report test-results/<run-dir>/ --json  # machine-readable

# Web dashboard
python3 tests/benchmark/web_view.py   # → http://localhost:8080
```

### Fixture vs live sidecar

```bash
python3 tests/benchmark/analyze.py fixtures
python3 tests/benchmark/analyze.py check
```

## Dependencies

- **Docker** — `run_benchmark.py` uses `docker` CLI (must run from host, not inside container)
- **`/var/run/docker.sock`** — required for `test_runner.py` (mount + host network)
- **COCO dataset** at `tests/test-dataset/` — dockerignored, volume-mounted at runtime
- **Docker images**: `wbia-core:latest`, optionally `wildme/wbia:latest`
- Pure-Python Spearman rho (no scipy dependency)
