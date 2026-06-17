# COCO Benchmark — Multi-Target Regression Testing

## Goal

Run the same COCO subset through multiple identification backends and prove
all produce identical results.  Each backend runs in its own Docker container
and exposes a REST API.  A driver script orchestrates the full lifecycle:
pick subset → fan out to targets → collect results → compare.

## Targets

| Tag | Image | Description |
|-----|-------|-------------|
| `wildme/wbia:latest` | `wbia` | Current stable WBIA release |
| `wildme/wbia:nightly` | `wbia` | Nightly WBIA build |
| `wildme/wbia:dev` | `wbia` | Bleeding-edge WBIA |
| `wbia-core:latest` | `wbia-core` | Local wbia-core sidecar |

All targets receive the same inputs and produce the same output schema.

## Architecture

```
run_benchmark.py  (driver — single entry point)
    │
    ├── coco/               COCO subset loader
    │   └── loader.py       read COCO JSON, load images, yield subset
    │
    ├── targets/            backend drivers (one per target type)
    │   ├── base.py         abstract TargetRunner
    │   ├── wbia.py         WBIA-style multi-step REST (add → annot → query → poll)
    │   └── core.py         wbia-core single-shot REST
    │
    ├── runner.py            orchestrator: for each target → start → run → save → stop
    │
    ├── compare.py           result comparator: diff across targets
    │
    └── sidecar/             Flask app + Dockerfile for wbia-core container
        ├── app.py
        ├── requirements.txt
        └── Dockerfile
```

## COCO Subset Loader

Reads `instances_train2020.json` and corresponding JPEGs.

```
CocoLoader(
    coco_json="tests/test-dataset/annotations/instances_train2020.json",
    image_dir="tests/test-dataset/images/train2020",
)
```

```python
def select_subset(
    n_annots: int = 100,
    species: str | None = None,        # filter by species
    seed: int = 42,                     # deterministic shuffle
    n_queries: int = 10,                # how many to mark as queries
) -> Subset:
    """Returns a Subset with images, annotations, and query indices."""

@dataclass
class Subset:
    images: list[ImageData]             # loaded JPEG bytes
    annotations: list[Annotation]       # COCO annotations with bbox, species, individual_ids
    query_indices: list[int]            # indices into annotations to use as queries

@dataclass
class ImageData:
    image_id: int
    bytes: bytes                         # raw JPEG bytes
    width: int
    height: int

@dataclass
class Annotation:
    annot_id: int
    image_id: int
    bbox: tuple[int, int, int, int]      # xywh
    species: str
    individual_ids: list[str]            # ground truth names
```

### Name mapping

Each COCO annotation has `individual_ids: list[str]`. For identification
ground truth we use `individual_ids[0]` as the primary name.

## Output Schema

All targets produce the same per-query result:

```json
{
  "query_annot_index": 0,
  "annot_scores": [
    {"aid": "coco-annot-001", "score": 12.34, "num_matches": 42, "rank": 1},
    {"aid": "coco-annot-002", "score": 8.56, "num_matches": 31, "rank": 2}
  ],
  "config": {
    "pipeline_root": "vsmany",
    "K": 4,
    "fg_on": true,
    "sv_on": false
  },
  "timing_ms": 1234
}
```

## API Contract: wbia-core Sidecar

Single-shot, no persistent database.  Accepts all data in one request.

### `POST /api/v1/identify/`

```json
{
  "query_image": "<base64-encoded JPEG bytes>",
  "query_bbox": [20, 10, 260, 180],
  "query_theta": 0.0,
  "database": [
    {
      "aid": "coco-annot-001",
      "image_bytes": "<base64-encoded JPEG bytes>",
      "bbox": [10, 20, 200, 150],
      "theta": 0.0,
      "name_uuid": "indiv-abc-123"
    }
  ],
  "config": {
    "pipeline_root": "vsmany",
    "K": 4,
    "fg_on": false,
    "sv_on": false
  }
}
```

Response:

```json
{
  "status": "completed",
  "response": {
    "annot_scores": [
      {"aid": "coco-annot-001", "score": 12.34, "num_matches": 42},
      {"aid": "coco-annot-002", "score": 8.56, "num_matches": 31}
    ],
    "timing_ms": 567
  }
}
```

### `GET /api/health/`

```json
{"status": "ok", "version": "0.1.0"}
```

## API Contract: WBIA Targets

The driver wraps the standard WBIA multi-step flow to produce the same output
schema.  Internally it:

1. `POST /api/image/json/` — upload images
2. `POST /api/annot/json/` — create annotations (bbox, species)
3. `POST /api/engine/query/graph/` — start identification job
4. `POST /api/engine/job/result/` — poll until complete
5. Parse `annot_score_list` from the result

The driver normalises WBIA's response into the standard `annot_scores` array
so the comparison step is target-agnostic.

## Driver CLI

```
usage: run_benchmark.py [-h] --n-annots N [--n-queries N] [--species S]
                        [--seed N] [--targets T [T ...]]
                        [--coco-json PATH] [--coco-images PATH]
                        [--results-dir PATH] [--keep-containers]

Run COCO subset against one or more identification backends.

options:
  --n-annots N           Total annotations in the subset (default: 100)
  --n-queries N          Number of query annotations (default: 10)
  --species S            Filter: giraffe_masai, zebra_plains, or all
  --seed N               Deterministic shuffle seed (default: 42)
  --targets T [T ...]    Targets to test (default: wbia-core)
                         Choices: wbia-core, wbia-latest, wbia-nightly, wbia-dev
  --coco-json PATH       Path to COCO JSON (default: tests/test-dataset/...)
  --coco-images PATH     Path to COCO images directory
  --results-dir PATH     Output directory (default: test-run-results-<timestamp>)
  --keep-containers      Don't stop containers after run
```

### Output structure

```
test-run-results-20260605T120000/
├── summary.json                     # cross-target comparison
├── config.json                      # CLI args + subset info
├── target-wbia-latest/
│   ├── manifest.json                # target info, container logs
│   ├── query_000/                    # one dir per query
│   │   ├── request.json
│   │   ├── response.json
│   │   └── scores.json              # parsed annot_scores (canonical format)
│   ├── query_001/
│   └── ...
├── target-wbia-nightly/
│   └── ...
├── target-wbia-dev/
│   └── ...
└── target-wbia-core/
    └── ...
```

## Comparison

`compare.py` reads all target result dirs and computes per-query and
aggregate agreement metrics:

| Metric | Description |
|--------|-------------|
| **Top-1 agreement** | All targets return same `aid` at rank 1 |
| **Set overlap**    | Jaccard similarity of the returned annotation sets |
| **Score correlation** | Pairwise Pearson ρ between score vectors |
| **Rank correlation** | Pairwise Spearman ρ between ranking vectors |
| **Max score delta**  | Max absolute difference for any common (target, annot) pair |

A diff report is generated even on mismatch (not a hard pass/fail) so the
user can inspect what changed.

## Sidecar Implementation (wbia-core)

A minimal Flask application running inside a Docker container.

```
sidecar/
├── Dockerfile
├── requirements.txt
└── app.py
```

### `requirements.txt`

```
wbia-core @ file:///wbia-core
flask>=3.0
gunicorn>=22.0
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "1", "app:app"]
```

### Endpoints

- `GET /api/health/` → health check
- `POST /api/v1/identify/` → single-shot identification

The sidecar:
1. Decodes base64 images
2. Calls `wbia_core.extract_features()` for each database annotation
3. Calls `wbia_core.identify()` for each query
4. Returns scores matching the canonical schema

## WBIA Multi-Step Wrapper

The `WbiaTargetRunner` handles the WBIA-specific dance:

1. Start HTTP server to serve images to the WBIA container
2. `POST /api/image/json/` with `http://host.docker.internal:<port>/<filename>`
3. `POST /api/annot/json/` with bboxes and species
4. `POST /api/engine/query/graph/` with `pipeline_root: vsmany`
5. Poll until complete
6. Parse `annot_score_list` from result
7. Normalise into canonical `annot_scores` format

## Implementation Order

1. **Sidecar** — Flask app + Dockerfile + `wbia-core` dependency
   - Build and test locally with curl
   - Verify `extract_features` + `identify` round-trip

2. **COCO loader** — `coco/loader.py`
   - Parse COCO JSON, load images, `select_subset()`
   - Unit test with 5-annot subset

3. **Target runners** — `targets/base.py`, `targets/core.py`, `targets/wbia.py`
   - `CoreTargetRunner`: POST to sidecar → save response
   - `WbiaTargetRunner`: multi-step → parse → normalise → save

4. **Runner orchestrator** — `runner.py`
   - For each target: start container → wait healthy → run queries → save → stop
   - Docker compose per target with unique port mapping

5. **Comparator** — `compare.py`
   - Load all result dirs, compute metrics, write `summary.json`

6. **CLI driver** — `run_benchmark.py`
   - argparse → orchestrate → print summary

7. **Integration test** — one COCO subset through wbia-core sidecar
   - `pytest tests/benchmark/` — smoke test with 5 annots, 1 query

## Implementation notes

- The sidecar MUST pin the same `wbia-pyhesaff` version as WBIA to ensure
  identical feature extraction.
- WBIA containers use `extra_hosts: host.docker.internal:host-gateway` so the
  container can reach the driver's image server.
- For wbia-core sidecar there is no image server — images are sent inline
  as base64.
- Results dir is gitignored (`.gitignore` entry for `test-run-results-*`).
- All timestamps in ISO 8601.
- Container tags are read from environment or CLI, not hardcoded.
