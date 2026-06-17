# COCO Benchmark — Source of Truth API Contract

This document defines the data contracts for all components in the
multi-target regression testing system.  Every component reads and writes
the schemas defined here.  If the schemas don't change, no component needs
to know about any other component.

## 1. COCO Subset Schema (`coco` → `runner`)

```python
@dataclass
class CocoAnnotation:
    annot_id: int                      # COCO annotation id
    image_id: int                      # foreign key into COCO images
    bbox: tuple[int, int, int, int]    # xywh in pixel coords
    species: str                       # "giraffe_masai" | "zebra_plains"
    individual_ids: list[str]          # ground truth names (use [0] as primary)
    image: bytes                       # raw JPEG bytes, loaded from disk
    width: int                         # image pixel width
    height: int                        # image pixel height

@dataclass
class CocoSubset:
    annotations: list[CocoAnnotation]  # all annotations in the subset
    query_indices: list[int]           # indices into annotations[] to use as queries
    config: dict                       # seed, species filter, n_annots, etc.
```

Serialised to JSON for inter-component transport:

```json
{
  "config": {"n_annots": 100, "n_queries": 10, "species": null, "seed": 42},
  "query_indices": [0, 5, 12, 33, ...],
  "annotations": [
    {
      "annot_id": 1,
      "image_id": 100,
      "bbox": [120, 80, 400, 300],
      "species": "zebra_plains",
      "individual_ids": ["indiv_001"],
      "image_b64": "<base64-encoded JPEG>",
      "width": 2000,
      "height": 3000
    }
  ]
}
```

## 2. Sidecar Request Schema (`runner` → `wbia-core sidecar`)

### `POST /api/v1/identify/`

```json
{
  "query_image_b64": "<base64-encoded JPEG bytes>",
  "query_bbox": [120, 80, 400, 300],
  "query_theta": 0.0,
  "database": [
    {
      "aid": "coco-annot-1",
      "image_b64": "<base64-encoded JPEG bytes>",
      "bbox": [10, 20, 200, 150],
      "theta": 0.0,
      "name_uuid": null
    }
  ],
  "config": {
    "pipeline_root": "vsmany",
    "K": 4,
    "Knorm": 1,
    "Kpad": 0,
    "fg_on": false,
    "sv_on": false
  }
}
```

### `GET /api/health/`

No body.  Response:

```json
{"status": "ok", "version": "0.1.0", "service": "wbia-core"}
```

## 3. Sidecar Response Schema (`wbia-core sidecar` → `runner`)

### `200 OK`

```json
{
  "status": "completed",
  "response": {
    "annot_scores": [
      {"aid": "coco-annot-1", "score": 12.34, "num_matches": 42},
      {"aid": "coco-annot-2", "score": 8.56, "num_matches": 31}
    ],
    "timing_ms": 567
  }
}
```

### Error

```json
{
  "status": "error",
  "message": "Failed to extract features for coco-annot-5: HessianAffine error"
}
```

## 4. WBIA Normalised Response Schema (WBIA REST → `runner`)

The `WbiaTargetRunner` normalises WBIA's raw job result into the same
canonical format as the sidecar response:

```json
{
  "status": "completed",
  "response": {
    "annot_scores": [
      {"aid": "coco-annot-1", "score": 12.34, "num_matches": 42},
      {"aid": "coco-annot-2", "score": 8.56, "num_matches": 31}
    ],
    "timing_ms": 1234
  }
}
```

Normalisation logic:

```python
def normalise_wbia_result(raw: dict, annot_uuids: list[str]) -> list[dict]:
    """Parse WBIA's annot_score_list + num_matches into canonical annot_scores."""
    assert raw["status"] == "completed"
    annot_scores = raw["response"]["annot_score_list"]
    num_matches = raw["response"]["num_matches_list"]
    return [
        {"aid": annot_uuids[i], "score": float(s), "num_matches": int(n)}
        for i, (s, n) in enumerate(zip(annot_scores, num_matches))
    ]
```

### WBIA-specific config mapping

The `WbiaTargetRunner` maps the canonical config to whatever WBIA expects:

| Canonical key | WBIA query_config_dict key |
|---------------|---------------------------|
| `pipeline_root` | `pipeline_root` |
| `sv_on` | `sv_on` |
| `fg_on` | `fg_on` |
| `K` | `K` |

Always passes `"pipeline": "vsmany"` and `"pipeline_root": "vsmany"`.

## 5. Results Directory Schema (`runner` → disk)

### `test-run-results-<timestamp>/`

```
test-run-results-<YYYYMMDDTHHMMSS>/
├── summary.json                          # cross-target comparison
├── config.json                           # CLI args + subset info (frozen copy)
├── target-<name>/
│   ├── manifest.json                     # target info
│   ├── query_000/
│   │   ├── request.json                  # what was sent
│   │   └── response.json                 # raw response + canonical scores
│   ├── query_001/
│   │   ├── request.json
│   │   └── response.json
│   └── ...
├── target-<name>/
└── ...
```

### `manifest.json`

```json
{
  "target": "wbia-core",
  "image": "wbia-core:latest",
  "container_id": "abc123...",
  "started_at": "2026-06-05T12:00:00Z",
  "finished_at": "2026-06-05T12:05:00Z",
  "n_queries": 10,
  "total_timing_ms": 12345
}
```

### `query_NNN/request.json`

```json
{
  "query_annot_index": 0,
  "query_annot_id": 12345,
  "n_database_annots": 100,
  "config": {"pipeline_root": "vsmany", "K": 4, "fg_on": false, "sv_on": false}
}
```

(image data is NOT written to disk — only metadata)

### `query_NNN/response.json`

```json
{
  "response": {
    "annot_scores": [
      {"aid": "coco-annot-1", "score": 12.34, "num_matches": 42}
    ],
    "timing_ms": 567
  }
}
```

## 6. Summary Schema (`compare` → `summary.json`)

```json
{
  "run_id": "20260605T120000",
  "config": {"n_annots": 100, "n_queries": 10, "species": null, "seed": 42},
  "targets": ["wbia-core", "wbia-latest", "wbia-nightly", "wbia-dev"],
  "agreement": {
    "top1_identical": true,
    "all_rankings_match": true,
    "max_score_delta": 0.0,
    "spearman_below_pairs": []
  },
  "per_query": [
    {
      "query_index": 0,
      "top1_aids": {
        "wbia-core": "coco-annot-1",
        "wbia-latest": "coco-annot-1",
        "wbia-nightly": "coco-annot-1",
        "wbia-dev": "coco-annot-1"
      },
      "max_score_delta": 0.000001,
      "spearman_pairs": [
        {"a": "wbia-core", "b": "wbia-latest", "rho": 1.0},
        {"a": "wbia-core", "b": "wbia-nightly", "rho": 1.0},
        {"a": "wbia-core", "b": "wbia-dev", "rho": 1.0}
      ]
    }
  ],
  "errors": []
}
```

## 7. Config passed to identification

Default config used for all targets.  Overridable via CLI.

```json
{
  "pipeline_root": "vsmany",
  "K": 4,
  "Knorm": 1,
  "Kpad": 0,
  "fg_on": false,
  "sv_on": false
}
```

## 8. Error handling

All components follow the same convention:

- **Success:** `{"status": "completed", "response": {...}}`
- **Error:** `{"status": "error", "message": "description"}`
- **Timeout:** `{"status": "timeout", "message": "Job X not complete after 300s"}`
- **Infrastructure failure** (container won't start, network error): raise and let
  the driver decide whether to skip the target or abort.

Errors are recorded in `response.json` alongside any partial results.  The
`summary.json` has an `errors` array with per-target, per-query entries.

## 9. Version history

| Date | Change |
|------|--------|
| 2026-06-05 | Initial contract |
