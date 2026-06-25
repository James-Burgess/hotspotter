# HotSpotter Package Transition

This repo is currently named `wbia-core`, but the target package is `hotspotter`: a reusable HotSpotter algorithm library. The current sidecar and end-to-end identify flow are useful prototype/benchmark code, but production service and indexing responsibilities should move to `wildlife-id`.

## Target Boundary

`hotspotter` owns reusable HotSpotter algorithms:

- WBIA-compatible chip preprocessing helpers needed for HotSpotter parity.
- Hessian-affine/SIFT feature extraction and feature config/version/hash helpers.
- Data containers such as `FeatureSet`, `AnnotatedImage`, `Match`, and `ScoredMatch`.
- Stateless KNN helpers for standalone/local use.
- LNBNN weighting and filter primitives.
- Name scoring: fmech/nsum, max-csum, sumamech, canonical alignment.
- Spatial verification primitives over supplied keypoints/correspondences.
- Parquet trace writer for pipeline-stage checkpoints (WBIA-compatible schema).
- Deterministic fixture tests against WBIA behavior.

`hotspotter` should not own:

- Flask/FastAPI service APIs.
- Persistent feature stores.
- Long-lived FAISS/FLANN index lifecycle.
- Wildbook-specific request/response contracts.
- Project/site/species permission filters.
- MiewID, CLIP, YOLO, or other learned model inference.
- Production candidate search orchestration.

`wildlife-id` owns the stateful identification service: feature ingestion, persistence, candidate indexes, filtering, matching orchestration, and the production `/api/identify/` API. It should import `hotspotter`.

`pipeline` owns batch orchestration: process images, run detectors/embedders, import `hotspotter` for HotSpotter features, send feature payloads to `wildlife-id`, and compare old/new flows.

## Phase 1 Completed

### 1. Rename And Package — DONE

- Package renamed from `wbia_core` → `hotspotter` (`src/hotspotter/`).
- `pyproject.toml` updated with name, description.
- All imports updated in tests, scripts, and benchmarks.
- `wbia_core` compatibility shim kept in `src/wbia_core/__init__.py` re-exporting from `hotspotter`.
- `make_sver_shortlist` import bug fixed (`__all__` declared but not imported).

### 2. Move Chip Code Into The Library — DONE

- `_compute_affine_matrix()` and `extract_chip()` moved from `sidecar/api.py` into `hotspotter.chip`.
- Public API: `hotspotter.chip.extract_chip(img, bbox)`.
- Supports bbox, theta, resize policy, dimension size.
- FIXED: negative bbox coordinates now pass directly to `cv2.warpAffine` with `BORDER_CONSTANT` (matching WBIA's `extract_chip_from_img`).

### 3. Quarantine Or Move The Sidecar — DONE

- `sidecar/api.py` deleted. Flask removed from Dockerfile.
- Image entrypoint removed; docker image is now a pure library with bash as default command.
- `scripts/run_fixture.py` replaces sidecar for testing: loads images, extracts chips, runs `identify()`, prints JSON.
- Database built in **batch file order** (not queries-first) to match WBIA's descriptor stacking order.
- `scripts/compare_to_wbia.py` runs all configs and compares parquet traces against WBIA oracle.

### 4. WBIA Oracle Testing Infrastructure — DONE

- WBIA monkeypatch (`patches/wbia_parquet_trace.py`) writes parquet + `.npy` sidecars.
- In-container recorder (`patches/wbia_record_oracle_incontainer.py`) runs 9 configs × 3 queries.
- Hotspotter trace writer (`hotspotter.trace`) uses same schema and naming convention.
- File naming: `{config_label}_{query_index:06d}.parquet` with `trace_manifest.json`.
- Comparison script (`../scripts/compare_wbia_oracles.py`) with `--passing-rho 0.97` parity gate.

### 5. Parity Discrepancies Documented — DONE

- `docs/development/hotspotter-parity-discrepancies.md`: root causes ranked by impact with measured metrics.

## Phase 2 Completed

### FLANN Index — Query Excluded (Matching WBIA)

- `identify()` builds the FLANN index over database descriptors only (query excluded).
- AGENTS.md directive updated to reflect WBIA reality (WBIA indexes `qreq_.daids`, not qaids).
- `_compute_kpad` no longer counts self (query not in index).
- Kpad=0 matches WBIA oracle (query not in `database_annot_uuid_list`).

### Descriptor Stacking Order — DONE

- Database constructed in batch file order (not queries-first).
- AIDs now 1:1 with WBIA (hotspotter 0-based = WBIA 1-based - 1).
- Trace query index decoupled from database index (`trace_query_index` param on `identify()`).
- Result: neighbor ID match jumped from 7.2% to 72.98%.

### Chip / Features Row Consolidation — DONE

- All trace stages write 19 rows per file (matching WBIA).
- `trace_chips_and_features(database)` batches all annotations into single `_stage_rows` call.
- Chips schema: `aid`, `chip_fpath`, `chip_size` (WBIA-compatible).

### Chip Extraction — Negative BBox Fix — DONE

- `extract_chip()` passes raw bbox to affine transform; `cv2.warpAffine` with `BORDER_CONSTANT` handles negative coordinates.
- 19/19 annotations produce identical keypoint counts with WBIA.
- All 36,423 descriptors are bit-identical between systems.

### Knorm Config — DONE

- `HotSpotterConfig.knorm` field (`ge=1`), used by `identify()`.
- Knorm=2 tested and working in parity.

### Kpad Logic — DONE

- `_compute_kpad` respects `can_match_samename` guard.
- Dynamic Kpad correctly computes same-name padding.
- For parity configs (`can_match_samename=True`): Kpad=0 → 5-column neighbor arrays.

### Spatial Verification Backend — DONE

- Replaced `cv2.findHomography` with `vtool.spatially_verify_kpts()`.
- Docker preserves vendored `wbia-vtool 4.0.3`.
- SV agreement still 0.4762 — remaining gap in vtool input semantics.

## Remaining Build Work

### 4. Define Public API Levels

Expose a small high-level API for standalone users, and lower-level primitives for `wildlife-id`.

### 5. Fix Scoring Semantics

- WBIA `score_method="nsum"` means fmech/name scoring.
- WBIA `score_method="csum"` means max-csum name scoring.
- Current code uses `*_wbia` suffix for WBIA-style behavior.
- Decide whether public names should match WBIA semantics directly.

### 6. Remaining Config Gaps

- **`can_match_sameimg`**: Missing from `HotSpotterConfig`. WBIA defaults to `False` (filters same-image matches). For parity test set every image has 1 annotation, so it's a no-op. Still worth adding for completeness.
- **`normalizer_rule="name"`**: Hotspotter implementation not equivalent to WBIA — only checks the last normalizer column instead of scanning all Knorm candidates. Both systems default to `"last"` so this doesn't affect parity.

### 7. Spatial Verification Semantics

SV agreement still 0.4762. Need to align:
- match weights;
- dlen/extent inputs;
- refine method and thresholds;
- shortlist ordering;
- post-SV score update.

### 8. FLANN / PyFlann Version Alignment

Neighbor IDs match at 73%. The remaining 27% is from different pyflann/numpy
versions between Docker images. Aligning these would close most of the gap.

### 9. Comparer npy Path Resolution

Comparer can't load hotspotter `.npy` sidecars because metadata paths point
to Docker container paths. Fix to report accurate `neighbor_dist_pearson_r`
and `descriptor_cosine` (currently both show 0.00 but actual values are
r=0.98 and descriptors are bit-identical).

### 10. Automated Parity Gate

Add pytest parity test that imports `compare_wbia_oracles` logic, asserts
ρ ≥ 0.97 in CI once metrics are meaningful enough.

## Key Docs Reference

| Doc | Content |
|---|---|
| `hotspotter-parity-discrepancies.md` | Root causes, measured metrics, Phase 2 queue |
| `phase-1-baseline.md` | Pre-rename test baseline |
| `wbia-oracle-recording.md` | Recording workflow, comparison how-to |
| `hotspotter_permutations.md` | Per-book WBIA HotSpotter param survey |
