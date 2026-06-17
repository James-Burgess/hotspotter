# E2E Test Strategy — `wbia-core`

## Goal

Verify that `wbia-core.identify()` produces **identical rankings** to the
original WBIA pipeline given the same inputs.  "Identical" means:

1. The same set of scored annotations (same annot_uuids) is returned for a
   given query.
2. The ordering matches exactly (by descending score).
3. Per-match `score`, `num_matches`, and `sv_inliers` are within a tight
   tolerance (**1 % relative** or **1e-6 absolute**, whichever is larger).
4. The **determinism contract** holds: same config + same image → same
   features → same match results (bit-exact on identical hardware).

## Test matrix

| Dimension | Values | Priority |
|---|---|---|
| Species | giraffe, zebra, whale_shark, sea_turtle, polar_bear | P0 |
| Image quality | high (studio), medium (field), low (blurry/distant) | P0 |
| Subject pose | left-flank, right-flank, head-on, overhead | P1 |
| Dataset size | 10 images, 1 000 images, 100 000 images | P0 |
| Config variants | `sv_on=True/False`, `score_method=nsum/csum`, `knn=3/7/20`, `lnbnn_ratio=1.0/0.5` | P0 |
| Edge cases | empty features, single-feature match, all-self-name, duplicate images | P1 |

## Approach

### Phase 1: replay against recorded WBIA output (offline)

```
wbia-core/                     wbia-core/tests/testdata/
  identify(query, db, cfg)  →  expected_top_10.csv
                                expected_correspondences.npz
```

1. Record reference output from the real WBIA pipeline for a fixed seed.
2. Store as deterministic test fixtures (CSV for scores, NPZ for
   correspondences).
3. The CI replay test compares `wbia-core` output against these fixtures
   at the same species + image + config.

**Pros:** fast (< 100 ms per test), deterministic, no WBIA dependency at
test time.  
**Cons:** only covers the scenarios we thought to record.

### Phase 2: shadow-mode comparison against live WBIA (deployed)

```
wildlife-id  ──identify()──▶  shadow_comparator  ──▶  WBIA (ref)
                                  │
                                  └──csv──▶  diff report
```

1. Deploy `wildlife-id` with `WBIA_SHADOW_URL` configured.
2. Every identification request is duplicated: `wbia-core` ranks + WBIA
   ranks.
3. A shadow-comparator sidecar computes agreement metrics:
   - **Recall@N**: fraction of WBIA's top-N that appear in `wbia-core`'s top-N.
   - **Score delta**: mean absolute difference in scores for common matches.
   - **Rank correlation**: Spearman's ρ between the two ranked lists.
4. Alarms fire if agreement drops below a configurable threshold (default:
   95 % Recall@1, 90 % Recall@5).

**Pros:** covers all production traffic, catches regressions in the wild.  
**Pros:** accumulates a growing corpus of validated results.  
**Cons:** requires WBIA to still be running (shadow mode, Phases 2–5 of the
migration plan).

### Phase 3: bit-exact reproducibility validation (CI)

1. Run `identify()` twice on the same inputs.
2. Assert that the returned scores, num_matches, and correspondences are
   bit-exact identical.
3. Run on CPU-only and confirm determinism _within_ the same platform.
4. Document known deviations (e.g., faiss GPU may differ from CPU by
   float-rounding on L2 distances).

## Test data strategy

| Source | Size | Purpose |
|---|---|---|
| Synthetically generated (random descriptors) | Tiny (5 annots, 20 feats each) | Unit & integration (already done) |
| Fixtures from real WBIA (NPZ files) | Medium (5 species × 3 quality levels) | Replay tests (Phase 1) |
| Production shadow-mode logs | Large (100 000+ queries) | Agreement monitoring (Phase 2) |

Synthetic data lives in `tests/testdata/` committed to the repo.  The
recorded WBIA fixtures also live there.  Both are checked into git
(generally < 1 MB each).

## What would be needed to run Phase 1 today

### 1. Record fixture generator (one-time script)

```
scripts/record_wbia_fixtures.py
```

A Python script that:
1. Connects to a running WBIA instance.
2. For each (species, quality, config) cell in the matrix:
   - Loads a known set of images.
   - Calls WBIA's identification endpoint.
   - Saves query + database + config + WBIA results into an NPZ file.
3. Outputs: `tests/testdata/{species}_{quality}_{cfg_hash}.npz`

### 2. Replay test harness

A parametrized pytest test that:
1. Loads a fixture NPZ.
2. Builds `AnnotatedImage` list from the fixture's image data.
3. Calls `wbia_core.pipeline.identify()`.
4. Compares results against the WBIA reference in the fixture.
5. Tolerance: `rtol=0.01, atol=1e-6`.

### 3. CI environment with testdata

The CI runner needs:
- The `tests/testdata/` directory (committed).
- `faiss-cpu`, `opencv-python-headless`, `numpy`, `pydantic` (already in
  `pyproject.toml`).
- No WBIA or pyhesaff at test time (fixtures include pre-extracted
  features).

## Gap analysis: what is NOT testable today

| Gap | Why | When it closes |
|---|---|---|
| Feature extraction correctness | `pyhesaff` is a different codebase from WBIA's fork; parameters may diverge | After `source-extraction` ADR doc is written and pyhesaff fork is pinned |
| LNBNN normalizer agreement with WBIA | Need shadow-mode comparison against real WBIA to confirm | Phase 2 deployment |
| Spatial verification identical to WBIA | Our RANSAC uses `cv2.findHomography` with the same params, but WBIA may use different tuning (min_inliers, threshold, iteration count) | After Phase 1 fixtures are recorded and compared |
| Per-species match-rate deltas | Synthetic data is uniform; real data has species-specific feature distributions | Phase 2 production shadow mode |
| End-to-end latency vs WBIA | Synthetic test data is < 100 descriptors per annot; real images have 500–3000 | Phase 2 |

## Test classification

```
pytest -v                          # unit + integration (41 tests, ~0.4 s)
pytest --e2e                       # Phase 1 replay tests (requires fixtures)
pytest --e2e --slow                # Phase 1 + large-database stress tests
./scripts/shadow_compare.sh        # Phase 2 (requires WBIA_URL + WILDLIFE_ID_URL)
```

## Acceptance criteria

Before `wbia-core` can replace WBIA as the default pipeline:

1. [ ] Phase 1 replay tests pass for all **P0** matrix cells at `rtol=0.01`.
2. [ ] Phase 2 shadow-mode agreement: **Recall@1 ≥ 95 %** over a 7-day
      production window.
3. [ ] Determinism test: 10 consecutive runs produce identical results.
4. [ ] Performance: `identify()` is **not slower** than WBIA's equivalent
      pipeline on the same hardware (within 20 %).
5. [ ] Edge cases: empty database, query with no features, missing
      annotation, duplicate names — all handled without crashing.
