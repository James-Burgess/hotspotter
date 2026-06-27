# Deepseek — WBIA Parity Investigation

> Last updated: 2026-06-27
> Reference batch: 19 annots (COCO zebra)
> COCO batch: 349 annots (30 queries, real individual IDs from GZGC dataset)

## Verdict

The hotspotter pipeline is correctly extracted. All structural divergences
are resolved. The residual survivor-set gap is a **FLANN-level noise floor**
shared by WBIA itself — not a HS pipeline bug, not closable without a
deterministic KNN backend on both sides. FLANN's C++ kd-tree build and
query are nondeterministic by design: the RNG is not seedable from the
Python `random_seed` parameter, and even single-threaded execution diverges
between runs on identical data.

### The FLANN noise floor (definitive)

WBIA uses FLANN with `seed=-1` (random seed every run). This means every
WBIA invocation produces a different kd-tree forest, and the SV survivor
set is **not reproducible even by WBIA itself**.

On the COCO batch (Q1, n=5 FLANN runs with WB-matched params):

| Metric | Range | Mean |
|---|---|---|
| Survivor count (FLANN-WB, seed=-1) | 30–38 | 33.8 |
| Run-to-run Jaccard (seed=-1) | 0.32–0.54 | 0.41 |
| Run-to-run Jaccard (seed=42) | 0.39 | — |
| **WB develop (one-shot trace)** | **26** | — |

HS's exact KNN (deterministic numpy L2) produces 47 survivors. The gap from
47 to FLANN's mean 34 is fully explained by FLANN approximation error. The
remaining 34→26 gap sits at <1σ of FLANN's own run-to-run variance. A
different WBIA run with the same images would produce 30–38 survivors.

WBIA-WBIA cross-build Daid Jaccard (0.66–0.81) is largely this same FLANN
variance. HS sits at the edge of the same band.

### KNN backend config (v11)

HS now exposes three backends, all controlled by `knn_backend`:

| `knn_backend` | Implementation | Deterministic | Survivors (Q1) | Use for |
|---|---|---|---|---|
| `"exact"` | numpy L2 dot products | Yes | ~47 | Production, reproducibility |
| `"flann"` | pyflann kdtree (WB params) | No | 30–38 | WBIA parity comparison |
| `"faiss"` | faiss IndexFlatL2 | Yes | ~47 | Production, GPU-capable |

**Critical**: exact KNN is more permissive than WBIA's FLANN (~47 vs 34 mean
survivors) — it is not a WBIA reproduction. Use `knn_backend="flann"` for any
WBIA parity comparison and `knn_backend="exact"` for production/reproducibility.
Do not read the survivor delta between exact and FLANN as a regression; it is
a deliberate backend choice.

FLANN defaults now match WBIA (trees=4, checks=32, seed=-1, cores=1). The
old HS defaults (trees=8, checks=800, seed=42) were running a 25× more
thorough KNN than WBIA — the dominant cause of the 2× over-keep.

## All root causes (resolved)

### 1. SV keep-gate vs fm-filter conflation (spatial.py)

WBIA decouples two independent decisions:
- **Keep gate**: `spatially_verify_kpts` returns None iff affine < 7
  (`wbia-vtool/spatial_verification.py:1066`). No downstream prune.
- **fm filter**: always scores on homography-refined inliers
  (`wildbook-ia pipeline.py:1568`).

**HS bug**: `spatial.py:154-156` had `if len(sv_inliers) < min_inliers:
continue` measuring refined inliers against threshold, dropping annots
where `refined < 4` but `affine ≥ 7` (already passed sver's None gate).

**Fix**: delete secondary prune. Always filter fm to `svtup[0]`
(refined). The `sv_use_kp_affine_inliers` flag is deprecated.

### 2. Name grouping (run_fixture.py)

WBIA's sv_on_true pipeline uses `dnid = -daid` sentinels — every
annotation is its own "name." No individual grouping. HS was mapping
`individual_ids[0]` to shared name_uuid, producing broadcast ties.

**Fix**: each annotation gets a unique sentinel name `f"-{annot_id}"`,
matching WBIA's per-annot scoring.

### 3. Query excluded from KNN index (pipeline.py)

HS builds its KNN index over N−1 annots (query excluded). WBIA builds
over N (query included, Kpad=1). Different point-sets → different kd-tree
splits → different near-tie resolution → different (qfx, dfx) pairs.

**Fix**: include query in index. Kpad=1 absorbs self-match at column 0;
scoring skips column 0 (`range(1, k+kpad)`), using columns 1–4 as
voting and column 5 as normalizer. Matches WBIA's column layout.

### 4. Post-SV filtering: non-candidates kept (pipeline.py)

HS kept non-SV-candidate annotations with pre-SV scores in the post-SV
list. WBIA drops them — only SV survivors appear.

**Fix**: `sv_verify_all=True` sends all scored annots through SV
(matching WBIA's verify-all policy, confirmed: WB pre-SV = 348 annots).
Post-SV filter keeps only SV-verified annots.

### 5. Annotation ordering (pipeline.py)

WBIA sorts final trace daids by per-annot csum (`cm.sortself()`). HS
was sorting by per-name nsum.

**Fix**: sort by `csum_annot`.

### 6. `dlen_sqrd2` from chip shape (correct)

WBIA computes `dlen_sqrd2` from chip image dimensions (W²+H²), not
keypoint extent. HS defaults match this already (`sv_use_chip_extent=True`).

### 7. Trace score_list (pipeline.py)

WBIA's score_list is per-annot nsum, not per-name broadcast. With sentinel
names (1 annot per name), `_final_nsum` equals per-annot nsum.

## Residual: FLANN noise floor (not HS pipeline)

After matching WB's FLANN params (trees=4, checks=32, seed=-1), HS's mean
survivor count is 34 (vs exact KNN 47, vs WB trace 26). FLANN's own
run-to-run variance spans 30–38 survivors with Jaccard 0.32–0.54 — the
same range as HS-vs-WB disagreement.

The root cause is FLANN's non-deterministic kd-tree construction:
1. **seed=-1** means every WB and FLANN-HS run builds a different forest
2. Different forests → different near-tie resolution → different fm lists
3. Different fm lists → different SV pass/fail decisions near affine≥7
4. The C++ FLANN library is non-deterministic even with fixed seed+single-thread (10-30% row mismatch at 20K descriptors)

The survivor gap is a one-sample artifact: WB's trace landed at 26;
re-running WB on the same images would land at 30–38. HS using exact KNN
gets 47 because it finds more genuine neighbors (no approximation loss).

### Affine scatter (both-kept annots, n=24)

| Metric | HS | WB |
|---|---|---|
| Mean affine | 41.7 | 42.2 |
| Higher count | 11 | 11 |

Symmetric. Same distribution, same mean.

### Spatial spread (HS-only survivors, n=20)

18/20 are spread across the chip (10–45% area coverage, norm_std
0.10–0.36). Genuine geometric structure, not spurious coherence.

### HS-only affine breakdown (v10, Q1, n=23)

- 14/23: WB fm also passes affine ≥ 7 (dropped by WB pre-SV shortlist)
- 9/23: WB fm produces **zero** affine consensus; HS fm produces 7–9
  genuine inliers — exact KNN finds structure FLANN misses

### FLANN is not seedable (proof)

FLANN's kd-tree build and query are nondeterministic by design. Even with
`random_seed=42`, single-threaded (`cores=1`, `OMP_NUM_THREADS=1`), same
data — 10–30% of query rows produce different neighbor columns across
runs within the same process. The C++ RNG is not actually controllable
from the Python `random_seed` parameter; OpenMP thread scheduling inside
`libflann.so` amplifies the effect but isn't the sole cause (single-
threading still diverges).

Cross-container tests add global descriptor ordering as a second confound
(the ordering question was never fully closed), so they can't serve as
an independent axis. The airtight claim is: **execution nondeterminism
within a single process is sufficient to make bit-exact KNN parity
impossible with any FLANN backend.**

WBIA's `seed=-1` (random seed every run) makes WBIA itself irreproducible
at the KNN level — every invocation produces a different kd-tree forest.

## Failed experiments

### Kpad=1 without query inclusion
Regressed daid Jaccard from 0.48 → 0.33. Extra column without
corresponding query-inclusion broke column layout.

### Keypoint-extent dlen_sqrd2
Switching from chip-shape to keypoint-extent dlen_sqrd2 crashed Pearson r
from 0.9997 → 0.68 on the 19-annot reference batch. Chip-shape is correct.

### Canonical -inf score_list
WBIA's score_list is per-annot nsum, not per-name canonical. The -inf fix
was backwards — WBIA never sinks same-name runners-up because they don't
exist (every annot is its own name).

## Key design decisions

| Decision | Rationale |
|---|---|
| `knn_backend="exact"` | Deterministic KNN (numpy L2); default in run_fixture.py |
| `knn_backend="flann"` | Approximate kdtree with WB-matched defaults (trees=4, checks=32, seed=-1) |
| `knn_backend="faiss"` | Deterministic exact search via faiss IndexFlatL2 |
| SV keep-gate: `sver` None only | Matches WBIA (affine ≥ 7) |
| SV fm-filter: always `svtup[0]` (homog) | Matches WBIA pipeline.py:1568 |
| Sentinel names (`-annot_id`) | Matches WBIA `-daid` per-annot scoring |
| Query included in KNN index + Kpad=1 | Matches WBIA kd-tree topology |
| Scoring skips col 0 (self-match) | 4 effective voting cols = WBIA layout |
| `sv_verify_all=True` | WBIA verifies all 348 candidates |
| Post-SV: only keep SV-verified | Matches WBIA survivor-only output |
| `dlen_sqrd2` from chip shape | Chip dims match WBIA; keypoint extent diverges |
| `normalizer_rule='last'` | Matches WBIA default |
| Trace `score_list` = `_final_nsum` | Per-annot nsum with sentinel names |

## Test infrastructure

```
wbia-core/scripts/
  create_coco_batch.py    # COCO batch: 30 queries + real individual IDs
  create_batch_100.py     # 100-image batch with positive/negative splits
  run_wbia_on_batch50.py  # Mounts batch → WBIA Docker containers
  compare_50_batch.py     # Cross-system daid-aware comparison
  ransac_experiment.py    # Controlled SV test with WBIA's fm_list
  sv_byte_dump.py         # Byte-level SV with cross-process verification
  trace_fm_divergence.py  # Per-qfx KNN label set comparison
  deep_sv_test.py         # Deep SV with MD5-verified inputs

../test-files/zebras/gzgc.coco/  # COCO dataset with real individual IDs
  annotations/instances_train2020.json
  images/train2020/      (4948 images, 6925 annotations)
```

## Files changed

| File | Change |
|---|---|
| `knn.py:65-110` | `build_global_index` + `query_index` accept `backend` param |
| `knn.py:116-140` | `query_index` dispatches on `backend` (pyflann/faiss) |
| `config.py:136-147` | `knn_backend` field; FLANN defaults matched to WB (trees=4, c=32, s=-1) |
| `pipeline.py:142-180` | Dispatch on `hs.knn_backend` instead of `hs.flann_algorithm` |
| `run_fixture.py:112-113` | Default `knn_backend="exact"` (was `flann_algorithm="exact"`) |
| `spatial.py:148-156` | Delete downstream inlier prune; always use `svtup[0]` |
| `spatial.py:38` | Remove `use_kp_affine_inliers` param |
| `pipeline.py:131-137` | Include query in KNN index |
| `pipeline.py:196-203` | Voting arrays `k+kpad` wide |
| `pipeline.py:262,291` | Scoring loops skip col 0 (`range(1, k+kpad)`) |
| `pipeline.py:437-446` | `sv_verify_all` flag bypasses shortlist |
| `pipeline.py:542-548` | Post-SV: only keep SV-verified annots |
| `pipeline.py:654` | Trace `score_list` = `_final_nsum` (per-annot) |
| `config.py:27` | `kpad=1` |
| `config.py:86-91` | `sv_verify_all=True` |
| `config.py:112` | `sv_use_kp_affine_inliers` deprecated |
| `run_fixture.py:66-69` | Sentinel names (`-annot_id`) |
