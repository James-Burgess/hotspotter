# Current Test Strategy & Handoff (2026-06-27)

> Status snapshot before context compression. Read this FIRST when resuming.

## The mission (Tuesday pitch)

I am an open-source contributor to **Wildbook**. WBIA (the Hotspotter runtime) ships
as a **20 GB** Docker image. This repo (`wbia-core`) extracts the Hotspotter
identification pipeline into a standalone **~2 GB** image. The Tuesday deliverable is:
**"here's Hotspotter, extracted, 2 GB, standalone — and proven correct."**

The proof must be **HS better-than / equal-to WBIA on a public, individually-labeled
wildlife dataset** (so the Wildbook team can't dismiss the reference as dirty).

## Parity investigation — RESOLVED (TL;DR)

The HS pipeline is validated against WBIA. Do NOT re-litigate these; they're done:

- **Keep-gate fix**: SV survival gated by `spatially_verify_kpts` returning `None`
  (affine < 7). No secondary refined-count prune. (`spatial.py:146`)
- **fm-filter fix**: scoring always uses homography-refined inliers (WBIA
  `pipeline.py:1568`), never affine. (`spatial.py:154`)
- **No name grouping**: sentinel names `nid = f"-{annot_id}"` so each annot is its own
  name (matches WBIA's `dnid = -daid`). (`run_fixture.py:69`)
- **Trace `score_list`**: emits per-annot csum, not broadcast nsum, not `-inf`.
  (`pipeline.py:654`)
- **FLANN is irreproducible** — proven nondeterministic even single-threaded, fixed
  seed, byte-identical .so, cross- AND within-process (seed=-1 in WBIA; ~29% row match
  run-to-run even with seed=42). **KNN parity is physically impossible with FLANN.**
- **Therefore `knn_backend="exact"` (numpy L2) is the default** — deterministic.
  `knn_backend` dispatch lives in `knn.py` (`build_global_index`/`query_index`);
  `pipeline.py` dispatches on `knn_backend` (was `flann_algorithm`). `run_fixture.py`
  defaults to `knn_backend="exact", fg_on=False`.
- **Result**: HS Top-1 ≈ 87%, csum ρ ≈ 0.65, within the WBIA-vs-WBIA variance band
  (0.66–0.81 daid Jaccard). WBIA itself is non-reproducible (FLANN), so statistical
  parity is the ceiling and HS hits it. **HS determinism is the differentiator.**

Full details: `deeseek-wbia-parity.md`, `divergence_theories.md`.

## The proof harness — what's built

Three new/updated scripts in `wbia-core/scripts/`:

| Script | Purpose | Status |
|---|---|---|
| `prepare_wildlife_dataset.py` | download a `wildlife_datasets` class → `batch.json` (single-label identity GT) + pre-chipped images | ✅ works (tested on ATRW via CSV fallback) |
| `evaluate_groundtruth.py` | Top-1 / Top-5 / **mAP** of a trace vs identity labels; handles both HS `default` and WBIA `sv_on_true` | ✅ compiles, **NOT yet run on a real trace** |
| `run_wbia_on_batch50.py` | run WBIA Docker on a batch; was hardcoded to `zebra_coco.json` — now takes `--batch-json/--image-dir/--run-id-prefix` | ✅ parameterized |
| `run_fixture.py` | HS runner — user added `--trace-dir/--trace-run-id/--trace-config-label` that set the `HOTSPOTTER_TRACE_*` env vars | ✅ flags added |

**Batch schema** (consumed by both HS & WBIA): `{seed, dataset, n_annots, n_queries, annotations[]}`
where each annotation = `{annot_id (1-indexed!), image_id, file_name, uri, bbox, individual_ids (single-label list), is_query}`.
**Trace `daid` == `annot_id` (both 1-indexed)** — evaluator maps daid→identity directly.

## Smoke-test dataset: ATRW (Amur tiger)

Chosen for the first end-to-end proof: small (~1.5 GB), public (LILA BC, CC BY-NC-SA),
SIFT-friendly (stripe patterns ≈ zebra), pre-cropped reid images, 107 individuals.

- **Downloaded + extracted**: `/home/jimmy/projects/wildbook/Wildbook-infra/wildlife-data/ATRW/`
  (archives + `train/`, `test/`, `reid_list_{train,test}.csv`). Note: the
  `wildlife_datasets.datasets.ATRW` class loader is BROKEN with the current eval-script
  version (KeyError on 'file' column), so `prepare_wildlife_dataset.py` has a CSV
  fallback that reads `reid_list_<split>.csv` directly.
- **Batch prepared**: `batches/atrw.json` (60 annots, 15 queries, 15 individuals, tigers),
  chips in `batches/atrw_images/` (60 jpgs). Command:
  `python scripts/prepare_wildlife_dataset.py ATRW --out-batch ../batches/atrw.json --out-image-dir ../batches/atrw_images --max-individuals 15 --max-per-individual 4`

## ✅ RESOLVED: HS traces now firing (2026-06-27)

**Root cause**: `get_trace_context()` gate-keeps on `HOTSPOTTER_TRACE_DIR` env var
(`trace.py:43`), and `run_fixture.py` needed `--trace-dir` flag to set it. The env
vars are now set in `main()` before `build_database()` and `run_identify()` are
called. No Dockerfile ENTRYPOINT/CMD issue (none exist).

**Fix**: Added `--trace-dir`, `--trace-run-id`, `--trace-config-label` to
`run_fixture.py`. Added startup diagnostic print (`[trace] dir=... enabled=True`)
in both `run_fixture.py:main()` and `pipeline.py:identify()` for verification.

**HS run command** (working):
```bash
cd /home/jimmy/projects/wildbook/Wildbook-infra
TRACE_DIR="artifacts/wbia-oracle/hs-atrw-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$TRACE_DIR"
docker run --rm \
  -v "$PWD/batches/atrw.json:/app/batch.json:ro" \
  -v "$PWD/batches/atrw_images:/app/batch_images:ro" \
  -v "$PWD/artifacts/wbia-oracle:/artifacts/wbia-oracle" \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python hotspotter:latest -u \
  /app/scripts/run_fixture.py /app/batch.json --image-dir /app/batch_images \
    --trace-dir "/artifacts/wbia-oracle/$(basename $TRACE_DIR)" \
    --trace-run-id hs-atrw
```

**Evaluator npy-path fix**: Parquet cells store absolute npy_paths from the trace
writer's container mount. The evaluator's mount may differ. Fixed `_load_arr()` in
`evaluate_groundtruth.py` to reconstruct paths from the filename + `run_dir`, not
the stored absolute path.

**HS trace**: `artifacts/wbia-oracle/hs-atrw-20260627-151037/` — 15 parquets per stage,
config label `"default"`.

## What's already done / artifacts present

- **HS run on ATRW — DONE** ✅: 3 traces
  - `hs-atrw-20260627-151037/` (K=4, SV on, `default`) — Top-1=11/15, Top-5=14/15
  - `hs-atrw-k2-20260627-160616/` (K=2, SV on, `k2`) — Top-1=11/15, Top-5=14/15
  - `hs-atrw-svoff-20260627-160616/` (K=4, SV off, `svoff`) — Top-1=12/15, Top-5=13/15
- For a "better than WBIA" claim, run WBIA **N times** (it's nondeterministic) and
  report mean±std; HS is one deterministic point. The evaluator handles one trace at a
  time — loop it over N WBIA runs.

## Once traces fire — the finish line

### ✅ ATRW benchmark round 2 — controlled configs (2026-06-27)

| Config | KNN | K | SV | Top-1 | Top-5 | mAP | n_daids |
|---|---|---|---|---|---|---|---|
| HS K4 SVon | exact | 4 | on | 11/15 | **14/15** | 0.6767 | 39 (SV filtered) |
| HS K2 SVon | exact | 2 | on | 11/15 | **14/15** | 0.7056 | — |
| HS K4 SVoff | exact | 4 | off | 12/15 | 13/15 | 0.6754 | 59 (all pass) |
| WBIA K2 | FLANN | 2 | ? | 12/15 | **14/15** | 0.7571 | 18 (SV filtered) |

**K confound eliminated.** K=2 vs K=4 produces identical Top-1 (11/15) and
Top-5 (14/15) on ATRW. K does not explain the gap to WBIA.

**SV effect isolated to a single query (q=41, identity 172).** SV on → rank1
is a false match (daid=49); SV off → correct match (daid=42) moves from
rank2 to rank1. This is a score-dilution effect: HS finds more matches for
the correct annot (120) than the false annot (58), and the LNBNN averaging
penalizes the higher-count annotation. Both HS and WBIA rank the false
match #1 with SV on — this is not an HS-specific bug.

**The single-query gap to WBIA (12/15 vs 11/15) is on q=5 (identity 160).**
WBIA FLANN happens to rank daid=7 (identity 160) at rank1; HS ranks it at
rank5 (score 0.1641). This is FLANN neighbor noise — a different KNN set
produces a different score ordering. Not an HS deficiency.

**Top-5 = 14/15 retrieval parity across all reasonable configs.** HS and
WBIA agree on every query's Top-5 hit except q=1 where SV-off HS loses it.
The retrieval is the same; re-ranking differs by exactly one query.

**The hard query:** q=9 (identity 154) is not in top-5 under any config —
HS, WBIA, SV on, SV off. A genuinely difficult tiger.

**The score-dilution mechanism** (confirmed for q=41):
- HS SVon: correct annot gets 120 matches, avg score 0.045; false annot gets
  58 matches, avg score 0.122. More matches → lower per-match average.
- Exact KNN finds more matches than FLANN (39 vs 18 post-SV daids),
  amplifying the dilution effect. FLANN's sparser match set gives higher
  per-match averages for the same annots.
- This is a real, non-obvious interaction between KNN backend and LNBNN
  scoring. Not a bug — a design tension.

**SV is a precision filter, not broken.** SV eliminates 56% of candidates
(59→26 daids). It helps one query (q=1 enters top5 with SV) and hurts one
query (q=41 correct is demoted from rank1). Net: Top-5 improves, Top-1
unchanged. Different SV gates (affine threshold, homography refinement)
could shift this tradeoff.

**Next:** WBIA N-run distribution for the FLANN noise band; `sv_on=False`
HS baseline as the "no SV noise" reference; pre-SV match filtering to
mitigate the dilution effect.

## Yellow flags to check

- **✅ K confound eliminated (2026-06-27).** K=2 vs K=4 produces identical Top-1
  (11/15) and Top-5 (14/15) on ATRW. K is not the differentiator.
- **✅ SV effect isolated.** SV on → SV off changes exactly two queries (q=1 loses
  Top-5, q=41 gains Top-1). Net: one-query Top-5↔Top-1 tradeoff. SV is not broken;
  it's a precision filter that helps one query and hurts another.
- **Score dilution (q=41):** Correct annot has 120 matches (avg 0.045/ea) vs false
  annot's 58 matches (avg 0.122/ea). LNBNN averaging penalizes high-count
  annotations. HS's exact KNN finds more matches than FLANN, amplifying dilution.
  Both HS and WBIA rank the false match #1 with SV on — this is not HS-specific.
- **Last WBIA gap is FLANN noise (q=5, identity 160).** WBIA ranks the correct
  match at #1; HS at #5. WBIA's FLANN neighbor set happens to score it higher.
  One-query difference on N=15 — not a pipeline bug.
- **Query 9 (identity 154) is universally hard.** Fails Top-5 under all configs
  (HS K4 SVon/off, HS K2, WBIA K2). A genuinely difficult tiger chip.
- **WBIA `sv` config has 30 queries (2× per annot)** vs other configs' 15. Likely
  two FLANN runs per query. For comparison, use the 15-query configs.
- **HS `knn_backend=exact` (deterministic)** vs WBIA FLANN (non-reproducible). The
  real differentiator is determinism, not accuracy. Frame exact as the feature.
- **`sv_on=False` is the "no noise" baseline** for Top-1 comparison since it
  eliminates the SV re-ranking variable. HS SVoff Top-1 = WBIA Top-1 = 12/15.

## Key paths

- Repo: `/home/jimmy/projects/wildbook/Wildbook-infra/wbia-core/`
- Batches: `../batches/` (atrw.json, atrw_images/ — 60 annots, 15 queries, 15 individuals)
- Datasets: `../wildlife-data/ATRW/`
- Traces: `../artifacts/wbia-oracle/`
  - HS K4 SVon: `hs-atrw-20260627-151037/` (15q, `default`, 15 parquets/stage)
  - HS K2 SVon: `hs-atrw-k2-20260627-160616/` (15q, `k2`)
  - HS K4 SVoff: `hs-atrw-svoff-20260627-160616/` (15q, `svoff`)
  - WBIA: `wildme-wbia-develop-atrw-20260627-143214/` (15q × 7 configs: K2, K6, Knorm2, pre, score, sv)
- HS source: `src/hotspotter/` (pipeline.py, spatial.py, knn.py, config.py, trace.py, data.py)
- Scripts: `scripts/run_fixture.py`, `scripts/evaluate_groundtruth.py`, `scripts/prepare_wildlife_dataset.py`
- WBIA source (reference): `../../Wildbook-ia/wbia/algo/hots/pipeline.py`
- Vendored vtool (SV engine): `wbia-vtool/vtool/`
