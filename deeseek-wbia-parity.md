# Deepseek — WBIA Parity Investigation

> Last updated: 2026-06-27
> Reference batch: 19 annots (COCO zebra)
> COCO batch: 349 annots (30 queries, real individual IDs from GZGC dataset)

## Verdict

Pipeline is correctly extracted. Name grouping was the last structural mismatch — fixed.

## COCO batch results (v6 — sentinel names, per-annot scoring)

| Metric | Value |
|---|---|
| Top-1 annot | **87%** (26/30) |
| Top-1 name | **87%** (26/30) |
| Top-3 annot overlap | **73%** |
| Top-5 annot overlap | **67%** |
| Csum ρ (daid-aware) | **+0.57** (median +0.63) |

Daid Jaccard (0.07) is misleading — HS traces the full 140-202 annot tail
while WBIA traces only 5-34 SV survivors. Top-k metrics are the correct
comparison.

## Root causes (all fixed)

### 1. SV keep-gate vs fm-filter conflation (spatial.py)

WBIA decouples two decisions:
- **Keep gate**: `spatially_verify_kpts` returns None iff affine < 7
  (wbia-vtool/spatial_verification.py:1066). No downstream prune.
- **fm filter**: always filters to homography-refined inliers
  (wildbook-ia pipeline.py:1568).

**HS bug**: had a secondary `if len(sv_inliers) < min_inliers: continue`
measuring refined inliers against the threshold, dropping annots where
`refined < 4` but `affine >= 7` (already passed sver's None gate).
`sv_use_kp_affine_inliers` flag conflated keep-gate and fm-filter.

**Fix**: delete secondary prune, always use `svtup[0]` for fm filter.
Flag deprecated.

### 2. Name grouping (run_fixture.py)

WBIA's sv_on_true pipeline uses `dnid = -daid` sentinels — every
annotation is its own "name."  No individual grouping.  HS was mapping
`individual_ids[0]` to shared name_uuid, causing same-individual
annotations to be pooled in nsumech scoring, producing broadcast ties.

**Fix**: each annotation gets a unique sentinel name `f"-{annot_id}"`,
matching WBIA's per-annot scoring.

### 3. Annotation ordering (pipeline.py)

WBIA sorts by per-annot csum (`cm.sortself()`).  HS was sorting by
per-name nsum, which grouped same-name annots together.

**Fix**: sort by `csum_annot` in final trace.

## Failed experiments

### KNN query inclusion (Kpad=1)
Including the query in the KNN index regressed daid Jaccard from 0.48 →
0.33. WBIA's Kpad=1 is tightly coupled to internal column layout.

### Keypoint-extent dlen_sqrd2
Switching from chip-shape to keypoint-extent dlen_sqrd2 crashed Pearson r
from 0.9997 → 0.68 on the 19-annot reference batch. Reverted; chip-shape
dlen is correct for parity.

## Design decisions

| Decision | Rationale |
|---|---|
| SV keep-gate: sver None only | Matches WBIA (affine ≥ 7) |
| SV fm-filter: always homog refined | Matches WBIA pipeline.py:1568 |
| Sentinel names (`-annot_id`) | Matches WBIA `-daid` per-annot scoring |
| `dlen_sqrd2` from chip shape | Chip dims match WBIA; keypoint extent diverges |
| Query excluded from KNN | Kpad=1 inclusion regresses results |
| `normalizer_rule='last'` | Matches WBIA default |

## Remaining gaps

1. **fm-list Jaccard (~80%)** — per-qfx KNN sets differ for 99% of
   features. Same descriptor sets, different (qfx, dfx) assignments.
2. **RANSAC hypothesis variance** — identical fm input → different
   inlier counts across HS/WBIA processes. `.so`, wrappers MD5-identical.
   Suspect numpy version (1.26.4 vs 1.23.5) or subtle param wiring.
3. **Result list size** — HS traces full tail (140-202 annots); WBIA
   traces SV survivors only (5-34). Top-k metrics unaffected.
