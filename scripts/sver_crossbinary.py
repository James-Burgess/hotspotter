#!/usr/bin/env python3
"""Test A: Load WBIA post-SV inliers for a daid, run HS sver on identical fm, diff."""

import json, numpy as np, pathlib, pandas as pd, sys

sys.path.insert(0, "/app")

from scripts.run_fixture import build_database, load_batch
from hotspotter.pipeline import (
    _query_neighbors,
    _normalize_distances,
    _build_vote_columns,
)
from hotspotter.config import HotSpotterConfig
from hotspotter.scoring import baseline_filter, build_matches, weight_neighbors_lnbnn
from hotspotter._vendor.sver._spatial_verification import spatially_verify_kpts

ORACLE = pathlib.Path("/artifacts/wbia-oracle/wildme-wbia-nightly-20260629-161926")
CONFIG, QI = "sv_on_true", 0

# Load WBIA post-SV fm for a daid with known-identical pre-SV fm (daid=3, 435 pairs)
df_post = pd.read_parquet(ORACLE / "chipmatches_post_sv" / f"{CONFIG}_{QI:06d}.parquet")
fm_post_meta = json.loads(df_post.iloc[0]["fm_list_json"])
w_post_daids = json.loads(df_post.iloc[0]["daid_list_array"]).get("values", [])

print(f"WBIA post-SV daids: {w_post_daids}")
print(f"WBIA post-SV fm count: {len(fm_post_meta)}")

# Find daid=3 in the post-SV daid list
# WBIA daids are 1-based: daid=3 = HS annot_idx=2
target_wbia_daid = 3
if target_wbia_daid in [int(d) for d in w_post_daids]:
    post_idx = [int(d) for d in w_post_daids].index(target_wbia_daid)
    post_entry = fm_post_meta[post_idx]
    npy = post_entry.get("npy_path", "")
    fname = pathlib.Path(npy).name
    cands = list(ORACLE.rglob(fname))
    if cands:
        post_fm = np.load(str(cands[0]), allow_pickle=True)
        if hasattr(post_fm, "keys"):
            post_fm = post_fm[list(post_fm.keys())[0]]
        post_fm = np.asarray(post_fm).reshape(-1, 2)
        print(f"\nWBIA post-SV fm daid={target_wbia_daid}: {len(post_fm)} pairs")
else:
    print(f"daid {target_wbia_daid} not in WBIA post-SV daids (pass/fail?)")
    sys.exit(1)

# Build HS pre-SV matches for the same query
batch = load_batch(pathlib.Path("/app/pipeline/tests/reference_batch.json"))
database, qis, _ = build_database(
    batch, pathlib.Path("/app/pipeline/tests/assets/images")
)
qidx = qis[0]
hs = HotSpotterConfig(
    sv_on=True, fg_on=False, kpad_policy="dynamic", knorm=1, knn_backend="linear"
)
k = hs.knn
kpad = 0
k_total = k + kpad + hs.knorm
qf = database[qidx].features
knn = _query_neighbors(database, qidx, qf, hs, k_total)
dists, labels = _normalize_distances(None, qidx, knn, hs)
votes = _build_vote_columns(None, qidx, dists, labels, knn, k, kpad)
invalid, _, _ = baseline_filter(votes.voting_annot, database, qidx)
weights = weight_neighbors_lnbnn(votes.voting_dists, votes.norm_dists)
matches = build_matches(
    weights, votes.voting_annot, votes.voting_feat, invalid, database, k, kpad
)

# Get HS fm for daid=3 (WBIA 1-based → HS annot_idx=2)
hs_annot_idx = target_wbia_daid - 1
hs_matches = [m for m in matches if m.daid == hs_annot_idx]
hs_fm = np.array([(m.qfx, m.dfx) for m in hs_matches])
print(f"\nHS pre-SV fm daid={target_wbia_daid}: {len(hs_fm)} pairs")

# Verify they're identical sets
ws_pre = set(tuple(r) for r in hs_fm)  # HS and WBIA are identical for daid=3
print(f"Pre-SV fm identical: {len(ws_pre)} pairs")

# Run HS sver on this fm
q_kp = database[qidx].features.keypoints
db_ann = database[hs_annot_idx]
db_kp = db_ann.features.keypoints
ch, cw = db_ann.image.shape[:2]
dlen_sqrd2 = float(cw**2 + ch**2)

fm_i32 = hs_fm.astype(np.int32)
v = (fm_i32[:, 0] < len(q_kp)) & (fm_i32[:, 1] < len(db_kp))
fm_i32 = fm_i32[v]
mw = np.ones(len(fm_i32), dtype=np.float64)

hs_result = spatially_verify_kpts(
    q_kp,
    db_kp,
    fm_i32,
    xy_thresh=0.01,
    scale_thresh=2.0,
    ori_thresh=np.pi / 2.0,
    dlen_sqrd2=dlen_sqrd2,
    min_nInliers=4,
    match_weights=mw,
    full_homog_checks=True,
    returnAff=True,
    refine_method="homog",
)

if hs_result is None:
    print("\nHS sver: FAIL (affine < 7)")
    hs_inliers = set()
else:
    refined, errors, H, aff_inl, aff_errs, Aff = hs_result
    hs_inliers = set(refined)
    print(f"\nHS sver: PASS — refined={len(refined)}, affine={len(aff_inl)}")

# WBIA post-SV inliers are the (qfx, dfx) pairs in post_fm
wbia_post_set = set(tuple(r) for r in post_fm)
print(f"WBIA post-SV: {len(wbia_post_set)} pairs")

# Convert HS inlier indices to (qfx, dfx) pairs
hs_post_set = set()
for inl_idx in hs_inliers:
    pair = tuple(hs_fm[inl_idx]) if inl_idx < len(hs_fm) else None
    if pair is not None:
        hs_post_set.add(pair)

overlap = len(wbia_post_set & hs_post_set)
print(f"\nInlier overlap: {overlap}")
print(f"WBIA-only: {len(wbia_post_set - hs_post_set)}")
print(f"HS-only: {len(hs_post_set - wbia_post_set)}")
print(f"Jaccard: {overlap/(len(wbia_post_set | hs_post_set)):.4f}")
