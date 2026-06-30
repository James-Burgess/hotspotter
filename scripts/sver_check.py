#!/usr/bin/env python3
"""Compare pre-SV fm lists per daid between WBIA and HS for sv_on_true query 0."""

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

df_w = pd.read_parquet(ORACLE / "chipmatches_pre_sv" / f"{CONFIG}_{QI:06d}.parquet")
fm_w_meta = json.loads(df_w.iloc[0]["fm_list_json"])
w_daids = json.loads(df_w.iloc[0]["daid_list_array"]).get("values", [])


def load_fms(run, meta):
    arrays = []
    for entry in meta:
        npy = entry.get("npy_path", "")
        fname = pathlib.Path(npy).name if npy else None
        if fname:
            cands = list(run.rglob(fname))
            if cands:
                arr = np.load(str(cands[0]), allow_pickle=True)
                if hasattr(arr, "keys"):
                    arr = arr[list(arr.keys())[0]]
                arr = np.asarray(arr).reshape(-1, 2)
                arrays.append(arr)
    return arrays


w_fms = load_fms(ORACLE, fm_w_meta)

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

# Build per-daid maps
w_by_daid = {int(d): fm for d, fm in zip(w_daids, w_fms)}
h_by_daid = {}
for m in matches:
    wbia_daid = m.daid + 1  # HS 0-based → WBIA 1-based
    h_by_daid.setdefault(wbia_daid, []).append(m)

h_fm_by_daid = {}
for d, ms in h_by_daid.items():
    h_fm_by_daid[d] = np.array([(m.qfx, m.dfx) for m in ms])

all_daids = sorted(set(w_by_daid.keys()) | set(h_fm_by_daid.keys()))
print(f"WBIA: {len(w_by_daid)} daids, HS: {len(h_fm_by_daid)} daids")

q_kp = database[qidx].features.keypoints

for d in all_daids:
    wf = w_by_daid.get(d)
    hf = h_fm_by_daid.get(d)
    if wf is None:
        print(f"  daid={d:>2}: HS only ({len(hf)} pairs)")
        continue
    if hf is None:
        print(f"  daid={d:>2}: WBIA only ({len(wf)} pairs)")
        continue

    ws = set(tuple(r) for r in wf)
    hs_set = set(tuple(r) for r in hf)
    ov = len(ws & hs_set)
    wo = len(ws - hs_set)
    ho = len(hs_set - ws)

    # Run sver on both if they differ
    if wo > 0 or ho > 0:
        print(f"  daid={d:>2}: W={len(wf)} H={len(hf)} ov={ov} wo={wo} ho={ho}")

        # Run sver on both fms
        db_idx = d - 1  # WBIA 1-based → HS 0-based
        if db_idx >= len(database):
            continue
        db_ann = database[db_idx]
        db_kp = db_ann.features.keypoints
        ch, cw = db_ann.image.shape[:2]
        dlen_sqrd2 = float(cw**2 + ch**2)

        def run_sver(fm, label):
            fm_i32 = fm.astype(np.int32)
            v = (fm_i32[:, 0] < len(q_kp)) & (fm_i32[:, 1] < len(db_kp))
            fm_i32 = fm_i32[v]
            mw = np.ones(len(fm_i32), dtype=np.float64)
            try:
                return spatially_verify_kpts(
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
            except Exception as e:
                print(f"      sver error: {e}")
                return None

        wr = run_sver(wf, "WBIA-fm")
        hr = run_sver(hf, "HS-fm")
        w_pass = wr is not None
        h_pass = hr is not None
        w_inl = len(wr[0]) if wr else 0
        h_inl = len(hr[0]) if hr else 0
        print(f"      WBIA sver: {'PASS' if w_pass else 'FAIL'} (refined={w_inl})")
        print(f"      HS sver: {'PASS' if h_pass else 'FAIL'} (refined={h_inl})")

        # if both pass, check if inlier sets match
        if wr and hr:
            w_inliers = set(wr[0])
            h_inliers = set(hr[0])
            inl_ov = len(w_inliers & h_inliers)
            print(
                f"      Inlier overlap: {inl_ov}/{max(len(w_inliers), len(h_inliers))}"
            )

        # If WBIA-only pairs, test them specifically
        if wo > 0:
            only_w = sorted(ws - hs_set)
            print(f"      WBIA-only pairs sample: {only_w[:3]}")
        if ho > 0:
            only_h = sorted(hs_set - ws)
            print(f"      HS-only pairs sample: {only_h[:3]}")
    else:
        print(f"  daid={d:>2}: W={len(wf)} H={len(hf)} ov={ov} (match)")
