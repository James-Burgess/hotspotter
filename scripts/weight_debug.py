#!/usr/bin/env python3
"""Compare weight matrices per-cell for K2 query 0."""

import json, numpy as np, pathlib, pandas as pd, sys, os

sys.path.insert(0, "/app")

from hotspotter.scoring import weight_neighbors_lnbnn, baseline_filter, build_matches
from hotspotter.pipeline import (
    _query_neighbors,
    _normalize_distances,
    _build_vote_columns,
)
from hotspotter.config import HotSpotterConfig
from scripts.run_fixture import build_database, load_batch

ORACLE = pathlib.Path("/artifacts/wbia-oracle/wildme-wbia-nightly-20260629-161926")
HS = None
for d in sorted(os.listdir("/tmp"), reverse=True):
    dp = pathlib.Path(f"/tmp/{d}")
    if (
        dp.is_dir()
        and d.startswith("hotspotter-trace-")
        and (dp / "linear" / "nearest_neighbors").exists()
    ):
        HS = dp / "linear"
        break

CFG, QI = "K2", 0

# Load WBIA KNN dists
df_w = pd.read_parquet(ORACLE / "nearest_neighbors" / f"{CFG}_{QI:06d}.parquet")
meta_w = json.loads(str(df_w.iloc[0]["neighbor_dists_array"]))
w_path = pathlib.Path(meta_w["npy_path"])
wbia_dists = np.load(str(next(ORACLE.rglob(w_path.name))), allow_pickle=True)
print(f"WBIA dists: {wbia_dists.shape}")

# Load HS KNN dists
df_h = pd.read_parquet(HS / "nearest_neighbors" / f"{CFG}_{QI:06d}.parquet")
meta_h = json.loads(str(df_h.iloc[0]["neighbor_dists_array"]))
h_path = pathlib.Path(meta_h["npy_path"])
hs_arr = np.load(str(next(HS.rglob(h_path.name))), allow_pickle=True)
if hasattr(hs_arr, "keys"):
    hs_arr = hs_arr[list(hs_arr.keys())[0]]
hs_dists = np.asarray(hs_arr)
print(f"HS dists: {hs_dists.shape}")

# Build HS pipeline
batch = load_batch(pathlib.Path("/app/pipeline/tests/reference_batch.json"))
database, qis, _ = build_database(
    batch, pathlib.Path("/app/pipeline/tests/assets/images")
)
qidx = qis[0]
hs_cfg = HotSpotterConfig(
    knn=2, fg_on=False, kpad_policy="dynamic", knorm=1, knn_backend="linear"
)
k = hs_cfg.knn
kpad = 0
k_total = k + kpad + hs_cfg.knorm
qf = database[qidx].features
knn = _query_neighbors(database, qidx, qf, hs_cfg, k_total)
dists, labels = _normalize_distances(None, qidx, knn, hs_cfg)
votes = _build_vote_columns(None, qidx, dists, labels, knn, k, kpad)

# Verify dists identical
dists_eq = np.allclose(wbia_dists, hs_dists)
print(
    f"Dists identical: {dists_eq} max delta: {np.abs(wbia_dists - hs_dists).max():.2e}"
)

# Compute weights both ways
normk = np.full(1538, hs_cfg.knorm - 1, dtype=np.int32)

v_w = wbia_dists[:, : k + kpad].astype(np.float64)
n_w = wbia_dists[:, k + kpad :].astype(np.float64)
w_w = weight_neighbors_lnbnn(v_w, n_w, normk=normk)

v_h = hs_dists[:, : k + kpad].astype(np.float64)
n_h = hs_dists[:, k + kpad :].astype(np.float64)
w_h = weight_neighbors_lnbnn(v_h, n_h, normk=normk)

delta = np.abs(w_w - w_h)
print(
    f"\nWeight delta: max={delta.max():.2e}, nonzero={(delta > 0).sum()} / {delta.size}"
)

# Build matches from both weight sets
invalid, _, _ = baseline_filter(votes.voting_annot, database, qidx)
m_w = build_matches(
    w_w, votes.voting_annot, votes.voting_feat, invalid, database, k, kpad
)
m_h = build_matches(
    w_h, votes.voting_annot, votes.voting_feat, invalid, database, k, kpad
)
print(f"\nMatches: WBIA-weights={len(m_w)}, HS-weights={len(m_h)}")

w_set = {(m.qfx, m.dfx) for m in m_w}
h_set = {(m.qfx, m.dfx) for m in m_h}
print(f"Overlap: {len(w_set & h_set)}")

# Zero crossing analysis
wb_pos = w_w > 0
wb_zero = w_w == 0
h_pos = w_h > 0
h_zero = w_h == 0

wb_pos_h_zero = (wb_pos & h_zero).sum()
h_pos_wb_zero = (h_pos & wb_zero).sum()
print(
    f"\nZero crossings: WBIA>0 & HS=0: {wb_pos_h_zero}, HS>0 & WBIA=0: {h_pos_wb_zero}"
)

if wb_pos_h_zero + h_pos_wb_zero > 0:
    print("\nFirst 10 zero crossings:")
    crossed = (
        np.where(wb_pos & h_zero) if wb_pos_h_zero > 0 else np.where(h_pos & wb_zero)
    )
    for i in range(min(10, len(crossed[0]))):
        r, c = crossed[0][i], crossed[1][i]
        print(
            f"  [{r},{c}]: W={w_w[r,c]:.6e} H={w_h[r,c]:.6e} vdist_w={v_w[r,c]:.6f} ndist_w={n_w[r,0]:.6f}"
        )
