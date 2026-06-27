#!/usr/bin/env python3
"""Trace fm_list divergence for a single daid between HS and WBIA."""

import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HS = Path(sys.argv[1])
WB = Path(sys.argv[2])
QI = int(sys.argv[3])
TEST_DAID = int(sys.argv[4])


def _npy(val, run):
    if isinstance(val, np.ndarray):
        return val
    p = json.loads(val) if isinstance(val, str) else val
    if isinstance(p, dict):
        if "npy_path" in p:
            full = Path(p["npy_path"])
            if not full.is_absolute():
                full = run / "final_scores" / p["npy_path"]
            return np.load(str(full))
        if "values" in p:
            return np.array(p["values"])
    return np.array(p)


def get_fm(daid_list, fm_raw, target, run):
    if isinstance(fm_raw, str):
        fm_raw = json.loads(fm_raw)
    for d, item in zip(daid_list, fm_raw):
        if int(d) == target and isinstance(item, dict) and "npy_path" in item:
            full = Path(item["npy_path"])
            if not full.is_absolute():
                full = run / "final_scores" / item["npy_path"]
            return np.load(str(full))
    return None


hs_nn = pd.read_parquet(HS / "nearest_neighbors" / f"default_{QI:06d}.parquet").iloc[0]
wb_nn = pd.read_parquet(WB / "nearest_neighbors" / f"sv_on_true_{QI:06d}.parquet").iloc[
    0
]
hs_labels = _npy(hs_nn["neighbor_idxs_array"], HS)
hs_dists = _npy(hs_nn["neighbor_dists_array"], HS)
wb_labels = _npy(wb_nn["neighbor_idxs_array"], WB)
wb_dists = _npy(wb_nn["neighbor_dists_array"], WB)
print(
    f"Q{QI} daid={TEST_DAID}  KNN: HS={hs_labels.shape} WB={wb_labels.shape}  identical={np.array_equal(hs_labels, wb_labels)} dist_identical={np.allclose(hs_dists, wb_dists, atol=1e-5)}"
)

hs_bf = pd.read_parquet(
    HS / "baseline_neighbor_filter" / f"default_{QI:06d}.parquet"
).iloc[0]
wb_bf = pd.read_parquet(
    WB / "baseline_neighbor_filter" / f"sv_on_true_{QI:06d}.parquet"
).iloc[0]
hs_valid = _npy(hs_bf["valid_array"], HS)
wb_valid = _npy(wb_bf["valid_array"], WB)
vf = np.array_equal(hs_valid, wb_valid)
print(
    f"Baseline valid: HS={hs_valid.shape} sum={hs_valid.sum()} WB={wb_valid.shape} sum={wb_valid.sum()} match={vf}"
)

hs_pre = pd.read_parquet(HS / "chipmatches_pre_sv" / f"default_{QI:06d}.parquet").iloc[
    0
]
wb_pre = pd.read_parquet(
    WB / "chipmatches_pre_sv" / f"sv_on_true_{QI:06d}.parquet"
).iloc[0]
hs_daids = _npy(hs_pre["daid_list_array"], HS).astype(int)
wb_daids = _npy(wb_pre["daid_list_array"], WB).astype(int)
hs_fm = get_fm(hs_daids, hs_pre["fm_list_json"], TEST_DAID, HS)
wb_fm = get_fm(wb_daids, wb_pre["fm_list_json"], TEST_DAID, WB)

hs_set = {tuple(r) for r in hs_fm}
wb_set = {tuple(r) for r in wb_fm}
hsonly = sorted(hs_set - wb_set)
wbonly = sorted(wb_set - hs_set)
common = hs_set & wb_set
print(
    f"\nfm: HS={len(hs_fm)} WB={len(wb_fm)} overlap={len(common)} HS-only={len(hsonly)} WB-only={len(wbonly)}"
)

nc = (
    min(hs_labels.shape[1], hs_valid.shape[1])
    if hs_valid.shape[1] < hs_labels.shape[1]
    else hs_labels.shape[1]
)
nw = (
    min(wb_labels.shape[1], wb_valid.shape[1])
    if wb_valid.shape[1] < wb_labels.shape[1]
    else wb_labels.shape[1]
)

# For HS-only pairs: is the daid in KNN and was it filtered?
print(f"\n--- HS-only pairs: daid={TEST_DAID} in KNN? filtered? ---")
for qfx, dfx in hsonly[:10]:
    in_knn = [j for j in range(nc) if int(hs_labels[qfx, j]) == TEST_DAID]
    in_knn_valid = [j for j in in_knn if hs_valid[qfx, j]]
    print(
        f"  (qfx={qfx}, dfx={dfx}): KNN cols with daid={in_knn}, valid={in_knn_valid}"
    )

print(f"\n--- WB-only pairs: daid={TEST_DAID} in KNN? filtered? ---")
for qfx, dfx in wbonly[:10]:
    in_knn = [j for j in range(nw) if int(wb_labels[qfx, j]) == TEST_DAID]
    in_knn_valid = [j for j in in_knn if wb_valid[qfx, j]]
    print(
        f"  (qfx={qfx}, dfx={dfx}): KNN cols with daid={in_knn}, valid={in_knn_valid}"
    )

# Compare KNN label sets per qfx
print(f"\n--- qfx where KNN labels differ ---")
diff_qfx = 0
for qfx in range(hs_labels.shape[0]):
    hs_set_cols = {int(hs_labels[qfx, j]) for j in range(nc) if hs_valid[qfx, j]}
    wb_set_cols = {int(wb_labels[qfx, j]) for j in range(nw) if wb_valid[qfx, j]}
    if hs_set_cols != wb_set_cols:
        diff_qfx += 1
        if diff_qfx <= 5:
            print(
                f"  qfx={qfx}: HS daids={sorted(hs_set_cols)} WB daids={sorted(wb_set_cols)}"
            )
print(f"  Total qfx with different KNN daid sets: {diff_qfx}/{hs_labels.shape[0]}")
