#!/usr/bin/env python3
"""Reproduce: run SV with exact WBIA inputs and compare inliers."""

import json, sys, importlib, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HS = Path(sys.argv[1])
WB = Path(sys.argv[2])
QI = int(sys.argv[3])
DAID = int(sys.argv[4])
sver = importlib.import_module("vtool.spatial_verification")


def _npy(v, run):
    if isinstance(v, np.ndarray):
        return v
    p = json.loads(v) if isinstance(v, str) else v
    if isinstance(p, dict):
        if "npy_path" in p:
            full = Path(p["npy_path"])
            if not full.is_absolute():
                full = run / "final_scores" / p["npy_path"]
            return np.load(str(full))
        if "values" in p:
            return np.array(p["values"])
    return np.array(p)


# Load query keypoints
hs_kp = pd.read_parquet(HS / "features_keypoints" / f"default_{QI:06d}.parquet")
wb_kp = pd.read_parquet(WB / "features_keypoints" / f"sv_on_true_{QI:06d}.parquet")
hs_qaid = int(
    pd.read_parquet(HS / "nearest_neighbors" / f"default_{QI:06d}.parquet").iloc[0][
        "qaid"
    ]
)
wb_qaid = int(
    pd.read_parquet(WB / "nearest_neighbors" / f"sv_on_true_{QI:06d}.parquet").iloc[0][
        "qaid"
    ]
)

q_kp_hs = _npy(hs_kp[hs_kp["aid"] == hs_qaid].iloc[0]["keypoints_array"], HS)
q_kp_wb = _npy(wb_kp[wb_kp["aid"] == wb_qaid].iloc[0]["keypoints_array"], WB)
db_kp_hs = _npy(hs_kp[hs_kp["aid"] == DAID].iloc[0]["keypoints_array"], HS)
db_kp_wb = _npy(wb_kp[wb_kp["aid"] == DAID].iloc[0]["keypoints_array"], WB)

print(f"HS q_kp: {q_kp_hs.shape} md5={hashlib.md5(q_kp_hs.tobytes()).hexdigest()[:12]}")
print(f"WB q_kp: {q_kp_wb.shape} md5={hashlib.md5(q_kp_wb.tobytes()).hexdigest()[:12]}")
print(
    f"HS db_kp: {db_kp_hs.shape} md5={hashlib.md5(db_kp_hs.tobytes()).hexdigest()[:12]}"
)
print(
    f"WB db_kp: {db_kp_wb.shape} md5={hashlib.md5(db_kp_wb.tobytes()).hexdigest()[:12]}"
)
print(f"kp match: q={np.allclose(q_kp_hs,q_kp_wb)} db={np.allclose(db_kp_hs,db_kp_wb)}")

# Load fm_list
hs_pre = pd.read_parquet(HS / "chipmatches_pre_sv" / f"default_{QI:06d}.parquet").iloc[
    0
]
wb_pre = pd.read_parquet(
    WB / "chipmatches_pre_sv" / f"sv_on_true_{QI:06d}.parquet"
).iloc[0]


def get_fm(row, run, daid):
    daids = _npy(row["daid_list_array"], run).astype(int)
    r = row["fm_list_json"]
    items = json.loads(r) if isinstance(r, str) else r
    for d, item in zip(daids, items):
        if int(d) == daid and isinstance(item, dict) and "npy_path" in item:
            full = Path(item["npy_path"])
            if not full.is_absolute():
                full = run / "final_scores" / item["npy_path"]
            return np.load(str(full))
    return None


hs_fm = get_fm(hs_pre, HS, DAID)
wb_fm = get_fm(wb_pre, WB, DAID)
if hs_fm is not None:
    print(f"HS fm: {hs_fm.shape} md5={hashlib.md5(hs_fm.tobytes()).hexdigest()[:12]}")
if wb_fm is not None:
    print(f"WB fm: {wb_fm.shape} md5={hashlib.md5(wb_fm.tobytes()).hexdigest()[:12]}")
if hs_fm is not None and wb_fm is not None:
    print(f"fm match: {np.array_equal(hs_fm, wb_fm)}")

# Chip dims
hs_ch = pd.read_parquet(HS / "chips" / f"default_{QI:06d}.parquet")
wb_ch = pd.read_parquet(WB / "chips" / f"sv_on_true_{QI:06d}.parquet")
hs_sz = _npy(hs_ch[hs_ch["aid"] == DAID].iloc[0]["chip_size"], HS)
wb_sz = _npy(wb_ch[wb_ch["aid"] == DAID].iloc[0]["chip_size"], WB)
hs_dlen = float(hs_sz[1] ** 2 + hs_sz[0] ** 2)
wb_dlen = float(wb_sz[1] ** 2 + wb_sz[0] ** 2)
print(f"HS chip: {hs_sz} dlen={hs_dlen:.0f}")
print(f"WB chip: {wb_sz} dlen={wb_dlen:.0f}")

# CALL SV with WB fm
print(f"\n--- SV with WB fm on HS keypoints ---")
mw = np.ones(len(wb_fm), dtype=np.float64)
svtup_hs = sver.spatially_verify_kpts(
    q_kp_hs,
    db_kp_hs,
    wb_fm,
    xy_thresh=0.01,
    scale_thresh=2.0,
    ori_thresh=np.pi / 2.0,
    dlen_sqrd2=hs_dlen,
    min_nInliers=4,
    match_weights=mw,
    returnAff=True,
    refine_method="homog",
)

hs_n = len(svtup_hs[0]) if svtup_hs else 0
print(f"HS kpts + WB fm: {'PASS' if svtup_hs else 'FAIL'} ({hs_n} inliers)")

# CALL SV with WB fm on WB keypoints
svtup_wb = sver.spatially_verify_kpts(
    q_kp_wb,
    db_kp_wb,
    wb_fm,
    xy_thresh=0.01,
    scale_thresh=2.0,
    ori_thresh=np.pi / 2.0,
    dlen_sqrd2=wb_dlen,
    min_nInliers=4,
    match_weights=mw,
    returnAff=True,
    refine_method="homog",
)
wb_n = len(svtup_wb[0]) if svtup_wb else 0
print(f"WB kpts + WB fm: {'PASS' if svtup_wb else 'FAIL'} ({wb_n} inliers)")

# The acid test: does the HS SV engine produce the same inliers as WB SV engine with same inputs?
print(f"\n--- Self-consistency test ---")
for i in range(5):
    svtup = sver.spatially_verify_kpts(
        q_kp_wb,
        db_kp_wb,
        wb_fm,
        xy_thresh=0.01,
        scale_thresh=2.0,
        ori_thresh=np.pi / 2.0,
        dlen_sqrd2=wb_dlen,
        min_nInliers=4,
        match_weights=mw,
        returnAff=True,
        refine_method="homog",
    )
    n = len(svtup[0]) if svtup else 0
    print(f"  Run {i}: {'PASS' if svtup else 'FAIL'} ({n} inliers)")
