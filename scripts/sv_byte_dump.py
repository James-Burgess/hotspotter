#!/usr/bin/env python3
"""Byte-level SV dump for a specific daid — settle input vs engine divergence.

Extracts q_kp, db_kp, fm_list, dlen_sqrd2 from WBIA trace,
runs spatially_verify_kpts with those exact inputs, compares to WBIA real SV.
"""

import json, sys, hashlib, importlib
from pathlib import Path
import numpy as np, pandas as pd

WB = Path(sys.argv[1])  # WBIA trace dir
QI = int(sys.argv[2])
DAID = int(sys.argv[3])
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("/tmp/sv_dump.npz")

sver = importlib.import_module("vtool.spatial_verification")


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


def load_kp(run_dir, prefix, qi):
    df = pd.read_parquet(run_dir / "features_keypoints" / f"{prefix}_{qi:06d}.parquet")
    return {
        int(r["aid"]): _npy(r["keypoints_array"], run_dir) for _, r in df.iterrows()
    }


def load_fm(run_dir, prefix, qi, daid):
    df = pd.read_parquet(run_dir / "chipmatches_pre_sv" / f"{prefix}_{qi:06d}.parquet")
    row = df.iloc[0]
    daids = _npy(row["daid_list_array"], run_dir).astype(int)
    fm_raw = row["fm_list_json"]
    if isinstance(fm_raw, str):
        fm_raw = json.loads(fm_raw)
    for d, item in zip(daids, fm_raw):
        if int(d) == daid and isinstance(item, dict) and "npy_path" in item:
            full = Path(item["npy_path"])
            if not full.is_absolute():
                full = run_dir / "final_scores" / item["npy_path"]
            return np.load(str(full))
    return None


def main():
    wb_prefix = "sv_on_true"
    hs_prefix = "default"

    # Find HS trace (same batch as WB)
    hs_runs = sorted(WB.parent.glob("hotspotter-v*-batch100-*"))
    hs_latest = hs_runs[-1] if hs_runs else None
    if hs_latest is None:
        print("No HS batch100 trace found — using WB inputs only")
    else:
        print(f"HS trace: {hs_latest.name}")

    # Get query aid
    wb_nn = pd.read_parquet(
        WB / "nearest_neighbors" / f"{wb_prefix}_{QI:06d}.parquet"
    ).iloc[0]
    wb_qaid = int(wb_nn["qaid"])

    # Load keypoints
    wb_kps = load_kp(WB, wb_prefix, QI)
    q_kp_wb = wb_kps[wb_qaid]
    db_kp_wb = wb_kps[DAID]

    # Load fm_list
    wb_fm = load_fm(WB, wb_prefix, QI, DAID)

    # Load chip dims for dlen_sqrd2
    wb_ch = pd.read_parquet(WB / "chips" / f"{wb_prefix}_{QI:06d}.parquet")
    wb_sz = _npy(wb_ch[wb_ch["aid"] == DAID].iloc[0]["chip_size"], WB)
    wb_dlen = float(wb_sz[1] ** 2 + wb_sz[0] ** 2)

    print(
        f"Q{QI} daid={DAID}: q_kp={q_kp_wb.shape} db_kp={db_kp_wb.shape} fm={wb_fm.shape if wb_fm is not None else 'None'} dlen={wb_dlen:.0f}"
    )

    # Save to npz
    np.savez(
        OUT,
        q_kp=q_kp_wb,
        db_kp=db_kp_wb,
        fm=wb_fm,
        dlen_sqrd2=np.array([wb_dlen]),
        qaid=np.array([wb_qaid]),
        daid=np.array([DAID]),
    )
    print(f"Saved inputs to {OUT}")

    # Check if daid 82 is in WB post-SV
    wb_post = pd.read_parquet(
        WB / "chipmatches_post_sv" / f"{wb_prefix}_{QI:06d}.parquet"
    ).iloc[0]
    wb_post_daids = set(_npy(wb_post["daid_list_array"], WB).astype(int))
    print(f"WB post-SV keeps daid {DAID}: {DAID in wb_post_daids}")
    print(f"WB post-SV count: {len(wb_post_daids)} annots")

    # Check if daid 82 is in HS post-SV (if HS trace available)
    if hs_latest is not None:
        hs_post = pd.read_parquet(
            hs_latest / "chipmatches_post_sv" / f"default_{QI:06d}.parquet"
        ).iloc[0]
        hs_post_daids = set(_npy(hs_post["daid_list_array"], hs_latest).astype(int))
        print(f"HS post-SV keeps daid {DAID}: {DAID in hs_post_daids}")
        print(f"HS post-SV count: {len(hs_post_daids)} annots")

    # Run SV with WB inputs
    mw = np.ones(len(wb_fm), dtype=np.float64)
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

    if svtup is None:
        print("SV returned None (affine < 4/7)")
    else:
        refined_n = len(svtup[0])
        affine_n = len(svtup[3])
        print(f"refined_inliers={refined_n}  affine_inliers={affine_n}")
        print(f"  refined >= 4: {refined_n >= 4}")
        print(f"  affine >= 4:  {affine_n >= 4}")
        print(f"  affine >= 7:  {affine_n >= 7}")

    # Self-consistency: run 5 times with same inputs
    print("\nSelf-consistency (5 runs, same inputs):")
    results = []
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
        if svtup is not None:
            results.append((len(svtup[0]), len(svtup[3])))
            print(f"  Run {i}: refined={results[-1][0]} affine={results[-1][1]}")
        else:
            results.append((0, 0))
            print(f"  Run {i}: None")

    refs = [r[0] for r in results]
    affs = [r[1] for r in results]
    print(f"refined: all same={len(set(refs)) == 1} values={set(refs)}")
    print(f"affine:  all same={len(set(affs)) == 1} values={set(affs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
