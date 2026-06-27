#!/usr/bin/env python3
"""Controlled RANSAC experiment: feed WBIA's pre-SV fm_list into HS's SV.

Tests whether RANSAC behavior differs when given identical fm input.
If post-SV daid Jaccard jumps → gap is fm-list. If it stays low → gap is RANSAC.
"""

import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import importlib

HS_RUN = Path(sys.argv[1])  # hotspotter run dir
WB_RUN = Path(sys.argv[2])  # WBIA run dir
QI = int(sys.argv[3])  # query index
HS_PREFIX = "default"
WB_PREFIX = "sv_on_true"

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


def load_keypoints(run_dir, prefix, qi):
    """Return {aid: keypoints_array}"""
    df = pd.read_parquet(run_dir / "features_keypoints" / f"{prefix}_{qi:06d}.parquet")
    kp_map = {}
    for _, row in df.iterrows():
        aid = int(row["aid"])
        kps = _npy(row["keypoints_array"], run_dir)
        kp_map[aid] = kps
    return kp_map


def load_pre_sv(run_dir, prefix, qi):
    """Return {aid: fm_array} for all pre-SV annotations."""
    df = pd.read_parquet(run_dir / "chipmatches_pre_sv" / f"{prefix}_{qi:06d}.parquet")
    row = df.iloc[0]
    daids = _npy(row["daid_list_array"], run_dir).astype(int)
    fm_raw = row["fm_list_json"]
    if isinstance(fm_raw, str):
        fm_raw = json.loads(fm_raw)
    fm_by_aid = {}
    for d, item in zip(daids, fm_raw):
        if isinstance(item, dict) and "npy_path" in item:
            full = Path(item["npy_path"])
            if not full.is_absolute():
                full = run_dir / "final_scores" / item["npy_path"]
            fm_by_aid[int(d)] = np.load(str(full))
    return fm_by_aid


def load_post_sv(run_dir, prefix, qi):
    """Return set of daids that passed SV."""
    df = pd.read_parquet(run_dir / "chipmatches_post_sv" / f"{prefix}_{qi:06d}.parquet")
    row = df.iloc[0]
    daids = _npy(row["daid_list_array"], run_dir).astype(int)
    return set(daids.tolist())


def main():
    print(f"Q{QI} | HS={HS_RUN.name} | WB={WB_RUN.name}")

    # Load keypoints from BOTH systems
    hs_kps = load_keypoints(HS_RUN, HS_PREFIX, QI)
    wb_kps = load_keypoints(WB_RUN, WB_PREFIX, QI)
    print(f"Keypoints: HS has {len(hs_kps)} annots, WB has {len(wb_kps)} annots")

    # Load pre-SV fm lists
    hs_pre_fm = load_pre_sv(HS_RUN, HS_PREFIX, QI)
    wb_pre_fm = load_pre_sv(WB_RUN, WB_PREFIX, QI)
    print(f"Pre-SV fm: HS={len(hs_pre_fm)} annots, WB={len(wb_pre_fm)} annots")

    # Load actual post-SV results
    hs_post = load_post_sv(HS_RUN, HS_PREFIX, QI)
    wb_post = load_post_sv(WB_RUN, WB_PREFIX, QI)
    print(f"Post-SV: HS={len(hs_post)} annots, WB={len(wb_post)} annots")
    daids = sorted(set(hs_pre_fm.keys()) | set(wb_pre_fm.keys()))
    pre_jac = len(set(hs_pre_fm) & set(wb_pre_fm)) / max(
        1, len(set(hs_pre_fm) | set(wb_pre_fm))
    )
    print(f"Pre-SV daid Jaccard: {pre_jac:.3f}")

    # Load query keypoints
    hs_qaid = int(
        pd.read_parquet(
            HS_RUN / "nearest_neighbors" / f"{HS_PREFIX}_{QI:06d}.parquet"
        ).iloc[0]["qaid"]
    )
    wb_qaid = int(
        pd.read_parquet(
            WB_RUN / "nearest_neighbors" / f"{WB_PREFIX}_{QI:06d}.parquet"
        ).iloc[0]["qaid"]
    )
    print(f"Query aid: HS={hs_qaid} WB={wb_qaid}")

    def get_chip_dims(run_dir, prefix, qi, aid):
        df = pd.read_parquet(run_dir / "chips" / f"{prefix}_{qi:06d}.parquet")
        for _, row in df.iterrows():
            if int(row["aid"]) == aid:
                size = _npy(row["chip_size"], run_dir)
                return size[1], size[0]  # (h, w) from (w, h)
        return 0, 0

    # EXPERIMENT: Run HS's SV with WBIA's exact fm_list and keypoints
    print(f"\n{'='*80}")
    print("EXPERIMENT: HS SV with WBIA's fm_list")
    print(f"{'='*80}")
    print(
        f"{'daid':>5}  {'fm |HS|':>8} {'|WB|':>8}  {'HS SV (own fm)':>15}  {'HS SV (WB fm)':>15}  {'WB real SV':>12}"
    )
    print(f"{'─'*5}  {'─'*8} {'─'*8}  {'─'*15}  {'─'*15}  {'─'*12}")

    # Use HS keypoints (same as WBIA since features are identical)
    q_kp = hs_kps[hs_qaid]

    sv_fm_pass = set()  # HS SV with WB fm → passed daids
    for daid in sorted(daids):
        if daid not in wb_pre_fm or daid not in hs_kps:
            continue

        wb_fm = wb_pre_fm[daid]
        db_kp = hs_kps[daid]

        h, w = get_chip_dims(HS_RUN, HS_PREFIX, QI, daid)
        dlen_sqrd2 = float(w**2 + h**2)

        match_weights = np.ones(len(wb_fm), dtype=np.float64)

        svtup = sver.spatially_verify_kpts(
            q_kp,
            db_kp,
            wb_fm,
            xy_thresh=0.01,
            scale_thresh=2.0,
            ori_thresh=np.pi / 2.0,
            dlen_sqrd2=dlen_sqrd2,
            min_nInliers=4,
            match_weights=match_weights,
            returnAff=True,
            refine_method="homog",
        )

        inliers = 0
        if svtup is not None:
            refined_inliers, _, _, _, _, _ = svtup
            inliers = len(refined_inliers)
            if inliers >= 4:
                sv_fm_pass.add(daid)

        hs_orig_pass = daid in hs_post
        wb_real_pass = daid in wb_post

        print(
            f"  {daid:>3}  {len(hs_pre_fm.get(daid, [])):>8} {len(wb_fm):>8}  {'PASS' if hs_orig_pass else 'FAIL':>15}  {'PASS' if inliers >= 4 else f'FAIL({inliers})':>15}  {'PASS' if wb_real_pass else 'FAIL':>12}"
        )

    # Compute post-SV daid metrics
    print(f"\n--- Post-SV comparison ---")
    print(f"HS-SV (own fm):   {len(hs_post)} daids")
    print(f"HS-SV (WB fm):    {len(sv_fm_pass)} daids")
    print(f"WB real SV:       {len(wb_post)} daids")

    expt_jac = len(sv_fm_pass & wb_post) / max(1, len(sv_fm_pass | wb_post))
    orig_jac = len(hs_post & wb_post) / max(1, len(hs_post | wb_post))
    print(f"Daid Jaccard (HS own fm vs WB):  {orig_jac:.3f}")
    print(f"Daid Jaccard (HS WB fm vs WB):  {expt_jac:.3f}")

    if expt_jac > orig_jac + 0.1:
        print(
            f"\n→ GAP IS FM-LIST: using WB's fm_list closes the gap ({orig_jac:.2f}→{expt_jac:.2f})"
        )
    else:
        print(
            f"\n→ GAP IS RANSAC: even with identical fm input, results diverge ({orig_jac:.2f}→{expt_jac:.2f})"
        )
        print("  Check: PRNG seeding, param wiring, hypothesis generation")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
