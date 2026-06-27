#!/usr/bin/env python3
"""Debug SV drops: compare pre/post SV per-annot between hotspotter and WBIA."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HS = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("/artifacts/wbia-oracle/hotspotter-exact-batch50-20260626-203030")
)
WB = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path("/artifacts/wbia-oracle/wildme-wbia-latest-batch50-20260626-202423")
)
QI = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def load_arr(row, col, run_dir):
    v = row[col]
    if isinstance(v, np.ndarray):
        return v
    p = json.loads(v) if isinstance(v, str) else v
    if isinstance(p, dict):
        if "npy_path" in p:
            npy = p["npy_path"]
            full = Path(npy) if npy.startswith("/") else run_dir / "final_scores" / npy
            return np.load(str(full))
        if "values" in p:
            return np.array(p["values"])
    return np.array(p)


def get_fm_sizes(row, daids, run_dir):
    """Return {daid: fm_count} from fm_list_json column."""
    v = row["fm_list_json"]
    items = json.loads(v) if isinstance(v, str) else v
    result = {}
    for d, item in zip(daids, items):
        if isinstance(item, dict):
            if "npy_path" in item:
                npy = item["npy_path"]
                full = (
                    Path(npy) if npy.startswith("/") else run_dir / "final_scores" / npy
                )
                arr = np.load(str(full))
                result[int(d)] = len(arr)
            elif "values" in item:
                result[int(d)] = len(item["values"])
    return result


def main():
    # Load pre-SV
    hs_pre_df = pd.read_parquet(HS / "chipmatches_pre_sv" / f"default_{QI:06d}.parquet")
    wb_pre_df = pd.read_parquet(
        WB / "chipmatches_pre_sv" / f"sv_on_true_{QI:06d}.parquet"
    )
    hs_post_df = pd.read_parquet(
        HS / "chipmatches_post_sv" / f"default_{QI:06d}.parquet"
    )
    wb_post_df = pd.read_parquet(
        WB / "chipmatches_post_sv" / f"sv_on_true_{QI:06d}.parquet"
    )

    hs_pre_r = hs_pre_df.iloc[0]
    wb_pre_r = wb_pre_df.iloc[0]
    hs_post_r = hs_post_df.iloc[0]
    wb_post_r = wb_post_df.iloc[0]

    hs_pre_d = load_arr(hs_pre_r, "daid_list_array", HS).astype(int)
    wb_pre_d = load_arr(wb_pre_r, "daid_list_array", WB).astype(int)
    hs_post_d = load_arr(hs_post_r, "daid_list_array", HS).astype(int)
    wb_post_d = load_arr(wb_post_r, "daid_list_array", WB).astype(int)

    hs_kept = set(hs_post_d)
    wb_kept = set(wb_post_d)
    wb_kept_hs_dropped = wb_kept - hs_kept
    hs_kept_wb_dropped = hs_kept - wb_kept
    both_kept = hs_kept & wb_kept
    both_dropped = (set(hs_pre_d) - hs_kept) & (set(wb_pre_d) - wb_kept)

    print(f"Query {QI}")
    print(
        f"  HS: {len(hs_pre_d)} pre → {len(hs_post_d)} post (drop {len(hs_pre_d)-len(hs_post_d)})"
    )
    print(
        f"  WB: {len(wb_pre_d)} pre → {len(wb_post_d)} post (drop {len(wb_pre_d)-len(wb_post_d)})"
    )
    print(
        f"  Pre daid Jaccard: {len(set(hs_pre_d)&set(wb_pre_d))/len(set(hs_pre_d)|set(wb_pre_d)):.3f}"
    )
    print(f"  Post daid Jaccard: {len(both_kept)/len(hs_kept|wb_kept):.3f}")
    print(f"  Both kept: {len(both_kept)}  Both dropped: {len(both_dropped)}")
    print(
        f"  WB kept, HS dropped: {len(wb_kept_hs_dropped)} — {sorted(wb_kept_hs_dropped)}"
    )
    print(
        f"  HS kept, WB dropped: {len(hs_kept_wb_dropped)} — {sorted(hs_kept_wb_dropped)}"
    )

    # FM sizes
    hs_fm = get_fm_sizes(hs_pre_r, hs_pre_d, HS)
    wb_fm = get_fm_sizes(wb_pre_r, wb_pre_d, WB)

    # Keypoint counts from features_keypoints
    hs_kp = {}
    wb_kp = {}
    hs_kp_df = pd.read_parquet(HS / "features_keypoints" / f"default_{QI:06d}.parquet")
    wb_kp_df = pd.read_parquet(
        WB / "features_keypoints" / f"sv_on_true_{QI:06d}.parquet"
    )
    for _, r in hs_kp_df.iterrows():
        hs_kp[int(r["aid"])] = int(r["num_keypoints"])
    for _, r in wb_kp_df.iterrows():
        wb_kp[int(r["aid"])] = int(r["num_keypoints"])

    print(f"\n{'WB kept, HS dropped':—^70}")
    print(
        f"{'daid':>5}  {'HS fm':>7} {'WB fm':>7} {'HS kp':>7} {'WB kp':>7}  {'fm diff':>7}"
    )
    print(f"{'—'*5}  {'—'*7} {'—'*7} {'—'*7} {'—'*7}  {'—'*7}")
    for d in sorted(wb_kept_hs_dropped):
        hfm = hs_fm.get(d, -1)
        wfm = wb_fm.get(d, -1)
        hkp = hs_kp.get(d, -1)
        wkp = wb_kp.get(d, -1)
        print(f"  {d:>3}  {hfm:>7} {wfm:>7} {hkp:>7} {wkp:>7}  {wfm-hfm:>+7}")

    print(f"\n{'HS kept, WB dropped':—^70}")
    for d in sorted(hs_kept_wb_dropped):
        hfm = hs_fm.get(d, -1)
        wfm = wb_fm.get(d, -1)
        print(
            f"  {d:>3}  HS fm={hfm} WB fm={wfm}  HS kp={hs_kp.get(d,-1)} WB kp={wb_kp.get(d,-1)}"
        )

    print(f"\n{'Both kept (compare csum)':—^70}")
    print(
        f"{'daid':>5}  {'HS csum':>10} {'WB csum':>10}  {'Δ':>10}  {'HS fm':>7} {'WB fm':>7}"
    )
    print(f"{'—'*5}  {'—'*10} {'—'*10}  {'—'*10}  {'—'*7} {'—'*7}")

    hs_final_df = pd.read_parquet(HS / "final_scores" / f"default_{QI:06d}.parquet")
    wb_final_df = pd.read_parquet(WB / "final_scores" / f"sv_on_true_{QI:06d}.parquet")
    hs_csum_arr = load_arr(hs_final_df.iloc[0], "annot_score_list_array", HS)
    wb_csum_arr = load_arr(wb_final_df.iloc[0], "annot_score_list_array", WB)

    csum_hs = {
        int(d): float(c) for d, c in zip(hs_post_d, hs_csum_arr) if np.isfinite(c)
    }
    csum_wb = {
        int(d): float(c) for d, c in zip(wb_post_d, wb_csum_arr) if np.isfinite(c)
    }

    for d in sorted(both_kept):
        ch = csum_hs.get(d, 0)
        cw = csum_wb.get(d, 0)
        hfm = hs_fm.get(d, -1)
        wfm = wb_fm.get(d, -1)
        print(f"  {d:>3}  {ch:>10.4f} {cw:>10.4f}  {ch-cw:>+10.4f}  {hfm:>7} {wfm:>7}")

    # SV compare: run with exact same inputs
    print(f"\n{'SV ENGINE TEST':—^70}")
    import importlib

    sver = importlib.import_module("vtool.spatial_verification")

    # Pick a test daid that WB kept but HS dropped
    test_daids = sorted(wb_kept_hs_dropped)
    if not test_daids:
        print("No annotations to test!")
        return 0

    # We need to load actual keypoints and fm_lists
    # Load query keypoints
    q_kp_arrs = {}
    for f in sorted((HS / "features_keypoints" / "arrays").glob("*_keypoints.npy")):
        pass  # need mapping

    # Simpler: compare first droppable annotation
    print(f"\nTesting daid {test_daids[0]}...")

    # Get HS and WB fm arrays for this daid
    hs_fm_raw = hs_pre_r["fm_list_json"]
    wb_fm_raw = wb_pre_r["fm_list_json"]
    if isinstance(hs_fm_raw, str):
        hs_fm_raw = json.loads(hs_fm_raw)
    if isinstance(wb_fm_raw, str):
        wb_fm_raw = json.loads(wb_fm_raw)

    hs_fm_arr = None
    wb_fm_arr = None
    for d, item in zip(hs_pre_d, hs_fm_raw):
        if int(d) == test_daids[0] and isinstance(item, dict) and "npy_path" in item:
            hs_fm_arr = np.load(item["npy_path"])
            break
    for d, item in zip(wb_pre_d, wb_fm_raw):
        if int(d) == test_daids[0] and isinstance(item, dict) and "npy_path" in item:
            wb_fm_arr = np.load(item["npy_path"])
            break

    print(f"  HS fm shape: {hs_fm_arr.shape if hs_fm_arr is not None else 'N/A'}")
    print(f"  WB fm shape: {wb_fm_arr.shape if wb_fm_arr is not None else 'N/A'}")

    if hs_fm_arr is not None and wb_fm_arr is not None:
        hs_pairs = {tuple(r) for r in hs_fm_arr}
        wb_pairs = {tuple(r) for r in wb_fm_arr}
        overlap = hs_pairs & wb_pairs
        print(
            f"  fm pair overlap: {len(overlap)} / {min(len(hs_pairs), len(wb_pairs))}"
        )
        print(f"  fm pair Jaccard: {len(overlap)/len(hs_pairs|wb_pairs):.3f}")
        if len(hs_pairs - wb_pairs) <= 10:
            print(f"  HS-only pairs: {hs_pairs - wb_pairs}")
        if len(wb_pairs - hs_pairs) <= 10:
            print(f"  WB-only pairs: {wb_pairs - hs_pairs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
