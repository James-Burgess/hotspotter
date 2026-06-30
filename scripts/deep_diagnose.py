#!/usr/bin/env python3
"""Deep diagnosis: load WBIA raw FLANN labels and check what they actually contain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_array(run_dir: Path, array_meta_str: str) -> np.ndarray | None:
    meta = (
        json.loads(array_meta_str)
        if isinstance(array_meta_str, str)
        else array_meta_str
    )
    path_str = meta.get("npy_path")
    if path_str:
        path = Path(path_str)
        if not path.exists():
            for candidate in run_dir.rglob(path.name):
                path = candidate
                break
        if path.exists():
            arr = np.load(path, allow_pickle=True)
            if hasattr(arr, "keys"):
                arr = arr[list(arr.keys())[0]]
            return np.asarray(arr)
    values = meta.get("values")
    if values is not None:
        return np.asarray(values)
    return None


def main():
    oracle = Path("/artifacts/wbia-oracle/wildme-wbia-nightly-20260629-115910")
    config = "sv_on_true"
    qi = 0

    # ---- Load WBIA nearest_neighbors ----
    df_nn = pd.read_parquet(oracle / "nearest_neighbors" / f"{config}_{qi:06d}.parquet")
    raw_dists = load_raw_array(oracle, df_nn.iloc[0]["neighbor_dists_array"])
    raw_labels = load_raw_array(oracle, df_nn.iloc[0]["neighbor_idxs_array"])

    print(f"===== WBIA nearest_neighbors: {config} query {qi} =====")
    print(f"raw_dists shape: {raw_dists.shape}, dtype: {raw_dists.dtype}")
    print(f"raw_labels shape: {raw_labels.shape}, dtype: {raw_labels.dtype}")

    # What do the label values look like?
    print(f"\nraw_labels range: [{raw_labels.min()}, {raw_labels.max()}]")
    print(f"raw_labels unique values: {len(np.unique(raw_labels))}")
    print(f"\nColumn 0 (closest neighbor):")
    col0 = raw_labels[:, 0]
    print(f"  range: [{col0.min()}, {col0.max()}]")
    print(f"  unique: {len(np.unique(col0))}")

    # Check if labels are descriptor positions or daid values
    # Descriptor positions would range 0..N_descriptors (~36423)
    # Daid values (1-based) would range 1..19
    unique_sorted = sorted(np.unique(col0))
    print(f"  first 20 unique col0 values: {unique_sorted[:20]}")
    if len(unique_sorted) <= 30:
        print(f"  all unique col0 values: {unique_sorted}")
    print(f"  max label: {raw_labels.max()}")

    # ---- Load WBIA neighbor_weights ----
    try:
        df_nw = pd.read_parquet(
            oracle / "neighbor_weights" / f"{config}_{qi:06d}.parquet"
        )
        print(f"\n===== WBIA neighbor_weights =====")
        print(f"Columns: {list(df_nw.columns)}")
        for col in df_nw.columns:
            val = df_nw.iloc[0][col]
            if isinstance(val, str) and len(val) < 300:
                print(f"  {col}: {val[:200]}")

        # Find weight column
        weight_col = None
        for col in df_nw.columns:
            if "weight" in col.lower() or "lnbnn" in col.lower():
                weight_col = col
                break
        if weight_col is None and len(df_nw.columns) > 5:
            weight_col = df_nw.columns[5]  # try 6th column

        if weight_col is not None:
            w_meta = json.loads(df_nw.iloc[0][weight_col])
            w_arr = load_raw_array(oracle, df_nw.iloc[0][weight_col])
            if w_arr is not None:
                print(f"\nWBIA weights shape: {w_arr.shape}")
                print(
                    f"WBIA weights: min={w_arr.min():.6f}, max={w_arr.max():.6f}, "
                    f"mean={w_arr.mean():.6f}, nonzero={int((w_arr > 0).sum())}"
                )
        else:
            print("(no weight column found)")
    except Exception as e:
        print(f"neighbor_weights load error: {e}")

    # ---- Load WBIA baseline_filter valid array ----
    try:
        df_bf = pd.read_parquet(
            oracle / "baseline_neighbor_filter" / f"{config}_{qi:06d}.parquet"
        )
        bf_valid = load_raw_array(oracle, df_bf.iloc[0]["valid_array"])
        if bf_valid is not None:
            print(f"\n===== WBIA baseline_filter =====")
            print(f"valid shape: {bf_valid.shape}")
            print(f"valid dtype: {bf_valid.dtype}")
            print(f"valid=True count: {int(bf_valid.sum())} / {bf_valid.size}")
        else:
            # Try inline values
            val_str = df_bf.iloc[0].get("valid_array")
            if val_str:
                meta = json.loads(val_str) if isinstance(val_str, str) else val_str
                vals = meta.get("values")
                if vals is not None:
                    arr = np.array(vals)
                    print(f"\n===== WBIA baseline_filter (inline) =====")
                    print(f"valid shape: {arr.shape}")
                    print(f"valid=True count: {int(arr.sum())} / {arr.size}")
    except Exception as e:
        print(f"baseline_filter load error: {e}")

    # ---- Load chipmatches_pre_sv to see fm_list content ----
    try:
        df_cm = pd.read_parquet(
            oracle / "chipmatches_pre_sv" / f"{config}_{qi:06d}.parquet"
        )
        print(f"\n===== WBIA chipmatches_pre_sv =====")
        print(f"Columns: {list(df_cm.columns)}")
        fm_json = df_cm.iloc[0].get("fm_list_json")
        if fm_json:
            fm_meta = json.loads(fm_json) if isinstance(fm_json, str) else fm_json
            if isinstance(fm_meta, list) and len(fm_meta) > 0:
                print(f"  fm_list: {len(fm_meta)} arrays")
                for i, fm_entry in enumerate(fm_meta[:3]):
                    print(f"  fm[{i}]: shape={fm_entry.get('shape')}")
    except Exception as e:
        print(f"chipmatches load error: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
