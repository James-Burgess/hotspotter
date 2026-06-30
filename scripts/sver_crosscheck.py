#!/usr/bin/env python3
"""Cross-check: feed WBIA pre-SV fm_list through HS sver, and vice versa.

Also test HS sver determinism on identical input.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ORACLE = Path("/artifacts/wbia-oracle/wildme-wbia-nightly-20260629-161926")


def find_hs_trace() -> Path | None:
    import os as _os

    for d in sorted(_os.listdir("/tmp"), reverse=True):
        dp = Path(f"/tmp/{d}")
        if (
            dp.is_dir()
            and d.startswith("hotspotter-trace-")
            and (dp / "linear" / "chipmatches_pre_sv").exists()
        ):
            return dp / "linear"
    return None


def load_fm_arrays(run: Path, stage: str, file_name: str) -> list[np.ndarray]:
    df = pd.read_parquet(run / stage / file_name)
    fm_json = df.iloc[0]["fm_list_json"]
    fm_meta = json.loads(fm_json) if isinstance(fm_json, str) else fm_json
    arrays = []
    for entry in fm_meta:
        npy = entry.get("npy_path", "")
        fname = Path(npy).name if npy else None
        if fname:
            candidates = list(run.rglob(fname))
            if candidates:
                arr = np.load(str(candidates[0]), allow_pickle=True)
                if hasattr(arr, "keys"):
                    arr = arr[list(arr.keys())[0]]
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 2)
                arrays.append(arr)
    return arrays


def main():
    hs = find_hs_trace()
    if hs is None:
        print("No HS trace found")
        return 1

    config, qi = "sv_on_true", 0
    fname = f"{config}_{qi:06d}.parquet"

    w_fms = load_fm_arrays(ORACLE, "chipmatches_pre_sv", fname)
    h_fms = load_fm_arrays(hs, "chipmatches_pre_sv", fname)

    print(f"WBIA pre-SV: {len(w_fms)} annots, {sum(len(f) for f in w_fms)} matches")
    print(f"HS pre-SV:   {len(h_fms)} annots, {sum(len(f) for f in h_fms)} matches")

    # Find an annotation with many matches in both systems
    best_annot = None
    best_size = 0
    for i, (wf, hf) in enumerate(zip(w_fms, h_fms)):
        w_set = {tuple(r) for r in wf}
        h_set = {tuple(r) for r in hf}
        overlap = len(w_set & h_set)
        union = len(w_set | h_set)
        jac = overlap / union if union else 0
        min_size = min(len(wf), len(hf))
        if min_size > best_size and jac > 0.95:
            best_size = min_size
            best_annot = (i, wf, hf, jac)

    if best_annot is None:
        print("No suitable annotation found")
        return 1

    annot_idx, wf, hf, jac = best_annot
    print(
        f"\nSelected annot {annot_idx}: WBIA={len(wf)}, HS={len(hf)}, Jaccard={jac:.4f}"
    )

    # Check if the fm arrays are identical (ordered)
    wf_i32 = wf.astype(np.int32)
    hf_i32 = hf.astype(np.int32)
    arrays_equal = np.array_equal(wf_i32, hf_i32)
    print(f"  Fm arrays equal (ordered): {arrays_equal}")
    if not arrays_equal:
        w_set = {tuple(r) for r in wf}
        h_set = {tuple(r) for r in hf}
        only_w = w_set - h_set
        only_h = h_set - w_set
        print(f"  WBIA-only pairs: {len(only_w)}, HS-only pairs: {len(only_h)}")
        if only_w:
            sample = sorted(only_w)[:3]
            print(f"  WBIA-only examples: {sample}")
        if only_h:
            sample = sorted(only_h)[:3]
            print(f"  HS-only examples: {sample}")

    # Now run sver on both fm arrays
    from hotspotter._vendor.sver._spatial_verification import spatially_verify_kpts
    from scripts.run_fixture import build_database, load_batch

    batch = load_batch(Path("/app/pipeline/tests/reference_batch.json"))
    database, qis, _ = build_database(batch, Path("/app/pipeline/tests/assets/images"))
    qidx = qis[0]
    q_kp = database[qidx].features.keypoints

    # Get the daid for this annot from the WBIA daid list
    df_w = pd.read_parquet(ORACLE / "chipmatches_pre_sv" / fname)
    w_daids = json.loads(df_w.iloc[0]["daid_list_array"]).get("values", [])

    if annot_idx >= len(w_daids):
        print(f"annot_idx {annot_idx} out of range for daid list ({len(w_daids)})")
        return 1

    daid_1based = int(w_daids[annot_idx])  # e.g., daid=5
    hs_annot_idx = daid_1based - 1  # 0-based
    print(f"\nWBIA daid={daid_1based}, HS annot_idx={hs_annot_idx}")

    db_ann = database[hs_annot_idx]
    db_kp = db_ann.features.keypoints
    chip_h, chip_w = db_ann.image.shape[:2]
    dlen_sqrd2 = float(chip_w**2 + chip_h**2)

    print(f"  DB annot {hs_annot_idx}: {len(db_kp)} kpts, chip {chip_w}x{chip_h}")

    def run_sver(fm: np.ndarray, label: str) -> tuple | None:
        """Run SV and return the result tuple or None."""
        if fm.shape[-1] != 2:
            fm = fm.reshape(-1, 2)
        fm_i32 = fm.astype(np.int32)
        valid = (fm_i32[:, 0] < len(q_kp)) & (fm_i32[:, 1] < len(db_kp))
        fm_i32 = fm_i32[valid]

        match_weights = np.ones(len(fm_i32), dtype=np.float64)
        try:
            result = spatially_verify_kpts(
                q_kp,
                db_kp,
                fm_i32,
                xy_thresh=0.01,
                scale_thresh=2.0,
                ori_thresh=np.pi / 2.0,
                dlen_sqrd2=dlen_sqrd2,
                min_nInliers=4,
                match_weights=match_weights,
                full_homog_checks=True,
                returnAff=True,
                refine_method="homog",
            )
        except Exception as exc:
            print(f"    {label} error: {exc}")
            return None
        return result

    # Run WBIA fm through HS sver
    print("\n=== Running WBIA fm_list through HS sver ===")
    try:
        w_result = run_sver(wf_i32, "WBIA-fm")
        if w_result is None:
            print("  PASS = None (affine < 7)")
        else:
            refined, errors, H, aff_inl, aff_errs, Aff = w_result
            print(f"  PASS: refined={len(refined)} inliers, affine={len(aff_inl)}")

        # Run HS fm through HS sver
        print(f"\n=== Running HS fm_list through HS sver ===")
        h_result = run_sver(hf_i32, "HS-fm")
        if h_result is None:
            print("  PASS = None (affine < 7)")
        else:
            refined, errors, H, aff_inl, aff_errs, Aff = h_result
            print(f"  PASS: refined={len(refined)} inliers, affine={len(aff_inl)}")
    except Exception as e:
        print(f"  SV error: {e}")
        import traceback

        traceback.print_exc()

    # Determinism check: run HS sver twice on WBIA's fm
    print("\n=== Determinism check: HS sver × 2 on WBIA fm ===")
    results = []
    for i in range(2):
        r = run_sver(wf_i32, f"run{i}")
        if r is not None:
            refined, errors, H, aff_inl, aff_errs, Aff = r
            results.append(refined)
            print(f"  run {i}: refined inliers={sorted(refined)[:5]}...")
        else:
            results.append(None)
            print(f"  run {i}: None (affine < 7)")

    if results[0] is not None and results[1] is not None:
        same = np.array_equal(results[0], results[1])
        if not same:
            diff = (results[0] != results[1]).sum()
            print(f"  DETERMINISTIC? NO — {diff} elements differ")
        else:
            print(f"  DETERMINISTIC? YES — identical results")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
