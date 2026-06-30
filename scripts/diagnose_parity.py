#!/usr/bin/env python3
"""Differential diagnosis: feed WBIA intermediate results into HS pipeline stages.

Answers: if we plug WBIA's FLANN neighbours into HS's downstream, do the scores match?
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from hotspotter.config import HotSpotterConfig, IdentificationConfig
from hotspotter.data import AnnotatedImage, FeatureSet, Match, ScoredMatch
from hotspotter.name_scoring import score_matches_with_names
from hotspotter.scoring import baseline_filter, build_matches, weight_neighbors_lnbnn

SIFT_MAX_SQRT_DIST = 2.0 * (512.0**2.0)


def load_array(run_dir: Path, array_meta: dict | str) -> np.ndarray | None:
    """Load a numpy array from parquet *_array metadata."""
    if isinstance(array_meta, str):
        array_meta = json.loads(array_meta)
    if array_meta is None:
        return None
    path_str = array_meta.get("npy_path")
    if path_str:
        path = Path(path_str)
        # Resolve common path issues
        if not path.exists():
            # Try relative to run_dir with tail matching
            for candidate in run_dir.rglob(path.name):
                path = candidate
                break
        if path.exists():
            arr = np.load(path, allow_pickle=True)
            if hasattr(arr, "keys"):
                arr = arr[list(arr.keys())[0]]
            return np.asarray(arr)
    values = array_meta.get("values")
    if values is not None:
        return np.asarray(values)
    return None


def load_wbia_neighbors(
    oracle: Path, config: str, query_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load WBIA's raw FLANN distances and labels."""
    df = pd.read_parquet(
        oracle / "nearest_neighbors" / f"{config}_{query_idx:06d}.parquet"
    )
    dists = load_array(oracle, df.iloc[0]["neighbor_dists_array"])
    labels = load_array(oracle, df.iloc[0]["neighbor_idxs_array"])
    return dists, labels


def load_wbia_daid_list(
    oracle: Path, stage: str, config: str, query_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load WBIA's daid_list, annot_scores, name_scores, score_list."""
    df = pd.read_parquet(oracle / stage / f"{config}_{query_idx:06d}.parquet")
    daids = load_array(oracle, df.iloc[0]["daid_list_array"])
    annot_scores = load_array(oracle, df.iloc[0]["annot_score_list_array"])
    name_scores = load_array(oracle, df.iloc[0]["name_score_list_array"])
    score_list = load_array(oracle, df.iloc[0]["score_list_array"])
    return daids, annot_scores, name_scores, score_list


def diagnose_query(
    oracle: Path,
    database: list[AnnotatedImage],
    query_annot_index: int,
    config: str,
):
    """Feed WBIA's FLANN results into HS pipeline and compare outputs."""
    hs = HotSpotterConfig(
        knn=4,
        knorm=1,
        kpad=0,
        sv_on=True,
        fg_on=False,
        kpad_policy="dynamic",
    )
    k = hs.knn
    kpad = 0  # WBIA uses kpad=0 for this config
    knorm = hs.knorm

    print(f"\n{'='*60}")
    print(f"Config: {config}, Query: {query_annot_index}")
    print(f"K={k}, Kpad={kpad}, Knorm={knorm}")
    print(f"{'='*60}")

    # Step 1: Load WBIA's raw FLANN results
    wbia_dists, wbia_labels = load_wbia_neighbors(oracle, config, 0)
    print(f"\nWBIA raw_dists shape: {wbia_dists.shape}, dtype: {wbia_dists.dtype}")

    # Step 2: Normalize distances (same as HS's _normalize_distances)
    # Try BOTH with and without normalization to match WBIA
    dists_normalized = (np.maximum(wbia_dists, 0.0) / SIFT_MAX_SQRT_DIST).astype(
        np.float64
    )
    dists_normalized = np.sqrt(dists_normalized)

    # Also try raw (no normalization) — WBIA may use raw distances
    dists_raw = np.maximum(wbia_dists, 0.0).astype(np.float64)
    dists_raw = np.sqrt(dists_raw)

    print(
        f"\nRaw WBIA dist stats: min={dists_raw.min():.1f}, max={dists_raw.max():.1f}, mean={dists_raw.mean():.1f}"
    )
    print(
        f"Normalized dist stats: min={dists_normalized.min():.6f}, max={dists_normalized.max():.6f}, mean={dists_normalized.mean():.6f}"
    )

    # Use the trace distances directly — they are already normalized by WBIA
    dists_norm = wbia_dists.astype(np.float64)
    print(
        f"Using trace dists as-is: min={dists_norm.min():.6f}, max={dists_norm.max():.6f}"
    )

    # Step 3: Build vote columns (same logic as _build_vote_columns)
    n_qfxs = dists_norm.shape[0]
    voting_dists = dists_norm[:, : k + kpad]
    norm_dists = dists_norm[:, k + kpad :]
    voting_annot = np.full((n_qfxs, k + kpad), -1, dtype=np.int32)
    voting_feat = np.full((n_qfxs, k + kpad), -1, dtype=np.int32)

    # Map label indices → annot/feat indices
    # WBIA's labels reference a query-excluded index — build annot_of_desc
    # from non-query annotations only, mapping positions to original database indices
    non_query = [(i, ann) for i, ann in enumerate(database) if i != query_annot_index]
    n_total = sum(len(ann.features) for _, ann in non_query)
    annot_of_desc = np.empty(n_total, dtype=np.int32)
    feat_of_desc = np.empty(n_total, dtype=np.int32)
    offset = 0
    for orig_idx, ann in non_query:
        n = len(ann.features)
        annot_of_desc[offset : offset + n] = orig_idx
        feat_of_desc[offset : offset + n] = np.arange(n, dtype=np.int32)
        offset += n

    for j in range(k + kpad):
        col = wbia_labels[:, j]
        valid = (col >= 0) & (col < n_total)
        voting_annot[valid, j] = annot_of_desc[col[valid]]
        voting_feat[valid, j] = feat_of_desc[col[valid]]

    print(f"  Voting annots: {np.unique(voting_annot[voting_annot >= 0])}")
    print(f"  Query annot index: {query_annot_index}")

    # Step 4: Baseline filter
    invalid, impossible_set, qname = baseline_filter(
        voting_annot,
        database,
        query_annot_index,
        can_match_samename=True,
        can_match_sameimg=False,
    )
    # Compare with WBIA's baseline filter
    try:
        df_bf = pd.read_parquet(
            oracle / "baseline_neighbor_filter" / f"{config}_000000.parquet"
        )
        wbia_valid = load_array(oracle, df_bf.iloc[0]["valid_array"])
        if wbia_valid is not None:
            hs_invalid_flat = invalid.ravel()
            wbia_valid_flat = (
                wbia_valid.ravel() if wbia_valid.shape == invalid.shape else None
            )
            if wbia_valid_flat is not None:
                match_pct = (hs_invalid_flat == ~wbia_valid_flat).mean() * 100
                print(f"\nBaseline filter match: {match_pct:.1f}%")
    except Exception as e:
        print(f"  (could not compare baseline: {e})")

    # Step 5: LNBNN weights
    weights = weight_neighbors_lnbnn(
        voting_dists,
        norm_dists,
        normonly_on=False,
        bar_l2_on=False,
        ratio_thresh=None,
        lnbnn_ratio=1.0,
    )
    print(f"\nHS weights shape: {weights.shape}")
    print(
        f"HS weights stats: min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}, nonzero={int((weights > 0).sum())}"
    )

    # Compare with WBIA's neighbor_weights
    try:
        df_nw = pd.read_parquet(
            oracle / "neighbor_weights" / f"{config}_000000.parquet"
        )
        wbia_w = load_array(oracle, df_nw.iloc[0]["weight_lnbnn_array"])
        if wbia_w is not None and wbia_w.shape == weights.shape:
            pearson = np.corrcoef(weights.ravel(), wbia_w.ravel())[0, 1]
            print(f"WBIA weights shape: {wbia_w.shape}")
            print(
                f"WBIA weights stats: min={wbia_w.min():.6f}, max={wbia_w.max():.6f}, mean={wbia_w.mean():.6f}, nonzero={int((wbia_w > 0).sum())}"
            )
            print(f"Weights Pearson r: {pearson:.6f}")
    except Exception as e:
        print(f"  (could not compare weights: {e})")

    # Step 6: Build matches
    matches = build_matches(
        weights, voting_annot, voting_feat, invalid, database, k, kpad
    )
    print(f"\nHS matches: {len(matches)}")
    print(f"  Unique qfx: {len({m.qfx for m in matches})}")
    print(f"  Unique daid: {len({m.daid for m in matches})}")
    daids_set = {m.daid for m in matches}
    print(f"  daids: {sorted(daids_set)}")

    # Step 7: Score matches
    annot_uuids = [a.annot_uuid for a in database]
    annot_name_map = {
        a.annot_uuid: a.name_uuid for a in database if a.name_uuid is not None
    }
    csum, name_scores, canonical = score_matches_with_names(
        matches, annot_uuids, annot_name_map, "nsum_wbia"
    )

    daid_scores = np.full(len(database), -np.inf)
    for annot_uuid, score in canonical.items():
        idx = annot_uuids.index(annot_uuid)
        daid_scores[idx] = score

    # Compare with WBIA final scores
    try:
        wbia_daids, wbia_annot, wbia_name, wbia_score = load_wbia_daid_list(
            oracle, "final_scores", config, 0
        )
        if wbia_daids is not None and wbia_score is not None:
            print(f"\nWBIA final daids: {wbia_daids.shape}, scores: {wbia_score.shape}")

            # Align daids: WBIA uses 1-based, HS uses 0-based
            wbia_daids_0b = wbia_daids - 1
            wbia_score_map = {
                int(d): float(s) for d, s in zip(wbia_daids_0b, wbia_score)
            }

            hs_daids_with_score = [
                (int(d), float(s)) for d, s in enumerate(daid_scores) if np.isfinite(s)
            ]
            print(f"HS scored daids: {len(hs_daids_with_score)}")
            print(f"WBIA scored daids: {len(wbia_score_map)}")

            # Print side-by-side
            common = sorted(
                set(int(d) for d, _ in hs_daids_with_score) & set(wbia_score_map.keys())
            )
            if common:
                print(f"Common daids: {len(common)}")
                print(f"{'daid':>6} {'HS_score':>10} {'WBIA_score':>10} {'delta':>10}")
                for daid in common:
                    hs_s = daid_scores[daid]
                    wb_s = wbia_score_map[daid]
                    print(
                        f"{daid:>6} {hs_s:>10.6f} {wb_s:>10.6f} {abs(hs_s - wb_s):>10.6f}"
                    )
            else:
                print("NO COMMON DAIDS between HS and WBIA!")
    except Exception as e:
        print(f"  (could not compare final scores: {e})")


def main():
    oracle = Path("/artifacts/wbia-oracle/wildme-wbia-nightly-20260629-115910")
    config = "sv_on_true"

    # Load the reference batch and build database
    import sys

    sys.path.insert(0, "/app")
    from scripts.run_fixture import build_database, load_batch

    batch_path = Path("/app/pipeline/tests/reference_batch.json")
    image_dir = Path("/app/pipeline/tests/assets/images")

    if not batch_path.exists() or not image_dir.exists():
        print("Batch/images not found. Run inside Docker with proper mounts.")
        return 1

    batch = load_batch(batch_path)
    database, query_indices, _ = build_database(batch, image_dir)

    print(f"Database: {len(database)} annotations")
    print(f"Queries at indices: {query_indices}")
    print(f"Features per annot: {[len(a.features) for a in database]}")

    for qidx in query_indices[:1]:
        diagnose_query(oracle, database, qidx, config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
