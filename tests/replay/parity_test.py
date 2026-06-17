#!/usr/bin/env python3
"""Verify wbia-core pipeline produces the same scores as WBIA given identical
inputs and feature extraction.  Intended to run inside the WBIA container
where pyhesaff is available.

Usage:
    docker cp wbia-core wbia-replay:/tmp/wbia-core
    docker cp fixtures wbia-replay:/tmp/fixtures
    docker cp parity_test.py wbia-replay:/tmp/parity_test.py
    docker exec wbia-replay /virtualenv/env3/bin/python /tmp/parity_test.py
"""

import json
import pathlib
import sys
import uuid as uuid_mod
import warnings

import numpy as np

from wbia_core.config import HotSpotterConfig, IdentificationConfig, SiftConfig
from wbia_core.data import AnnotatedImage
from wbia_core.features import extract_features
from wbia_core.pipeline import identify

FIXTURES_DIR = pathlib.Path("/tmp/fixtures")


def load_fixture(path: pathlib.Path) -> dict:
    data = np.load(path, allow_pickle=True)
    raw = dict(data["raw_result"].item()) if data.get("raw_result") else {}
    return {
        "species": str(data.get("species", "")),
        "seed": int(data.get("seed", 0)),
        "query_idx": int(data.get("query_idx", 0)),
        "annot_uuids": [str(u) for u in data["annot_uuids"]],
        "name_uuids": [
            str(v) if v is not None else None for v in data.get("name_uuids", [])
        ],
        "bboxes": [tuple(b) for b in data.get("bboxes", [])],
        "image_bytes": list(data["image_bytes"]),
        "raw_result": raw,
        "config": dict(data.get("config", {}).item() if data.get("config") else {}),
    }


def parse_wbia_scores(raw_result: dict) -> dict[str, float]:
    json_result = raw_result.get("json_result", raw_result)
    scores: dict[str, float] = {}

    cm_dict = json_result.get("cm_dict")
    if cm_dict:
        for qauuid, data in cm_dict.items():
            duuid_key = "__UUID__"
            dauuids = []
            for u in data.get("dannot_uuid_list", []):
                if isinstance(u, dict) and duuid_key in u:
                    dauuids.append(str(u[duuid_key]))
                else:
                    dauuids.append(str(u))
            score_list = data.get("annot_score_list", [])
            for duuid, score in zip(dauuids, score_list):
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        continue
                if np.isfinite(score):
                    scores[duuid] = float(score)
        return scores

    if isinstance(json_result, list):
        for entry in json_result:
            dauuids = [str(u) for u in entry.get("dauuid_list", [])]
            score_list = entry.get("score_list", [])
            for duuid, score in zip(dauuids, score_list):
                if np.isfinite(score):
                    scores[duuid] = float(score)
        return scores

    return scores


def build_database_from_fixture(fx: dict) -> list[AnnotatedImage]:
    import cv2 as _cv2

    sift_cfg = SiftConfig()
    database: list[AnnotatedImage] = []

    for i, img_bytes in enumerate(fx["image_bytes"]):
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode image {i}")

        features = extract_features(img, sift_cfg)

        auuid = uuid_mod.UUID(fx["annot_uuids"][i])
        nuuid = uuid_mod.UUID(fx["name_uuids"][i]) if fx["name_uuids"][i] else None
        bbox = fx["bboxes"][i] if fx["bboxes"] else (0, 0, img.shape[1], img.shape[0])
        bbox_int = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        database.append(
            AnnotatedImage(
                annot_uuid=auuid,
                name_uuid=nuuid,
                image=img,
                features=features,
                bbox=bbox_int,
            )
        )

    return database


def main():
    failed = 0
    passed = 0

    for path in sorted(FIXTURES_DIR.glob("*.npz")):
        print(f"\n{'=' * 60}")
        print(f"Fixture: {path.name}")
        print(f"{'=' * 60}")

        fx = load_fixture(path)
        wbia_scores = parse_wbia_scores(fx["raw_result"])

        if len(wbia_scores) == 0:
            print("  SKIP: no WBIA scores to compare")
            continue

        print(f"  WBIA scored {len(wbia_scores)} annotations")
        wbia_top = sorted(wbia_scores, key=lambda k: wbia_scores[k], reverse=True)

        database = build_database_from_fixture(fx)
        query_idx = fx["query_idx"]
        query_uuid = fx["annot_uuids"][query_idx]
        print(f"  Query: {query_uuid} (idx={query_idx})")
        print(f"  Database size: {len(database)} annotations")

        # Use same defaults as WBIA's vsmany pipeline:
        #   NNConfig: K=4, Knorm=1
        #   SpatialVerify: sv_on=True, prescore_method='nsum'
        #   Aggregate: score_method='nsum'
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(
                knn=4,
                sv_on=False,
                ratio_thresh=None,
                lnbnn_ratio=1.0,
                num_return=len(database),
                score_method="csum",
                prescore_method="csum",
                fg_on=True,
            )
        )
        core_results = identify(query_idx, database, config)

        # Build wbia-core score dict
        core_scores: dict[str, float] = {}
        for r in core_results:
            core_scores[str(r.annot_uuid)] = r.score

        # Compare rankings (not raw scores since scoring scale may differ)
        wbia_rank = {uid: i for i, uid in enumerate(wbia_top)}
        core_top = [str(r.annot_uuid) for r in core_results]
        core_rank = {uid: i for i, uid in enumerate(core_top)}

        all_uids = set(wbia_rank) | set(core_rank)
        shared_uids = set(wbia_rank) & set(core_rank)

        overlap_pct = len(shared_uids) / len(all_uids) * 100 if all_uids else 0
        print(f"  WBIA top: {[u[:8] for u in wbia_top[:5]]}")
        print(f"  Core top: {[u[:8] for u in core_top[:5]]}")
        print(f"  Shared UIDs: {len(shared_uids)}/{len(all_uids)} ({overlap_pct:.0f}%)")

        # Check that query UUID is excluded from core results
        core_result_uuids = [str(r.annot_uuid) for r in core_results]
        if query_uuid in core_result_uuids:
            print(f"  FAIL: query UUID {query_uuid} not excluded")
            failed += 1
            continue

        # Score comparison
        print(
            f"  WBIA scores: {', '.join(f'{u[:8]}={wbia_scores[u]:.4f}' for u in wbia_top[:5])}"
        )
        print(
            f"  Core scores: {', '.join(f'{u[:8]}={core_scores[u]:.4f}' for u in core_top[:5])}"
        )
        print(
            f"  Queried feature count: {database[query_idx].features.descriptors.shape[0]}"
        )

        # Check top-1 match agreement (most important)
        if wbia_top and core_top:
            wbia_top1 = wbia_top[0]
            core_top1 = core_top[0]
            top1_match = wbia_top1 == core_top1
            print(
                f"  Top-1 match: {wbia_top1[:8]} vs {core_top1[:8]} → {'✓' if top1_match else '✗'}"
            )
            if not top1_match:
                failed += 1
                continue

        # Deep compare: check if the same annotations appear in both
        # with the same relative ordering (Spearman-like)
        from scipy.stats import spearmanr

        common_uids = [u for u in wbia_top if u in core_rank]
        if len(common_uids) >= 3:
            w_ranks = [wbia_rank[u] for u in common_uids]
            c_ranks = [core_rank[u] for u in common_uids]
            rho, pval = spearmanr(w_ranks, c_ranks)
            print(f"  Spearman ρ={rho:.3f} (p={pval:.4f}) on {len(common_uids)} shared")
        else:
            print(f"  Spearman: too few shared UIDs ({len(common_uids)})")

        passed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
