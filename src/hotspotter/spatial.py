"""Spatial verification using vtool's WBIA spatial verifier.

Uses exact per-feature correspondences threaded through from the
scoring stage (``ScoredMatch.correspondences``) to build the
feature-match arrays passed to ``vtool.spatial_verification``.
"""

from __future__ import annotations

import importlib
import uuid

import numpy as np

from hotspotter.data import AnnotatedImage, FeatureSet, ScoredMatch


def spatial_verify(
    matches: list[ScoredMatch],
    query_features: FeatureSet,
    database: list[AnnotatedImage],
    ransac_thresh: float = 3.0,
    min_inliers: int = 4,
    xy_thresh: float | None = None,
    scale_thresh: float | None = None,
    ori_thresh: float | None = None,
    use_chip_extent: bool = True,
    weight_inliers: bool = True,
) -> list[ScoredMatch]:
    """Run vtool spatial verification on each candidate.

    Only candidates with ``num_matches >= min_inliers`` are verified.
    The homography is computed from the exact per-feature correspondences
    stored in ``ScoredMatch.correspondences`` as ``(qfx, dfx)`` pairs.

    Args:
        matches: scored candidates.
        query_features: query image feature set.
        database: annotations in index order.
        ransac_thresh: Deprecated OpenCV threshold retained for API compatibility.
        min_inliers: minimum inliers to accept homography.
        xy_thresh: max spatial displacement (WBIA default 0.01).
        scale_thresh: max scale ratio (WBIA default 2.0).
        ori_thresh: max orientation delta in radians (WBIA default TAU/4).
        use_chip_extent: scale threshold by chip size (WBIA default True).
        weight_inliers: boost score by inlier ratio (WBIA default True).

    Returns:
        Updated list with ``sv_inliers`` and ``sv_homography`` populated.
    """
    _ = ransac_thresh
    try:
        sver = importlib.import_module("vtool.spatial_verification")
    except ImportError as ex:  # pragma: no cover - exercised only outside image
        raise ImportError(
            "vtool is required for spatial verification. Install the vendored "
            "wbia-vtool package or run inside the canonical Docker image."
        ) from ex

    q_kp = query_features.keypoints

    for sm in matches:
        if len(sm.correspondences) < min_inliers:
            continue

        ann_idx = next(
            i for i, a in enumerate(database) if a.annot_uuid == sm.annot_uuid
        )
        db_kp = database[ann_idx].features.keypoints

        fm = np.array(
            [
                (qfx, dfx)
                for qfx, dfx in sm.correspondences
                if qfx < q_kp.shape[0] and dfx < db_kp.shape[0]
            ],
            dtype=np.int32,
        )
        if len(fm) < min_inliers:
            continue

        dlen_sqrd2 = None
        if use_chip_extent and database[ann_idx].image is not None:
            height, width = database[ann_idx].image.shape[:2]
            dlen_sqrd2 = float(width * width + height * height)

        match_weights = np.ones(len(fm), dtype=np.float64)
        svtup = sver.spatially_verify_kpts(
            q_kp,
            db_kp,
            fm,
            xy_thresh=xy_thresh if xy_thresh is not None else 0.01,
            scale_thresh=scale_thresh if scale_thresh is not None else 2.0,
            ori_thresh=ori_thresh if ori_thresh is not None else np.pi / 2.0,
            dlen_sqrd2=dlen_sqrd2,
            min_nInliers=min_inliers,
            match_weights=match_weights,
            returnAff=False,
            refine_method="homog",
        )
        if svtup is None:
            continue

        refined_inliers, _refined_errors, H = svtup[0:3]
        inliers = int(len(refined_inliers))
        if inliers >= min_inliers:
            sm.sv_inliers = inliers
            sm.sv_homography = H
            if weight_inliers:
                sm.score = sm.score * (1.0 + 0.5 * inliers / len(fm))

    return matches


def make_sver_shortlist(
    scored: list[ScoredMatch],
    n_names: int = 40,
    n_annots_per_name: int = 3,
) -> list[ScoredMatch]:
    """Select a shortlist of candidates for spatial verification.

    Matches WBIA's ``make_chipmatch_shortlists``:
    1. Group candidates by ``name_uuid``.
    2. Sort names by their best annotation score descending.
    3. Keep top *n_names*.
    4. Within each name, keep top *n_annots_per_name*.
    """
    if not scored:
        return []

    name_best: dict[uuid.UUID, float] = {}
    name_annots: dict[uuid.UUID, list[ScoredMatch]] = {}
    for sm in scored:
        nid = sm.name_uuid
        if nid is None:
            continue
        name_annots.setdefault(nid, []).append(sm)
        cur = name_best.get(nid, float("-inf"))
        if sm.score > cur:
            name_best[nid] = sm.score

    sorted_names = sorted(name_best, key=lambda n: name_best[n], reverse=True)

    shortlist: list[ScoredMatch] = []
    for nid in sorted_names[:n_names]:
        annots = sorted(name_annots[nid], key=lambda s: s.score, reverse=True)
        shortlist.extend(annots[:n_annots_per_name])

    return shortlist
