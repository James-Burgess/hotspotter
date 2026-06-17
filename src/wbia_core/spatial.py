"""Spatial verification via RANSAC homography (OpenCV).

Uses exact per-feature correspondences threaded through from the
scoring stage (``ScoredMatch.correspondences``) to build the
keypoint pairs for ``cv2.findHomography``.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np

from wbia_core.data import AnnotatedImage, FeatureSet, ScoredMatch


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
    """Run spatial verification (RANSAC homography) on each candidate.

    Only candidates with ``num_matches >= min_inliers`` are verified.
    The homography is computed from the exact per-feature correspondences
    stored in ``ScoredMatch.correspondences`` as ``(qfx, dfx)`` pairs.

    Args:
        matches: scored candidates.
        query_features: query image feature set.
        database: annotations in index order.
        ransac_thresh: RANSAC reprojection threshold (pixels).
        min_inliers: minimum inliers to accept homography.
        xy_thresh: max spatial displacement (WBIA default 0.01).
        scale_thresh: max scale ratio (WBIA default 2.0).
        ori_thresh: max orientation delta in radians (WBIA default TAU/4).
        use_chip_extent: scale threshold by chip size (WBIA default True).
        weight_inliers: boost score by inlier ratio (WBIA default True).

    Returns:
        Updated list with ``sv_inliers`` and ``sv_homography`` populated.
    """
    q_kp = query_features.keypoints

    for sm in matches:
        if len(sm.correspondences) < min_inliers:
            continue

        ann_idx = next(
            i for i, a in enumerate(database) if a.annot_uuid == sm.annot_uuid
        )
        db_kp = database[ann_idx].features.keypoints

        q_pts: list[tuple[float, float]] = []
        db_pts: list[tuple[float, float]] = []

        for qfx, dfx in sm.correspondences:
            if qfx >= q_kp.shape[0] or dfx >= db_kp.shape[0]:
                continue
            if (
                xy_thresh is not None
                or scale_thresh is not None
                or ori_thresh is not None
            ):
                qk = q_kp[qfx]
                dk = db_kp[dfx]
                if xy_thresh is not None:
                    dx = abs(float(qk[0]) - float(dk[0]))
                    dy = abs(float(qk[1]) - float(dk[1]))
                    max_dim = (
                        max(
                            database[ann_idx].image.shape[0],
                            database[ann_idx].image.shape[1],
                        )
                        if use_chip_extent
                        else 1.0
                    )
                    if dx / max_dim > xy_thresh or dy / max_dim > xy_thresh:
                        continue
                if scale_thresh is not None:
                    q_scale = float(qk[2])
                    d_scale = float(dk[2])
                    if q_scale <= 0 or d_scale <= 0:
                        continue
                    ratio = max(q_scale, d_scale) / min(q_scale, d_scale)
                    if ratio > scale_thresh:
                        continue
                if ori_thresh is not None:
                    import math

                    q_ori = float(qk[5])
                    d_ori = float(dk[5])
                    diff = abs(q_ori - d_ori)
                    if diff > math.pi:
                        diff = 2.0 * math.pi - diff
                    if diff > ori_thresh:
                        continue
                q_pts.append((float(qk[0]), float(qk[1])))
                db_pts.append((float(dk[0]), float(dk[1])))
            else:
                q_pts.append((float(q_kp[qfx, 0]), float(q_kp[qfx, 1])))
                db_pts.append((float(db_kp[dfx, 0]), float(db_kp[dfx, 1])))

        if len(q_pts) < 4:
            continue

        q_pts_arr = np.array(q_pts, dtype=np.float32)
        db_pts_arr = np.array(db_pts, dtype=np.float32)

        H, mask = cv2.findHomography(q_pts_arr, db_pts_arr, cv2.RANSAC, ransac_thresh)
        if H is not None and mask is not None:
            inliers = int(mask.sum())
            if inliers >= min_inliers:
                sm.sv_inliers = inliers
                sm.sv_homography = H
                if weight_inliers:
                    sm.score = sm.score * (
                        1.0 + 0.5 * inliers / len(sm.correspondences)
                    )

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
