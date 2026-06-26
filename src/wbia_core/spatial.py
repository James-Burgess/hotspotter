"""Spatial verification via RANSAC homography (OpenCV).

Uses exact per-feature correspondences threaded through from the
scoring stage (``ScoredMatch.correspondences``) to build the
keypoint pairs for ``cv2.findHomography``.

After SV, per-inlier homography-error weights are computed following
WBIA's ``_spatial_verification`` (pipeline.py:1580-1592):

    homog_err_weight = 1.0 - sqrt(homog_xy_errors / xy_thresh_sqrd)

where ``homog_xy_errors`` are squared reprojection errors and
``xy_thresh_sqrd = dlen_sqrd * xy_thresh``.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np

from wbia_core.data import AnnotatedImage, FeatureSet, ScoredMatch


def _compute_dlen_sqrd(kpts: np.ndarray, image: np.ndarray) -> float:
    """Squared diagonal length of the keypoint extent.

    Mirrors WBIA's ``vt.get_kpts_dlen_sqrd``.  Falls back to the
    full image diagonal when there are too few keypoints.
    """
    if kpts.shape[0] >= 2:
        xs = kpts[:, 0]
        ys = kpts[:, 1]
        dlen_sqrd = float((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
        if dlen_sqrd > 0:
            return dlen_sqrd
    return float(image.shape[0] ** 2 + image.shape[1] ** 2)


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
    sver_output_weighting: bool = False,
) -> tuple[
    list[ScoredMatch],
    dict[uuid.UUID, tuple[list[int], list[int], np.ndarray]],
]:
    """Run spatial verification (RANSAC homography) on each candidate.

    Only candidates with enough correspondences are verified.  The
    homography is computed from the per-feature correspondences stored
    in ``ScoredMatch.correspondences`` as ``(qfx, dfx)`` pairs.

    Unlike the old boost-based approach, this function does NOT modify
    ``ScoredMatch.score``.  Instead it returns per-inlier homography-
    error weights so the caller can filter matches to inliers, apply
    the weights as a new fsv column, and re-run the scoring chain.

    Args:
        matches: scored candidates (shortlist).
        query_features: query image feature set.
        database: annotations in index order.
        ransac_thresh: RANSAC reprojection threshold (pixels).
        min_inliers: minimum inliers to accept homography.
        xy_thresh: max spatial displacement ratio (WBIA default 0.01).
        scale_thresh: max scale ratio (WBIA default 2.0).
        ori_thresh: max orientation delta in radians (WBIA default TAU/4).
        use_chip_extent: scale threshold by chip size (WBIA default True).
        weight_inliers: compute per-inlier homog-error weights.

    Returns:
        ``(matches, sv_results)`` where *matches* has ``sv_inliers`` and
        ``sv_homography`` populated for passing candidates, and
        *sv_results* maps ``annot_uuid`` to
        ``(inlier_qfxs, inlier_dfxs, homog_err_weights)``.
    """
    q_kp = query_features.keypoints
    sv_results: dict[uuid.UUID, tuple[list[int], list[int], np.ndarray]] = {}

    for sm in matches:
        if len(sm.correspondences) < min_inliers:
            continue

        ann_idx = next(
            i for i, a in enumerate(database) if a.annot_uuid == sm.annot_uuid
        )
        db_ann = database[ann_idx]
        db_kp = db_ann.features.keypoints

        q_pts_list: list[tuple[float, float]] = []
        db_pts_list: list[tuple[float, float]] = []
        corr_qfxs: list[int] = []
        corr_dfxs: list[int] = []

        for qfx, dfx in sm.correspondences:
            if qfx >= q_kp.shape[0] or dfx >= db_kp.shape[0]:
                continue
            qk = q_kp[qfx]
            dk = db_kp[dfx]
            if xy_thresh is not None:
                dx = abs(float(qk[0]) - float(dk[0]))
                dy = abs(float(qk[1]) - float(dk[1]))
                max_dim = (
                    max(db_ann.image.shape[0], db_ann.image.shape[1])
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
            q_pts_list.append((float(qk[0]), float(qk[1])))
            db_pts_list.append((float(dk[0]), float(dk[1])))
            corr_qfxs.append(qfx)
            corr_dfxs.append(dfx)

        if len(q_pts_list) < 4:
            continue

        q_pts_arr = np.array(q_pts_list, dtype=np.float32)
        db_pts_arr = np.array(db_pts_list, dtype=np.float32)

        H, mask = cv2.findHomography(q_pts_arr, db_pts_arr, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            continue

        inlier_flags = mask.ravel().astype(bool)
        inlier_indices = np.where(inlier_flags)[0]
        if len(inlier_indices) < min_inliers:
            continue

        sm.sv_inliers = len(inlier_indices)
        sm.sv_homography = H

        inlier_qfxs = [corr_qfxs[i] for i in inlier_indices]
        inlier_dfxs = [corr_dfxs[i] for i in inlier_indices]

        inlier_q_pts = q_pts_arr[inlier_indices].reshape(-1, 1, 2)
        reprojected = cv2.perspectiveTransform(inlier_q_pts, H).squeeze(axis=1)
        homog_xy_errors = np.sum(
            (reprojected - db_pts_arr[inlier_indices]) ** 2, axis=1
        )

        if sver_output_weighting and xy_thresh is not None:
            dlen_sqrd = _compute_dlen_sqrd(db_kp, db_ann.image)
            xy_thresh_sqrd = dlen_sqrd * xy_thresh
            if xy_thresh_sqrd > 0:
                ratio = np.clip(homog_xy_errors / xy_thresh_sqrd, 0.0, 1.0)
                homog_err_weight = 1.0 - np.sqrt(ratio)
            else:
                homog_err_weight = np.ones(len(inlier_indices), dtype=np.float64)
        else:
            homog_err_weight = np.ones(len(inlier_indices), dtype=np.float64)

        sv_results[sm.annot_uuid] = (
            inlier_qfxs,
            inlier_dfxs,
            homog_err_weight.astype(np.float64),
        )

    return matches, sv_results


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
