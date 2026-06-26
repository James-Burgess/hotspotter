"""Spatial verification using vtool's WBIA spatial verifier.

Uses exact per-feature correspondences threaded through from the
scoring stage (``ScoredMatch.correspondences``) to build the
feature-match arrays passed to ``vtool.spatial_verification``.

After SV, per-inlier homography-error weights are computed following
WBIA's ``_spatial_verification`` (pipeline.py:1580-1592):

    homog_err_weight = 1.0 - sqrt(homog_xy_errors / xy_thresh_sqrd)

where ``homog_xy_errors`` are squared reprojection errors (from vtool's
``L2_sqrd``) and ``xy_thresh_sqrd = dlen_sqrd * xy_thresh``.
"""

from __future__ import annotations

import importlib
import uuid

import numpy as np

from hotspotter.data import AnnotatedImage, FeatureSet, ScoredMatch


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
    """Run vtool spatial verification on each candidate.

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
        ransac_thresh: Deprecated OpenCV threshold retained for API compatibility.
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
    _ = ransac_thresh
    try:
        sver = importlib.import_module("vtool.spatial_verification")
    except ImportError as ex:  # pragma: no cover
        raise ImportError(
            "vtool is required for spatial verification. Install the vendored "
            "wbia-vtool package or run inside the canonical Docker image."
        ) from ex

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
        if use_chip_extent:
            matching_db_kpts = db_kp[fm[:, 1]]
            dlen_sqrd2 = _compute_dlen_sqrd(matching_db_kpts, db_ann.image)

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

        refined_inliers, refined_errors, H = svtup[0:3]
        inliers = int(len(refined_inliers))
        if inliers < min_inliers:
            continue

        sm.sv_inliers = inliers
        sm.sv_homography = H

        inlier_fm = fm[refined_inliers]
        inlier_qfxs = inlier_fm[:, 0].tolist()
        inlier_dfxs = inlier_fm[:, 1].tolist()

        if sver_output_weighting and xy_thresh is not None and dlen_sqrd2 is not None:
            homog_xy_errors = refined_errors[0].take(refined_inliers, axis=0)
            xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
            if xy_thresh_sqrd > 0:
                ratio = np.clip(homog_xy_errors / xy_thresh_sqrd, 0.0, 1.0)
                homog_err_weight = 1.0 - np.sqrt(ratio)
            else:
                homog_err_weight = np.ones(inliers, dtype=np.float64)
        else:
            homog_err_weight = np.ones(inliers, dtype=np.float64)

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
