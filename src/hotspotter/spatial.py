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
    sver_output_weighting: bool = False,
    dist_lookup: dict[tuple[int, int, int], float] | None = None,
) -> tuple[
    list[ScoredMatch],
    dict[uuid.UUID, tuple[list[int], list[int], np.ndarray]],
]:
    """Run vtool spatial verification on each candidate.

    Only candidates with enough correspondences are verified.  The
    homography is computed from the per-feature correspondences stored
    in ``ScoredMatch.correspondences`` as ``(qfx, dfx)`` pairs.

    Survival is gated solely by ``spatially_verify_kpts`` returning
    ``None`` (affine inliers < 7 for homography refinement, matching
    WBIA pipeline.py:1567).  There is no downstream inlier-count check.
    The fm-list passed to scoring is always filtered to
    homography-refined inliers (``svtup[0]``), matching WBIA
    pipeline.py:1568.

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
        from hotspotter._vendor.sver import _spatial_verification as sver
    except ImportError as ex:  # pragma: no cover
        raise ImportError("Spatial verification module not found.") from ex

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

        if dist_lookup is not None:
            ann_idx_val = ann_idx
            daid_idx = ann_idx_val
            weights = np.array(
                [
                    dist_lookup.get((daid_idx, int(qfx), int(dfx)), 0.0)
                    for qfx, dfx in fm
                ],
                dtype=np.float64,
            )
            sort_order = np.argsort(weights)[::-1]
            fm = fm[sort_order]

        dlen_sqrd2 = None
        if use_chip_extent:
            chip_h, chip_w = db_ann.image.shape[:2]
            dlen_sqrd2 = float(chip_w**2 + chip_h**2)
        elif len(fm) >= 2:
            matching_db_kpts = db_kp[fm[:, 1]]
            xs = matching_db_kpts[:, 0]
            ys = matching_db_kpts[:, 1]
            dlen_sqrd2 = float((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
            if dlen_sqrd2 <= 0:
                dlen_sqrd2 = None

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
            returnAff=True,
            refine_method="homog",
        )
        if svtup is None:
            continue

        refined_inliers, refined_errors, H, aff_inliers, aff_errors, Aff = svtup

        # WBIA always scores on homography-refined inliers (pipeline.py:1568).
        # The survival gate is enforced by sver's None return (affine < 7),
        # so no secondary length check is needed.
        sv_inliers = refined_inliers

        sm.sv_inliers = int(len(sv_inliers))
        sm.sv_homography = H

        inlier_fm = fm[sv_inliers]
        inlier_qfxs = inlier_fm[:, 0].tolist()
        inlier_dfxs = inlier_fm[:, 1].tolist()

        inliers = int(len(sv_inliers))

        if (
            sver_output_weighting
            and xy_thresh is not None
            and dlen_sqrd2 is not None
            and aff_inliers is not None
        ):
            homog_xy_errors = refined_errors[0]
            homog_inlier_set = set(refined_inliers)
            aff_to_homog_idx = {fx: hi for hi, fx in enumerate(refined_inliers)}
            xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
            if xy_thresh_sqrd > 0:
                weights = np.ones(inliers, dtype=np.float64)
                for i, fx in enumerate(sv_inliers):
                    if fx in homog_inlier_set:
                        h_idx = aff_to_homog_idx[fx]
                        err = homog_xy_errors[h_idx]
                        w = 1.0 - np.sqrt(np.clip(err / xy_thresh_sqrd, 0.0, 1.0))
                        weights[i] = w
                homog_err_weight = weights
            else:
                homog_err_weight = np.ones(inliers, dtype=np.float64)
        elif sver_output_weighting and xy_thresh is not None and dlen_sqrd2 is not None:
            homog_xy_errors = refined_errors[0].take(sv_inliers, axis=0)
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
    score_method: str = "nsum",
) -> list[ScoredMatch]:
    """Select a shortlist of candidates for spatial verification.

    Matches WBIA's ``make_chipmatch_shortlists`` → ``get_name_shortlist_aids``:

    * ``nsum`` (default): group by name, rank **names** by their name score
      (the canonical annotation's ``score``) and take the top *n_names*;
      within each name rank annotations by their **annot score (csum)** and
      take the top *n_annots_per_name* (``scoring.py:107-109``).
    * ``csum``: flat top-``n_names * n_annots_per_name`` annotations ranked
      by annot score (csum), matching ``get_annot_shortlist_aids``.

    Annotations with ``name_uuid is None`` are always carried as candidates
    when the flat path is taken; under the name path they are skipped
    (WBIA groups by nid, unnamed annots form negative-nid singleton groups).
    """
    if not scored:
        return []

    is_name_path = score_method in {"nsum", "nsum_wbia"}

    if not is_name_path:
        # Flat annot shortlist ranked by csum (WBIA get_annot_shortlist_aids).
        flat = sorted(scored, key=lambda s: s.annot_csum, reverse=True)
        return flat[: n_names * n_annots_per_name]

    name_best: dict[uuid.UUID, float] = {}
    name_annots: dict[uuid.UUID, list[ScoredMatch]] = {}
    for sm in scored:
        nid = sm.name_uuid
        if nid is None:
            continue
        name_annots.setdefault(nid, []).append(sm)
        # Names are ranked by name score; the canonical annotation carries
        # it as ``sm.score`` (max over the name == the name's fmech score).
        cur = name_best.get(nid, float("-inf"))
        if sm.score > cur:
            name_best[nid] = sm.score

    sorted_names = sorted(name_best, key=lambda n: name_best[n], reverse=True)

    shortlist: list[ScoredMatch] = []
    for nid in sorted_names[:n_names]:
        # Within a name, rank by the annotation score (csum), not canonical.
        annots = sorted(name_annots[nid], key=lambda s: s.annot_csum, reverse=True)
        shortlist.extend(annots[:n_annots_per_name])

    return shortlist
