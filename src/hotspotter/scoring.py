"""LNBNN scoring — baseline filter, weight computation, match building.

Single source of truth for the scoring stages that pipeline.identify()
orchestrates.  All algorithmic logic lives here; pipeline.py is a thin
caller that wires config + trace hooks around these functions.

WBIA reference: ``wbia/algo/hots/nn_weights.py`` + ``pipeline.py``
stages ``baseline_neighbor_filter`` → ``weight_neighbors`` →
``fg_weight`` → ``build_chipmatches``.
"""

from __future__ import annotations

import uuid

import numpy as np

from hotspotter.data import AnnotatedImage, Match


def per_feature_fg(ann: AnnotatedImage) -> np.ndarray:
    """Per-keypoint foreground weight for one annotation.

    Returns constant 1.0 for all keypoints — matching WBIA's
    ``empty_probchips`` fallback when no CNN/RF detector is available.
    """
    return np.ones(len(ann.features.keypoints), dtype=np.float64)


def _compute_fg_for_database(database: list[AnnotatedImage]) -> list[np.ndarray]:
    return [per_feature_fg(ann) for ann in database]


def baseline_filter(
    voting_annot: np.ndarray,
    database: list[AnnotatedImage],
    query_annot_index: int,
    can_match_samename: bool = True,
    can_match_sameimg: bool = False,
) -> tuple[np.ndarray, set[int], uuid.UUID | None]:
    """Mark voting columns that are self / same-name / same-image.

    Mirrors WBIA's ``baseline_neighbor_filter`` + ``build_impossible_daids_list``.

    Args:
        voting_annot: [N, K+Kpad] int32 — annotation index per voting column.
        database: annotations in index order.
        query_annot_index: query position in *database*.
        can_match_samename: when False, same-name annotations are impossible.
        can_match_sameimg: when False, same-image annotations are impossible.

    Returns:
        ``(invalid_mask, impossible_set, query_name_uuid)`` where
        *invalid_mask* is [N, K+Kpad] bool and *impossible_set* is the
        set of database indices that are impossible matches.
    """
    qann = database[query_annot_index]
    qname = qann.name_uuid
    qimg = qann.image_uuid

    same_name_set: set[int] = set()
    if not can_match_samename and qname is not None:
        same_name_set = {
            i
            for i, a in enumerate(database)
            if i != query_annot_index and a.name_uuid == qname
        }

    same_image_set: set[int] = set()
    if not can_match_sameimg and qimg is not None:
        same_image_set = {
            i
            for i, a in enumerate(database)
            if i != query_annot_index and a.image_uuid == qimg
        }

    impossible_set = same_name_set | same_image_set

    invalid = (voting_annot == query_annot_index) | np.isin(
        voting_annot, list(impossible_set) if impossible_set else []
    )
    return invalid, impossible_set, qname


def compute_normalizer_validity(
    voting_annot: np.ndarray,
    labels: np.ndarray,
    annot_of_desc: np.ndarray,
    n_total: int,
    database: list[AnnotatedImage],
    k: int,
    kpad: int,
    qname: uuid.UUID | None,
) -> np.ndarray:
    """Name-based normalizer validity check.

    A query feature's normalizer (last KNN column) is *invalid* if its
    annotation shares a name with any of that feature's voting annotations,
    or with the query name itself.  Mirrors WBIA's ``normalizer_rule='name'``.

    Returns:
        [N] bool array — True where the normalizer is valid.
    """
    n_qfxs = labels.shape[0]
    norm_col_labels = labels[:, -1]
    norm_ok = (norm_col_labels >= 0) & (norm_col_labels < n_total)
    norm_annots = np.full(n_qfxs, -1, dtype=np.int32)
    norm_annots[norm_ok] = annot_of_desc[norm_col_labels[norm_ok]]

    db_names = np.array([a.name_uuid for a in database], dtype=object)
    norm_names = np.full(n_qfxs, None, dtype=object)
    norm_names[norm_ok] = db_names[norm_annots[norm_ok]]

    voting_names = np.full_like(voting_annot, None, dtype=object)
    voting_ok = voting_annot >= 0
    voting_names[voting_ok] = db_names[voting_annot[voting_ok]]

    normalizer_valid = np.ones(n_qfxs, dtype=bool)
    for j in range(k + kpad):
        vn = voting_names[:, j]
        has_name = (norm_names != None) & (vn != None)  # noqa: E711
        conflict = np.zeros(n_qfxs, dtype=bool)
        conflict[has_name] = norm_names[has_name] == vn[has_name]
        normalizer_valid &= ~conflict

    if qname is not None:
        has_qn = norm_names != None  # noqa: E711
        qname_conflict = np.zeros(n_qfxs, dtype=bool)
        qname_conflict[has_qn] = norm_names[has_qn] == qname
        normalizer_valid &= ~qname_conflict

    normalizer_valid &= norm_ok
    return normalizer_valid


def weight_neighbors_lnbnn(
    voting_dists: np.ndarray,
    norm_dists: np.ndarray,
    normonly_on: bool = False,
    bar_l2_on: bool = False,
    ratio_thresh: float | None = None,
    lnbnn_ratio: float = 1.0,
    cos_on: bool = False,
    lograt_on: bool = False,
    const_on: bool = False,
    normk: np.ndarray | None = None,
) -> np.ndarray:
    """LNBNN weight computation with optional filter chain.

    Core formula (WBIA ``nn_weights.py``)::

        w = max(0, norm_dist - nn_dist)

    The normalizer distance is selected per query feature — by default
    the first normalizer column (index 0), but when *normk* is provided
    each row selects its own column via ``norm_dists[row, normk[row]]``,
    mirroring WBIA's ``vt.take_col_per_row(neighb_dist, neighb_normk)``.

    Optional modes applied in WBIA order:
    - ``cos_on``: convert L2 distances to soft cosine similarity via
      ``1/(1+d)`` before LNBNN (approximates cosine distance for
      normalized feature-space distances).
    - ``normonly_on``: replace nn_dist with norm_dist (debug).
    - ``bar_l2_on``: ``w *= 1.0 - nn_dist``.
    - ``ratio_thresh``: skip (zero) where ``nn_dist / norm_dist > thresh``,
      multiply surviving by ``1.0 - ratio``.
    - ``lnbnn_ratio``: zero where ``nn_dist > norm_dist * ratio``.
    - ``lograt_on``: apply ``log(w + 1)`` transform (WBIA ``loglnbnn_fn``).
    - ``const_on``: replace all non-zero weights with uniform 1.0
      (WBIA ``const_match_weighter``).

    Args:
        voting_dists: [N, K+Kpad] float64 — normalized distances.
        norm_dists: [N, Knorm] float64 — normalizer column(s).
        normonly_on: use norm_dist for both operands.
        bar_l2_on: apply bar-L2 multiplicative penalty.
        ratio_thresh: Lowe's ratio test threshold.
        lnbnn_ratio: LNBNN ratio clamp.
        cos_on: apply cosine-similarity transform to distances.
        lograt_on: apply log transform to weights.
        const_on: replace all non-zero weights with 1.0.
        normk: [N] int — per-feature normalizer column index.  When
            None, ``norm_dists[:, 0]`` is used for all rows.

    Returns:
        [N, K+Kpad] float64 weight matrix.
    """
    if normk is not None and len(normk) > 0:
        ndist = norm_dists[np.arange(len(normk)), normk.astype(int)][:, None]
    else:
        ndist = norm_dists[:, 0:1]
    if normonly_on:
        vdist = np.tile(ndist, (1, voting_dists.shape[1]))
    else:
        vdist = voting_dists

    if cos_on:
        weights = np.maximum(0.0, 1.0 - vdist)
        weights[vdist >= ndist] = 0.0
    else:
        weights = np.maximum(0.0, ndist - vdist)

    if lnbnn_ratio < 1.0:
        weights[vdist > ndist * lnbnn_ratio] = 0.0

    if bar_l2_on:
        weights *= np.maximum(0.0, 1.0 - vdist)

    if ratio_thresh is not None:
        ratio = np.divide(vdist, ndist, out=np.ones_like(vdist), where=ndist > 0)
        weights = np.where(ratio > ratio_thresh, 0.0, weights * (1.0 - ratio))

    if lograt_on:
        weights = np.log(weights + 1.0)

    if const_on:
        weights = np.where(weights > 0, 1.0, 0.0)

    return weights.astype(np.float64)


def apply_fg_weights(
    weights: np.ndarray,
    voting_annot: np.ndarray,
    voting_feat: np.ndarray,
    database: list[AnnotatedImage],
    query_annot_index: int,
    fg_weights: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Multiply weights by foreground confidence: ``w *= sqrt(q_fg * db_fg)``.

    Args:
        weights: [N, K+Kpad] float64.
        voting_annot: [N, K+Kpad] int32 — annotation indices.
        voting_feat: [N, K+Kpad] int32 — feature indices within annotation.
        database: annotations in index order.
        query_annot_index: query index in *database*.
        fg_weights: pre-computed per-annotation fg arrays (auto-computed if None).

    Returns:
        [N, K+Kpad] float64 — weighted copy.
    """
    if fg_weights is None:
        fg_weights = _compute_fg_for_database(database)

    q_fg = fg_weights[query_annot_index]
    n_qfxs, n_cols = voting_annot.shape
    result = weights.copy()

    for j in range(n_cols):
        db_idxs = voting_annot[:, j]
        feat_idxs = voting_feat[:, j]
        valid = (db_idxs >= 0) & (feat_idxs >= 0)
        if not valid.any():
            continue
        for qfx in np.where(valid)[0]:
            db_idx = db_idxs[qfx]
            dfx = feat_idxs[qfx]
            if db_idx >= len(fg_weights) or dfx >= len(fg_weights[db_idx]):
                continue
            fg = np.sqrt(q_fg[qfx] * fg_weights[db_idx][dfx])
            result[qfx, j] *= fg

    return result


def build_matches(
    weights: np.ndarray,
    voting_annot: np.ndarray,
    voting_feat: np.ndarray,
    invalid: np.ndarray,
    database: list[AnnotatedImage],
    k: int,
    kpad: int,
    normalizer_valid: np.ndarray | None = None,
    normks: np.ndarray | None = None,
) -> list[Match]:
    """Convert weighted vote columns into a flat list of Match objects.

    All K+Kpad voting columns are processed (no self-match column exists
    because the query is excluded from the index).  A match is accepted
    when all of the following hold::

        - db_idx ≥ 0 (valid annotation)
        - ``invalid[qfx, j]`` is False (baseline filter)
        - ``weights[qfx, j] > 0`` (positive LNBNN weight)
        - ``normalizer_valid[qfx]`` (name-rule check, if provided)
        - ``normks[qfx]`` (per-feature normalizer index finite, if provided)

    This mirrors WBIA's ``build_chipmatches`` which applies
    ``neighb_valid_agg = logical_and(neighb_valid0, filtvalids, filtnormks)``.

    Args:
        weights: [N, K+Kpad] float64 from ``weight_neighbors_lnbnn``.
        voting_annot: [N, K+Kpad] int32 — annotation indices.
        voting_feat: [N, K+Kpad] int32 — feature indices.
        invalid: [N, K+Kpad] bool — True where match is self/same-name/same-image.
        database: annotations in index order.
        k: number of KNN neighbours.
        kpad: padding columns.
        normalizer_valid: [N] bool — name-rule normalizer mask.
        normks: [N] bool — per-feature normalizer index finite
            (``True`` = valid normalizer found for this feature).

    Returns:
        Flat list of Match objects (one per surviving feature correspondence).
    """
    n_qfxs = weights.shape[0]
    matches: list[Match] = []
    for qfx in range(n_qfxs):
        if normalizer_valid is not None and not normalizer_valid[qfx]:
            continue
        if normks is not None and not normks[qfx]:
            continue
        for j in range(k + kpad):
            db_idx = int(voting_annot[qfx, j])
            if db_idx < 0 or invalid[qfx, j]:
                continue
            dfx = int(voting_feat[qfx, j])
            if dfx < 0:
                continue
            w = float(weights[qfx, j])
            if w <= 0:
                continue
            matches.append(
                Match(
                    qfx=qfx,
                    daid=db_idx,
                    dfx=dfx,
                    dist=w,
                    name_uuid=database[db_idx].name_uuid,
                )
            )
    return matches


def score_matches(
    matches: list[Match],
    database: list[AnnotatedImage],
    score_method: str = "nsum",
) -> list:
    """Aggregate per-feature matches into per-annotation scores.

    ``nsum`` (normalized sum)::
        score = sum(weights) / count(weights)
    ``csum`` (cumulative sum)::
        score = sum(weights)

    Returns:
        List of :class:`ScoredMatch`, sorted by descending score.
    """
    from hotspotter.data import ScoredMatch

    agg: dict[uuid.UUID, tuple[float, int, list[tuple[int, int]]]] = {}

    for m in matches:
        annot_uuid = database[m.daid].annot_uuid
        cur_score, cur_count, cur_corr = agg.get(annot_uuid, (0.0, 0, []))
        cur_corr.append((m.qfx, m.dfx))
        agg[annot_uuid] = (cur_score + m.dist, cur_count + 1, cur_corr)

    results: list[ScoredMatch] = []
    for annot_uuid, (total_weight, count, corrs) in agg.items():
        if score_method == "nsum":
            score = total_weight / count if count else 0.0
        elif score_method == "csum":
            score = total_weight
        else:
            raise ValueError(f"Unknown score_method: {score_method!r}")

        annot = database[[a.annot_uuid for a in database].index(annot_uuid)]
        results.append(
            ScoredMatch(
                annot_uuid=annot_uuid,
                name_uuid=annot.name_uuid,
                score=float(score),
                num_matches=count,
                correspondences=corrs,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
