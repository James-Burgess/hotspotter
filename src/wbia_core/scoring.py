"""LNBNN scoring and name aggregation.

Mirrors the stages in ``wildbook-ia/wbia/algo/hots/pipeline.py``:

1. ``baseline_neighbor_filter`` → :func:`filter_self_matches`
2. ``weight_neighbors`` → :func:`weight_neighbors_lnbnn`
3. ``fg_weight`` → :func:`apply_fg_weights`
4. ``build_chipmatches`` → :func:`build_matches`
5. ``score_chipmatch_list`` → :func:`score_matches`
"""

from __future__ import annotations

import uuid

import numpy as np

from wbia_core.data import AnnotatedImage, FeatureSet, Match, ScoredMatch


def per_feature_fg(ann: AnnotatedImage) -> np.ndarray:
    """Per-keypoint foreground weight for one annotation.

    Returns constant 1.0 for all keypoints — matching WBIA's
    ``empty_probchips`` fallback when no CNN/RF detector is available.
    When ``fg_on=False`` the result is identical regardless.
    """
    return np.ones(len(ann.features.keypoints), dtype=np.float64)


def filter_self_matches(
    distances: np.ndarray,
    labels: np.ndarray,
    database: list[AnnotatedImage],
    query_annot_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove neighbours that are the query itself or share its name.

    Uses vectorised lookup (no ``np.vectorize``).  Filtered entries are
    set to ``inf`` / ``-1`` but columns are NOT re-sorted — the caller
    must handle the normalizer column separately (see ``pipeline.py``).

    Args:
        distances: [N, K] float32 from faiss.
        labels: [N, K] int64 — annotation indices.
        database: annotations in index order.
        query_annot_index: index of the query annotation.

    Returns:
        Filtered (distances, labels) — same shapes.
    """
    distances = distances.copy()
    labels = labels.copy()
    query_annot = database[query_annot_index]
    query_name = query_annot.name_uuid

    is_self = labels == query_annot_index

    if query_name is not None:
        safe_labels = np.maximum(labels, 0)
        valid = labels < len(database)
        name_uuids = np.array([a.name_uuid for a in database], dtype=object)
        is_same_name = name_uuids == query_name
        same_name_mask = np.zeros_like(labels, dtype=bool)
        if valid.any():
            same_name_mask[valid] = is_same_name[safe_labels[valid]]
        mask = is_self | same_name_mask
    else:
        mask = is_self

    distances[mask] = np.inf
    labels[mask] = np.int64(-1)
    return distances, labels


def weight_neighbors_lnbnn(
    distances: np.ndarray,
    labels: np.ndarray,
    k: int,
    lnbnn_ratio: float = 1.0,
) -> np.ndarray:
    """Apply LNBNN (Local Naive Bayes Nearest Neighbour) weighting.

    WBIA formula (raw distance difference, as used in ``nn_weights.py``)::

        score[i, j] = max(0, norm_dist[i] - nn_dist[i, j])

    where *norm_dist* is the distance to the K-th neighbour
    (the normalizer, column index ``k`` in the input).

    Args:
        distances: [N, K+1] float32 — column ``k`` is the normalizer.
        labels: [N, K+1] int64 (unused in the formula).
        k: number of neighbours to score.
        lnbnn_ratio: scores where ``dist > norm * ratio`` are zeroed.

    Returns:
        [N, K] float64 weight matrix.
    """
    nn_dists = distances[:, :k]
    norm_dists = distances[:, k : k + 1]

    weights = np.maximum(0.0, norm_dists - nn_dists)

    if lnbnn_ratio < 1.0:
        weights[nn_dists > norm_dists * lnbnn_ratio] = 0.0

    return weights.astype(np.float64)


def _compute_fg_for_database(database: list[AnnotatedImage]) -> list[np.ndarray]:
    """Pre-compute per-feature foreground weights for all annotations."""
    return [per_feature_fg(ann) for ann in database]


def apply_fg_weights(
    lnbnn_weights: np.ndarray,
    labels: np.ndarray,
    local_labels: np.ndarray,
    database: list[AnnotatedImage],
    query_annot_index: int,
    fg_weights: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Multiply LNBNN weights by the feature-grouping (``fg``) column.

    Uses WBIA's formula::

        w = lnbnn_weight * sqrt(q_fg * db_fg)

    ``q_fg`` and ``db_fg`` are per-keypoint foreground weights sampled
    from each annotation's probability heatmap.

    Args:
        lnbnn_weights: [N, K] float64 from :func:`weight_neighbors_lnbnn`.
        labels: [N, K] int64 — db-annotation indices.
        local_labels: [N, K] int64 — local feature indices.
        database: annotations in index order.
        query_annot_index: query index in *database*.
        fg_weights: optional pre-computed fg weights (from
            :func:`_compute_fg_for_database`).  Computed on-the-fly if
            ``None``.

    Returns:
        [N, K] float64 — element-wise product of lnbnn and fg weights.
    """
    if fg_weights is None:
        fg_weights = _compute_fg_for_database(database)

    n, k = labels.shape
    q_fg = fg_weights[query_annot_index]

    result = lnbnn_weights.copy()

    for j in range(k):
        db_idxs = labels[:, j]
        local_idxs = local_labels[:, j]
        valid = db_idxs >= 0
        if not valid.any():
            continue
        for qfx, db_idx, dfx in zip(
            np.where(valid)[0],
            db_idxs[valid],
            local_idxs[valid],
        ):
            fg = np.sqrt(q_fg[qfx] * fg_weights[db_idx][dfx])
            result[qfx, j] *= fg

    return result


def build_matches(
    weights: np.ndarray,
    labels: np.ndarray,
    local_labels: np.ndarray,
    database: list[AnnotatedImage],
) -> list[Match]:
    """Convert the weighted neighbour table into a flat list of :class:`Match`.

    Only entries with ``weight > 0`` are emitted.

    Args:
        weights: [N, K] float64.
        labels: [N, K] int64 — annotation indices.
        local_labels: [N, K] int64 — local feature indices within each
            annotation.  Same shape as *labels*.
        database: annotations in index order.

    Returns:
        List of :class:`Match` objects.
    """
    matches: list[Match] = []
    n_features = weights.shape[0]
    for qfx in range(n_features):
        row = weights[qfx]
        lbl = labels[qfx]
        dfx = local_labels[qfx]
        nonzero = row > 0
        for j in np.where(nonzero)[0]:
            db_idx = lbl[j]
            if db_idx < 0 or db_idx >= len(database):
                continue
            annot = database[db_idx]
            matches.append(
                Match(
                    qfx=qfx,
                    daid=db_idx,
                    dfx=int(dfx[j]),
                    dist=float(row[j]),
                    name_uuid=annot.name_uuid,
                )
            )
    return matches


def score_matches(
    matches: list[Match],
    database: list[AnnotatedImage],
    score_method: str = "nsum",
) -> list[ScoredMatch]:
    """Aggregate per-feature matches into per-annotation scores.

    ``nsum`` (normalized sum)::

        score = sum(weights) / count(weights)

    ``csum`` (cumulative sum)::

        score = sum(weights)

    Args:
        matches: list from :func:`build_matches`.
        database: annotations in index order.
        score_method: ``"nsum"`` or ``"csum"``.

    Returns:
        List of :class:`ScoredMatch`, one per matched annotation,
        sorted by descending score.
    """
    agg: dict[uuid.UUID, tuple[float, int, list[tuple[int, int]]]] = {}

    for m in matches:
        annot_uuid = database[m.daid].annot_uuid
        cur_score, cur_count, cur_corr = agg.get(annot_uuid, (0.0, 0, []))
        cur_corr.append((m.qfx, m.dfx))
        agg[annot_uuid] = (cur_score + m.dist, cur_count + 1, cur_corr)

    results: list[ScoredMatch] = []
    for annot_uuid, (total_weight, count, corrs) in agg.items():
        if score_method == "nsum":
            score = total_weight / count
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
