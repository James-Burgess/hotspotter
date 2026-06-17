"""Name-level scoring — fmech (nsum) and canonical alignment.

Mirrors ``wildbook-ia/wbia/algo/hots/name_scoring.py``.
"""

from __future__ import annotations

import uuid

import numpy as np

from wbia_core.data import Match


def _compute_xy_combo_ids(
    keypoints: np.ndarray,
) -> dict[int, int]:
    """Map feature indices to XY-coordinate combo IDs.

    Keypoints at the same spatial position (rounded) get the same
    combo ID — used by ``query_rotation_heuristic`` to prevent
    rotated duplicate features from voting multiple times per name.

    Matches WBIA's ``vt.compute_unique_arr_dataids(xys1_)``.
    """
    xs = np.round(keypoints[:, 0]).astype(np.int64)
    ys = np.round(keypoints[:, 1]).astype(np.int64)
    xy_pairs = list(zip(xs, ys, strict=True))
    unique = {}
    next_id = 0
    combo: dict[int, int] = {}
    for fx, pair in enumerate(xy_pairs):
        cid = unique.setdefault(pair, next_id)
        if cid == next_id:
            next_id += 1
        combo[fx] = cid
    return combo


def group_matches_by_name(
    matches: list[Match],
) -> dict[uuid.UUID, list[Match]]:
    """Group *matches* by ``name_uuid``.  Matches with ``None`` name are skipped."""
    by_name: dict[uuid.UUID, list[Match]] = {}
    for m in matches:
        if m.name_uuid is not None:
            by_name.setdefault(m.name_uuid, []).append(m)
    return by_name


def compute_fmech_score(
    matches_by_name: dict[uuid.UUID, list[Match]],
    query_keypoints: np.ndarray | None = None,
) -> dict[uuid.UUID, float]:
    """Compute nsum / fmech name-level scores.

    WBIA algorithm (``compute_fmech_score``, name_scoring.py:53):

    1. Group feature matches by name.
    2. Within each name, group matches by query feature index (qfx)
       or, if *query_keypoints* is provided, by XY coordinate
       (``query_rotation_heuristic`` — prevents rotated duplicate
       features from voting multiple times per name).
    3. For each group, keep only the match with the **maximum**
       combined weight (``dist``).
    4. Sum the surviving feature scores to get the name score.
    """
    combo_from_fx: dict[int, int] | None = None
    if query_keypoints is not None:
        combo_from_fx = _compute_xy_combo_ids(query_keypoints)

    nsum: dict[uuid.UUID, float] = {}
    for name_uuid, name_matches in matches_by_name.items():
        if combo_from_fx is None:
            by_group: dict[int, list[Match]] = {}
            for m in name_matches:
                by_group.setdefault(m.qfx, []).append(m)
        else:
            by_group = {}
            for m in name_matches:
                cid = combo_from_fx.get(m.qfx, m.qfx)
                by_group.setdefault(cid, []).append(m)

        name_score = 0.0
        for group in by_group.values():
            best = max(group, key=lambda x: x.dist)
            name_score += best.dist

        nsum[name_uuid] = name_score
    return nsum


def compute_csum_annot_scores(
    matches: list[Match],
    annot_uuids: list[uuid.UUID],
) -> dict[uuid.UUID, float]:
    """Compute per-annotation csum scores from flat match list.

    Resolves *annot_uuid* from ``annot_uuids[m.daid]``.
    """
    csum: dict[uuid.UUID, float] = {}
    for m in matches:
        a_uuid = annot_uuids[m.daid]
        csum[a_uuid] = csum.get(a_uuid, 0.0) + m.dist
    return csum


def compute_maxcsum_name_score(
    csum_annot_scores: dict[uuid.UUID, float],
    annot_name_map: dict[uuid.UUID, uuid.UUID],
) -> dict[uuid.UUID, float]:
    """Compute max-per-name scores from per-annotation csum values.

    For each name, takes the maximum csum among its annotations.
    """
    name_scores: dict[uuid.UUID, float] = {}
    name_annots: dict[uuid.UUID, list[tuple[uuid.UUID, float]]] = {}
    for annot_uuid, score in csum_annot_scores.items():
        nid = annot_name_map.get(annot_uuid)
        if nid is not None:
            name_annots.setdefault(nid, []).append((annot_uuid, score))

    for nid, items in name_annots.items():
        name_scores[nid] = max(score for _, score in items)
    return name_scores


def compute_sumamech_name_score(
    csum_annot_scores: dict[uuid.UUID, float],
    annot_name_map: dict[uuid.UUID, uuid.UUID],
) -> dict[uuid.UUID, float]:
    """Compute per-name sum of per-annotation csum values."""
    name_scores: dict[uuid.UUID, float] = {}
    for annot_uuid, score in csum_annot_scores.items():
        nid = annot_name_map.get(annot_uuid)
        if nid is not None:
            name_scores[nid] = name_scores.get(nid, 0.0) + score
    return name_scores


def align_name_scores_with_annots(
    csum_annot_scores: dict[uuid.UUID, float],
    annot_name_map: dict[uuid.UUID, uuid.UUID],
    name_scores: dict[uuid.UUID, float],
) -> dict[uuid.UUID, float]:
    """Canonical name score alignment (WBIA ``align_name_scores_with_annots``).

    For each name, the name-level score is assigned to the single
    annotation with the highest csum for that name.  All other
    annotations of the same name are excluded (not present in the
    returned dict).
    """
    canonical: dict[uuid.UUID, float] = {}

    name_annots: dict[uuid.UUID, list[tuple[uuid.UUID, float]]] = {}
    for annot_uuid, score in csum_annot_scores.items():
        nid = annot_name_map.get(annot_uuid)
        if nid is not None and nid in name_scores:
            name_annots.setdefault(nid, []).append((annot_uuid, score))

    for nid, items in name_annots.items():
        best_annot, _ = max(items, key=lambda x: x[1])
        canonical[best_annot] = name_scores[nid]

    return canonical


def score_matches_with_names(
    matches: list[Match],
    annot_uuids: list[uuid.UUID],
    annot_name_map: dict[uuid.UUID, uuid.UUID],
    score_method: str = "nsum_wbia",
    query_keypoints: np.ndarray | None = None,
) -> tuple[dict[uuid.UUID, float], dict[uuid.UUID, float]]:
    """Full WBIA scoring chain: annot csum → name scoring → canonical.

    Args:
        matches: Flat feature-match list.
        annot_uuids: Mapping from database index → annot_uuid.
        annot_name_map: Mapping from annot_uuid → name_uuid.
        score_method: ``"nsum_wbia"`` (fmech), ``"csum_wbia"`` (max-per-name),
            or ``"sumamech"``.
        query_keypoints: Optional [N, 6] query keypoints for
            ``query_rotation_heuristic`` XY-dedup in fmech.

    Returns:
        ``(csum_annot_scores, canonical_scores)`` — both ``{annot_uuid: score}``.
    """
    csum: dict[uuid.UUID, float] = {}
    for m in matches:
        annot_uuid = annot_uuids[m.daid]
        csum[annot_uuid] = csum.get(annot_uuid, 0.0) + m.dist

    if score_method == "csum_wbia":
        name_scores = compute_maxcsum_name_score(csum, annot_name_map)
    elif score_method == "sumamech":
        name_scores = compute_sumamech_name_score(csum, annot_name_map)
    elif score_method == "nsum_wbia":
        matches_by_name = group_matches_by_name(matches)
        name_scores = compute_fmech_score(matches_by_name, query_keypoints)
    else:
        raise ValueError(f"Unknown score_method: {score_method!r}")

    canonical = align_name_scores_with_annots(csum, annot_name_map, name_scores)
    return csum, canonical
