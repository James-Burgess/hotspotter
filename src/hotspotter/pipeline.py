"""Stateless identification pipeline — top-level :func:`identify`.

Matches WBIA's global-index ``vsmany`` pipeline:

* A **single** FLANN index over database descriptors (query
  excluded), matching WBIA's ``NeighborIndex``  behaviour.
* ``K+Kpad`` voting columns + ``Knorm`` normaliser column; self /
  same-name matches are filtered from the voting columns only
  (WBIA's ``baseline_neighbor_filter``).
* Maintains the same data ordering as WBIA's ``daid_list`` so that
  the FLANN KD-tree structure is reproducible.
"""

from __future__ import annotations

import uuid
from typing import NamedTuple

import numpy as np

from hotspotter.config import IdentificationConfig
from hotspotter.data import AnnotatedImage, FeatureSet, Match, ScoredMatch
from hotspotter.knn import build_global_index, exact_knn, query_index
from hotspotter.name_scoring import score_matches_with_names
from hotspotter.scoring import (
    apply_fg_weights,
    baseline_filter,
    build_matches,
    compute_normalizer_validity,
    per_feature_fg,
    score_matches,
    weight_neighbors_lnbnn,
)
from hotspotter.spatial import make_sver_shortlist, spatial_verify

from hotspotter import debug_log as dlog

from hotspotter.trace import get_trace_context


def _compute_kpad(hs, query_annot_index: int, database: list[AnnotatedImage]) -> int:
    """Compute Kpad based on policy.

    'fixed' — use hs.kpad value directly.
    'dynamic' — count impossible (same-name and/or same-image) annotations
      that WBIA's ``build_impossible_daids_list`` would mask, mirroring
      ``Kpad_list = list(map(len, impossible_daids_list))``.  The query is
      excluded from the FLANN index (matching WBIA), so there is no
      self-match column to strip.
    """
    if hs.kpad_policy == "fixed":
        return hs.kpad

    qann = database[query_annot_index]
    impossible: set[int] = set()
    if not hs.can_match_samename and qann.name_uuid is not None:
        for i, a in enumerate(database):
            if i != query_annot_index and a.name_uuid == qann.name_uuid:
                impossible.add(i)
    if not hs.can_match_sameimg and qann.image_uuid is not None:
        for i, a in enumerate(database):
            if i != query_annot_index and a.image_uuid == qann.image_uuid:
                impossible.add(i)
    return len(impossible)


_WBIA_SCORE_METHODS = frozenset({"csum", "csum_wbia", "nsum", "nsum_wbia", "sumamech"})


class KnnResult(NamedTuple):
    raw_dists: np.ndarray
    raw_labels: np.ndarray
    annot_of_desc: np.ndarray
    feat_of_desc: np.ndarray
    n_total: int
    db_feature_sets: list


class VoteColumns(NamedTuple):
    voting_dists: np.ndarray
    norm_dists: np.ndarray
    voting_annot: np.ndarray
    voting_feat: np.ndarray


def _filter_query_features(database, query_annot_index, hs):
    query = database[query_annot_index].features
    feat_mask = np.ones(len(query), dtype=bool)
    if hs.minscale_thresh is not None:
        feat_mask &= query.keypoints[:, 2] >= hs.minscale_thresh
    if hs.maxscale_thresh is not None:
        feat_mask &= query.keypoints[:, 2] <= hs.maxscale_thresh
    if hs.fgw_thresh is not None:
        feat_mask &= per_feature_fg(database[query_annot_index]) >= hs.fgw_thresh
    if feat_mask.all():
        return query
    filtered = FeatureSet(
        keypoints=query.keypoints[feat_mask],
        descriptors=query.descriptors[feat_mask],
    )
    assert (
        len(filtered) > 0
    ), "All query features filtered out by minscale/maxscale/fgw_thresh"
    return filtered


def _query_neighbors(database, query_features, hs, k_total):
    db_feature_sets = [ann.features for ann in database]
    db_remap = np.arange(len(database), dtype=np.int32)
    if hs.knn_backend == "exact":
        all_descs = np.concatenate([fs.descriptors for fs in db_feature_sets], axis=0)
        n_total = all_descs.shape[0]
        annot_of_desc = np.empty(n_total, dtype=np.int32)
        feat_of_desc = np.empty(n_total, dtype=np.int32)
        offset = 0
        for i, fs in enumerate(db_feature_sets):
            n = len(fs)
            annot_of_desc[offset : offset + n] = db_remap[i]
            feat_of_desc[offset : offset + n] = np.arange(n, dtype=np.int32)
            offset += n
        db_feats = FeatureSet(
            keypoints=np.empty((n_total, 6), dtype=np.float64),
            descriptors=all_descs,
        )
        raw_dists, raw_labels = exact_knn(query_features, db_feats, k_total)
    else:
        backend = "pyflann" if hs.knn_backend == "flann" else "faiss"
        global_index, annot_of_desc_local, feat_of_desc = build_global_index(
            db_feature_sets,
            algorithm=hs.flann_algorithm,
            trees=hs.flann_trees,
            random_seed=hs.flann_random_seed,
            backend=backend,
        )
        annot_of_desc = db_remap[annot_of_desc_local]
        n_total = annot_of_desc.shape[0]
        raw_dists, raw_labels = query_index(
            global_index,
            query_features,
            k_total,
            checks=hs.flann_checks,
            cores=hs.flann_cores,
            backend=backend,
        )
    dlog.stage_global_index(db_feature_sets, annot_of_desc)
    dlog.stage_raw_dists(raw_dists, raw_labels)
    return KnnResult(
        raw_dists, raw_labels, annot_of_desc, feat_of_desc, n_total, db_feature_sets
    )


SIFT_MAX_SQRT_DIST = 2.0 * (512.0**2.0)


def _normalize_distances(ctx, query_annot_index, knn, hs):
    max_distance_sqrd = SIFT_MAX_SQRT_DIST
    dists = (np.maximum(knn.raw_dists, 0.0) / max_distance_sqrd).astype(np.float64)
    if ctx is not None:
        ctx.trace_neighbors(
            query_annot_index + 1, knn.raw_labels, dists.astype(np.float32)
        )
    if not hs.sqrd_dist_on:
        dists = np.sqrt(dists)
    dlog.stage_dist_norm(dists)
    return dists, knn.raw_labels.astype(np.int64)


def _build_vote_columns(ctx, query_annot_index, dists, labels, knn, k, kpad):
    n_qfxs = dists.shape[0]
    voting_dists = dists[:, : k + kpad]
    norm_dists = dists[:, k + kpad :]
    voting_annot = np.full((n_qfxs, k + kpad), -1, dtype=np.int32)
    voting_feat = np.full((n_qfxs, k + kpad), -1, dtype=np.int32)
    for j in range(k + kpad):
        col = labels[:, j]
        valid = (col >= 0) & (col < knn.n_total)
        voting_annot[valid, j] = knn.annot_of_desc[col[valid]]
        voting_feat[valid, j] = knn.feat_of_desc[col[valid]]
    dlog.stage_voting_cols(dists, labels, knn.annot_of_desc, k, kpad, query_annot_index)
    return VoteColumns(voting_dists, norm_dists, voting_annot, voting_feat)


def _score_and_build(
    ctx, query_annot_index, votes, dists, labels, knn, database, hs, k, kpad
):
    """Run the scoring chain: filter → weight → fg → build matches.

    Delegates all algorithmic logic to scoring.py; this function only
    wires config, calls the scoring functions in order, and emits
    trace/debug hooks between them.
    """
    invalid, impossible_set, qname = baseline_filter(
        votes.voting_annot,
        database,
        query_annot_index,
        can_match_samename=hs.can_match_samename,
        can_match_sameimg=hs.can_match_sameimg,
    )
    dlog.stage_filter_counts(
        dists, labels, knn.annot_of_desc, k, kpad, query_annot_index, impossible_set
    )
    if ctx is not None:
        ctx.trace_baseline_filter(query_annot_index + 1, ~invalid)

    normalizer_valid = None
    if hs.normalizer_rule == "name":
        normalizer_valid = compute_normalizer_validity(
            votes.voting_annot,
            labels,
            knn.annot_of_desc,
            knn.n_total,
            database,
            k,
            kpad,
            qname,
        )

    weights = weight_neighbors_lnbnn(
        votes.voting_dists,
        votes.norm_dists,
        normonly_on=hs.normonly_on,
        bar_l2_on=hs.bar_l2_on,
        ratio_thresh=hs.ratio_thresh,
        lnbnn_ratio=hs.lnbnn_ratio,
    )

    if hs.fg_on:
        fg_weights = [per_feature_fg(ann) for ann in database]
        weights = apply_fg_weights(
            weights,
            votes.voting_annot,
            votes.voting_feat,
            database,
            query_annot_index,
            fg_weights,
        )

    matches = build_matches(
        weights,
        votes.voting_annot,
        votes.voting_feat,
        invalid,
        database,
        k,
        kpad,
        normalizer_valid,
    )

    lnbnn_weights = [m.dist for m in matches]
    dlog.stage_lnbnn_weights(lnbnn_weights)
    if ctx is not None:
        ctx.trace_neighbor_weights(query_annot_index + 1, lnbnn_weights)

    return matches, qname


def _score_matches(matches, database, query_features, hs):
    annot_uuids = [a.annot_uuid for a in database]
    annot_name_map = {
        a.annot_uuid: a.name_uuid for a in database if a.name_uuid is not None
    }
    qk = query_features.keypoints if hs.rotation_invariance else None
    if hs.score_method not in _WBIA_SCORE_METHODS:
        raise ValueError(f"Unknown score_method: {hs.score_method!r}")
    csum_annot, name_scores, canonical = score_matches_with_names(
        matches, annot_uuids, annot_name_map, hs.score_method, qk
    )
    scored = _wbia_scores_to_scoredmatches(
        csum_annot, canonical, matches, database, annot_uuids
    )
    _annot_to_matches: dict[int, tuple[float, int]] = {}
    for sm in scored:
        daid = next(i for i, a in enumerate(database) if a.annot_uuid == sm.annot_uuid)
        _annot_to_matches[daid] = (csum_annot.get(sm.annot_uuid, 0.0), sm.num_matches)
    dlog.stage_match_to_annot(_annot_to_matches)
    return scored, csum_annot, name_scores, canonical


def _run_spatial_verification(
    scored, matches, database, query_features, hs, csum_annot, name_scores, canonical
):
    master_matches: list[Match] = list(matches)
    if not hs.sv_on:
        return scored, master_matches, csum_annot, name_scores, canonical

    prescored = scored
    if hs.prescore_method != hs.score_method:
        prescored = _prescore_candidates(matches, database, hs.prescore_method)
    if hs.sv_verify_all:
        sver_candidates = list(prescored)
    else:
        sver_candidates = make_sver_shortlist(
            prescored,
            n_names=hs.sv_n_name_shortlist,
            n_annots_per_name=hs.sv_n_annot_per_name,
            score_method=hs.score_method,
        )
    dist_lookup: dict[tuple[int, int, int], float] = {}
    for m in matches:
        dist_lookup[(m.daid, m.qfx, m.dfx)] = m.dist
    sver_candidates, sv_results = spatial_verify(
        sver_candidates,
        query_features,
        database,
        min_inliers=4,
        xy_thresh=hs.sv_xy_thresh,
        scale_thresh=hs.sv_scale_thresh,
        ori_thresh=hs.sv_ori_thresh,
        use_chip_extent=hs.sv_use_chip_extent,
        weight_inliers=hs.sv_weight_inliers,
        sver_output_weighting=hs.sv_sver_output_weighting,
        dist_lookup=dist_lookup,
    )

    if not sv_results:
        return scored, master_matches, csum_annot, name_scores, canonical

    match_lookup: dict[tuple[int, int, int], Match] = {}
    for m in matches:
        match_lookup[(m.daid, m.qfx, m.dfx)] = m
    sv_matches: list[Match] = []
    for annot_uuid, (inlier_qfxs, inlier_dfxs, weights) in sv_results.items():
        daid = next(i for i, a in enumerate(database) if a.annot_uuid == annot_uuid)
        for qfx, dfx, w in zip(inlier_qfxs, inlier_dfxs, weights, strict=True):
            orig = match_lookup.get((daid, qfx, dfx))
            if orig is None:
                continue
            sv_w = float(w) if hs.sv_sver_output_weighting else 1.0
            sv_matches.append(
                Match(
                    qfx=qfx,
                    daid=daid,
                    dfx=dfx,
                    dist=orig.dist * sv_w,
                    name_uuid=orig.name_uuid,
                    sv_weight=sv_w,
                )
            )

    annot_uuids = [a.annot_uuid for a in database]
    annot_name_map = {
        a.annot_uuid: a.name_uuid for a in database if a.name_uuid is not None
    }
    if hs.score_method in _WBIA_SCORE_METHODS:
        qk = query_features.keypoints if hs.rotation_invariance else None
        sv_csum, sv_name_scores, sv_canonical = score_matches_with_names(
            sv_matches, annot_uuids, annot_name_map, hs.score_method, qk
        )
        sv_scored = _wbia_scores_to_scoredmatches(
            sv_csum, sv_canonical, sv_matches, database, annot_uuids
        )
    else:
        simple_score_method = "csum" if hs.score_method == "nsum" else hs.score_method
        sv_scored = score_matches(sv_matches, database, simple_score_method)
        sv_name_scores = {}
        sv_csum = {}
        sv_canonical = {}

    sv_annot_uuids = set(sv_results.keys())
    sv_inlier_info = {
        sm.annot_uuid: (sm.sv_inliers, sm.sv_homography)
        for sm in sver_candidates
        if sm.sv_inliers > 0
    }
    sv_score_map = {sm.annot_uuid: sm for sm in sv_scored}
    for sm in scored:
        if sm.annot_uuid in sv_score_map:
            replaced = sv_score_map[sm.annot_uuid]
            inliers, H = sv_inlier_info.get(sm.annot_uuid, (0, None))
            sm.sv_inliers = inliers
            sm.sv_homography = H
            sm.score = replaced.score
            sm.num_matches = replaced.num_matches
            sm.correspondences = replaced.correspondences

    for sm in scored:
        au = sm.annot_uuid
        if au in sv_annot_uuids:
            csum_annot[au] = sv_csum.get(au, csum_annot.get(au, 0.0))
            canonical[au] = sv_canonical.get(au, canonical.get(au, -np.inf))
        nu = sm.name_uuid
        if nu is not None and nu in sv_name_scores:
            name_scores[nu] = sv_name_scores[nu]

    scored = [sm for sm in scored if sm.annot_uuid in sv_annot_uuids]

    sv_daid_set = {sm.daid for sm in sv_matches}
    master_matches = list(sv_matches) + [
        m for m in matches if m.daid not in sv_daid_set
    ]
    return scored, master_matches, csum_annot, name_scores, canonical


def _trace_ids(database, query_annot_index):
    uuid_to_daid = {ann.annot_uuid: i + 1 for i, ann in enumerate(database)}
    name_to_nid: dict = {}
    next_nid = 1
    for ann in database:
        if ann.name_uuid is not None and ann.name_uuid not in name_to_nid:
            name_to_nid[ann.name_uuid] = next_nid
            next_nid += 1
    qaid = query_annot_index + 1
    qann = database[query_annot_index]
    qnid = name_to_nid.get(qann.name_uuid, -1) if qann.name_uuid else -1
    return uuid_to_daid, name_to_nid, qaid, qnid


def _emit_chipmatches(
    ctx,
    stage,
    scored,
    fm_matches,
    database,
    csum_annot,
    name_scores,
    canonical,
    trace_ids,
    sort_by_csum=False,
):
    if ctx is None:
        return
    uuid_to_daid, name_to_nid, qaid, qnid = trace_ids
    daids = np.array([uuid_to_daid[sm.annot_uuid] for sm in scored], dtype=np.int64)
    dnids = np.array(
        [name_to_nid.get(sm.name_uuid, -1) if sm.name_uuid else -1 for sm in scored],
        dtype=np.int64,
    )
    csum = np.array(
        [float(csum_annot.get(sm.annot_uuid, 0.0)) for sm in scored], dtype=np.float64
    )
    nsum = np.array(
        [
            (
                float(name_scores.get(sm.name_uuid, 0.0))
                if sm.name_uuid and sm.name_uuid in name_scores
                else 0.0
            )
            for sm in scored
        ],
        dtype=np.float64,
    )
    canon = np.array(
        [float(canonical.get(sm.annot_uuid, -np.inf)) for sm in scored],
        dtype=np.float64,
    )
    if sort_by_csum:
        order = np.argsort(
            [-csum_annot.get(sm.annot_uuid, 0.0) for sm in scored], kind="stable"
        )
    else:
        order = np.argsort(daids)
    daids, dnids = daids[order], dnids[order]
    csum, nsum, canon = csum[order], nsum[order], canon[order]
    fm_list = _fm_arrays_for_scored([scored[i] for i in order], fm_matches, database)
    ctx.trace_chipmatches(
        stage, qaid, qnid, daids, dnids, csum, nsum, canon, fm_list, fsv_list=None
    )


def _emit_final(
    ctx,
    scored,
    fm_matches,
    database,
    csum_annot,
    name_scores,
    score_method,
    trace_ids,
):
    if ctx is None:
        return
    uuid_to_daid, name_to_nid, qaid, qnid = trace_ids
    daids = np.array([uuid_to_daid[sm.annot_uuid] for sm in scored], dtype=np.int64)
    dnids = -daids.copy()
    csum = np.array(
        [float(csum_annot.get(sm.annot_uuid, 0.0)) for sm in scored], dtype=np.float64
    )
    nsum = np.array(
        [
            (
                float(name_scores.get(sm.name_uuid, 0.0))
                if sm.name_uuid and sm.name_uuid in name_scores
                else 0.0
            )
            for sm in scored
        ],
        dtype=np.float64,
    )
    fm_list = _fm_arrays_for_scored(scored, fm_matches, database)
    ctx.trace_final_scores(
        qaid=qaid,
        qnid=qnid,
        score_method=_trace_score_method(score_method),
        daid_list=daids,
        dnid_list=dnids,
        annot_score_list=csum,
        name_score_list=nsum,
        score_list=nsum,
        fm_list=fm_list,
    )


def identify(
    query_annot_index: int,
    database: list[AnnotatedImage],
    config: IdentificationConfig | None = None,
    trace_query_index: int | None = None,
) -> list[ScoredMatch]:
    """Run the identification pipeline for one query against *database*.

    A single global FLANN index is built over database descriptors
    only (the query annotation is excluded, matching WBIA).  The
    ``K+Kpad+Knorm`` nearest neighbours are fetched; self /
    same-name matches are filtered from the voting columns only
    (the ``Knorm`` normaliser column is not filtered).

    Args:
        query_annot_index: index of the query annotation in *database*.
        database: all candidate annotations (may include the query).
        config: pipeline configuration.
        trace_query_index: sequential query number for trace filenames
            (defaults to *query_annot_index* when tracing is enabled).

    Returns:
        Top-``config.hotspotter.num_return`` scored matches descending.
    """
    if config is None:
        config = IdentificationConfig()
    hs = config.hotspotter

    dlog.stage_features(database)

    ctx = get_trace_context(
        query_index=(
            trace_query_index if trace_query_index is not None else query_annot_index
        )
    )

    if ctx is not None:
        ctx.trace_annotations(database, query_annot_index)
        ctx.trace_chips_and_features(database, query_annot_index)

    if config.pipeline != "HotSpotter":
        raise NotImplementedError(
            f"Pipeline {config.pipeline!r} is not yet implemented."
        )

    k = hs.knn
    kpad = _compute_kpad(hs, query_annot_index, database)
    k_total = k + kpad + hs.knorm

    query_features = _filter_query_features(database, query_annot_index, hs)
    knn = _query_neighbors(database, query_features, hs, k_total)

    dists, labels = _normalize_distances(ctx, query_annot_index, knn, hs)
    votes = _build_vote_columns(ctx, query_annot_index, dists, labels, knn, k, kpad)

    matches, qname = _score_and_build(
        ctx, query_annot_index, votes, dists, labels, knn, database, hs, k, kpad
    )

    scored, csum_annot, _name_scores, canonical = _score_matches(
        matches, database, query_features, hs
    )

    trace_ids = _trace_ids(database, query_annot_index)
    _emit_chipmatches(
        ctx,
        "chipmatches_pre_sv",
        scored,
        matches,
        database,
        csum_annot,
        _name_scores,
        canonical,
        trace_ids,
    )

    scored, _master_matches, csum_annot, _name_scores, canonical = (
        _run_spatial_verification(
            scored,
            matches,
            database,
            query_features,
            hs,
            csum_annot,
            _name_scores,
            canonical,
        )
    )

    _emit_chipmatches(
        ctx,
        "chipmatches_post_sv",
        scored,
        _master_matches,
        database,
        csum_annot,
        _name_scores,
        canonical,
        trace_ids,
        sort_by_csum=True,
    )

    dlog.stage_final(scored)

    # WBIA ranks results by the canonical name score (``cm.score_list``),
    # set via ``set_cannonical_name_score`` → ``align_name_scores_with_annots``
    # (best annotation per name carries the name score; same-name runners-up
    # get ``-inf`` and sink). ``ScoredMatch.score`` holds that canonical value,
    # so sorting by it reproduces ``get_top_aids``/``get_top_nids`` which
    # argsort ``cm.score_list`` descending. csum is used only as a stable
    # tiebreak among the ``-inf`` non-canonical annotations.
    scored.sort(
        key=lambda sm: (sm.score, csum_annot.get(sm.annot_uuid, 0.0)),
        reverse=True,
    )

    _emit_final(
        ctx,
        scored,
        _master_matches,
        database,
        csum_annot,
        _name_scores,
        hs.score_method,
        trace_ids,
    )

    return scored[: hs.num_return]


def _trace_score_method(score_method: str) -> str:
    """Map hotspotter implementation labels to WBIA trace labels."""
    if score_method == "nsum_wbia":
        return "nsum"
    if score_method == "csum_wbia":
        return "csum"
    return score_method


def _fm_arrays_for_scored(
    scored: list[ScoredMatch], matches: list[Match], database: list[AnnotatedImage]
) -> list[np.ndarray]:
    """Return WBIA-style per-chipmatch ``fm_list`` arrays.

    Each returned array is ``[N, 2]`` with ``(qfx, dfx)`` rows, ordered to match
    the scored result / ``daid_list`` order written in the same trace row.
    """
    uuid_to_daid = {ann.annot_uuid: daid for daid, ann in enumerate(database)}
    by_daid: dict[int, list[tuple[int, int]]] = {}
    for match in matches:
        by_daid.setdefault(match.daid, []).append((match.qfx, match.dfx))

    arrays = []
    for sm in scored:
        daid = uuid_to_daid.get(sm.annot_uuid)
        pairs = by_daid.get(daid, []) if daid is not None else []
        arrays.append(np.asarray(pairs, dtype=np.int32).reshape(-1, 2))
    return arrays


def _canonical_to_scoredmatches(
    canonical: dict[uuid.UUID, float],
    matches: list[Match],
    database: list[AnnotatedImage],
    annot_uuids: list[uuid.UUID],
) -> list[ScoredMatch]:
    """Convert canonical name score dict to ``ScoredMatch`` list."""
    match_count: dict[uuid.UUID, int] = {}
    corrs: dict[uuid.UUID, list[tuple[int, int]]] = {}
    for m in matches:
        a_uuid = annot_uuids[m.daid]
        match_count[a_uuid] = match_count.get(a_uuid, 0) + 1
        corrs.setdefault(a_uuid, []).append((m.qfx, m.dfx))

    results: list[ScoredMatch] = []
    for annot_uuid, score in canonical.items():
        annot = next(a for a in database if a.annot_uuid == annot_uuid)
        results.append(
            ScoredMatch(
                annot_uuid=annot_uuid,
                name_uuid=annot.name_uuid,
                score=score,
                num_matches=match_count.get(annot_uuid, 0),
                correspondences=corrs.get(annot_uuid, []),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _wbia_scores_to_scoredmatches(
    csum_annot: dict[uuid.UUID, float],
    canonical: dict[uuid.UUID, float],
    matches: list[Match],
    database: list[AnnotatedImage],
    annot_uuids: list[uuid.UUID],
) -> list[ScoredMatch]:
    """Convert WBIA name scores while preserving all matched annotations.

    WBIA keeps ``daid_list`` annotation-aligned and stores canonical name scores
    in ``score_list``: only the best annotation for each name receives the name
    score; same-name non-canonical annotations get ``-inf``.
    """
    match_count: dict[uuid.UUID, int] = {}
    corrs: dict[uuid.UUID, list[tuple[int, int]]] = {}
    for m in matches:
        annot_uuid = annot_uuids[m.daid]
        match_count[annot_uuid] = match_count.get(annot_uuid, 0) + 1
        corrs.setdefault(annot_uuid, []).append((m.qfx, m.dfx))

    uuid_to_annot = {ann.annot_uuid: ann for ann in database}
    results: list[ScoredMatch] = []
    for annot_uuid, _annot_csum in csum_annot.items():
        annot = uuid_to_annot[annot_uuid]
        results.append(
            ScoredMatch(
                annot_uuid=annot_uuid,
                name_uuid=annot.name_uuid,
                score=float(canonical.get(annot_uuid, -np.inf)),
                annot_csum=float(csum_annot.get(annot_uuid, 0.0)),
                num_matches=match_count.get(annot_uuid, 0),
                correspondences=corrs.get(annot_uuid, []),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _prescore_candidates(
    matches: list[Match],
    database: list[AnnotatedImage],
    score_method: str,
) -> list[ScoredMatch]:
    """Quick prescore for SV shortlisting using *score_method*.

    Returns a plain per-annot csum list for ``make_sver_shortlist``.
    """
    _wbia_methods = {"csum_wbia", "nsum_wbia", "sumamech"}
    if score_method in _wbia_methods:
        annot_uuids = [a.annot_uuid for a in database]
        annot_name_map = {
            a.annot_uuid: a.name_uuid for a in database if a.name_uuid is not None
        }
        csum_annot, _, canonical = score_matches_with_names(
            matches, annot_uuids, annot_name_map, score_method
        )
        return _wbia_scores_to_scoredmatches(
            csum_annot, canonical, matches, database, annot_uuids
        )
    else:
        simple_score_method = "csum" if score_method == "nsum" else score_method
        return score_matches(matches, database, simple_score_method)
