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

import numpy as np

from hotspotter.config import IdentificationConfig
from hotspotter.data import AnnotatedImage, FeatureSet, Match, ScoredMatch
from hotspotter.knn import build_global_index, exact_knn, query_index
from hotspotter.name_scoring import score_matches_with_names
from hotspotter.scoring import per_feature_fg, score_matches
from hotspotter.spatial import make_sver_shortlist, spatial_verify

from hotspotter import debug_log as dlog

from hotspotter.trace import get_trace_context


def _compute_kpad(hs, query_annot_index: int, database: list[AnnotatedImage]) -> int:
    """Compute Kpad based on policy.

    'fixed' — use hs.kpad value directly.
    'dynamic' — count impossible (same-name) annotations that WBIA's
      ``build_impossible_daids_list`` would mask.  The query is
      excluded from the FLANN index (matching WBIA), so there is
      no self-match column to strip.
    """
    if hs.kpad_policy == "fixed":
        return hs.kpad

    kpad = 0
    if not hs.can_match_samename:
        qname = database[query_annot_index].name_uuid
        if qname is not None:
            for i, a in enumerate(database):
                if i != query_annot_index and a.name_uuid == qname:
                    kpad += 1
    return kpad


def identify(
    query_annot_index: int,
    database: list[AnnotatedImage],
    config: IdentificationConfig = IdentificationConfig(),
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
    hs = config.hotspotter
    query_features = database[query_annot_index].features

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
    knorm = hs.knorm
    kpad = _compute_kpad(hs, query_annot_index, database)
    k_total = k + kpad + knorm  # e.g. 4 + 0 + 1 = 5

    # ---- 0. Pre-query feature filtering (minscale/maxscale/fgw_thresh) ----
    query_kp = query_features.keypoints
    query_desc = query_features.descriptors
    feat_mask = np.ones(len(query_features), dtype=bool)

    if hs.minscale_thresh is not None:
        scales = query_kp[:, 2]
        feat_mask &= scales >= hs.minscale_thresh
    if hs.maxscale_thresh is not None:
        scales = query_kp[:, 2]
        feat_mask &= scales <= hs.maxscale_thresh
    if hs.fgw_thresh is not None:
        q_fg = per_feature_fg(database[query_annot_index])
        feat_mask &= q_fg >= hs.fgw_thresh

    if not feat_mask.all():
        query_features = FeatureSet(
            keypoints=query_kp[feat_mask],
            descriptors=query_desc[feat_mask],
        )
        assert (
            len(query_features) > 0
        ), "All query features filtered out by minscale/maxscale/fgw_thresh"

    # ---- 1. Build global FLANN index over db annotations (query excluded) ----
    db_feature_sets: list[FeatureSet] = []
    db_to_original: list[int] = []
    for i, ann in enumerate(database):
        if i != query_annot_index:
            db_feature_sets.append(ann.features)
            db_to_original.append(i)
    db_remap = np.array(db_to_original, dtype=np.int32)

    if hs.flann_algorithm == "exact":
        import numpy as _np

        all_descs = _np.concatenate([fs.descriptors for fs in db_feature_sets], axis=0)
        n_total = all_descs.shape[0]
        annot_of_desc = _np.empty(n_total, dtype=_np.int32)
        feat_of_desc = _np.empty(n_total, dtype=_np.int32)
        offset = 0
        for i, fs in enumerate(db_feature_sets):
            n = len(fs)
            annot_of_desc[offset : offset + n] = db_remap[i]
            feat_of_desc[offset : offset + n] = _np.arange(n, dtype=_np.int32)
            offset += n
        db_feats = FeatureSet(
            keypoints=_np.empty((n_total, 6), dtype=_np.float64),
            descriptors=all_descs,
        )
        raw_dists, raw_labels = exact_knn(query_features, db_feats, k_total)
    else:
        global_index, _annot_of_desc_local, feat_of_desc = build_global_index(
            db_feature_sets,
            algorithm=hs.flann_algorithm,
            trees=hs.flann_trees,
            random_seed=hs.flann_random_seed,
        )
        annot_of_desc = db_remap[_annot_of_desc_local]
        n_total = annot_of_desc.shape[0]

        raw_dists, raw_labels = query_index(
            global_index,
            query_features,
            k_total,
            checks=hs.flann_checks,
            cores=hs.flann_cores,
        )

    dlog.stage_global_index(db_feature_sets, annot_of_desc)
    dlog.stage_raw_dists(raw_dists, raw_labels)

    # Post-hoc distance normalisation (WBIA VEC_PSEUDO_MAX_DISTANCE_SQRD)
    # WBIA divides raw SSE by 524288 inside NeighborIndex.knn() and traces
    # the normalized (squared) distances. sqrt is applied later in
    # weight_neighbors when sqrd_dist_on=False.
    max_distance_sqrd = 2.0 * (512.0**2.0)
    dists = (np.maximum(raw_dists, 0.0) / max_distance_sqrd).astype(np.float64)

    if ctx is not None:
        ctx.trace_neighbors(query_annot_index + 1, raw_labels, dists.astype(np.float32))

    if not hs.sqrd_dist_on:
        dists = np.sqrt(dists)
    dlog.stage_dist_norm(dists)
    labels = raw_labels.astype(np.int64)

    n_qfxs = dists.shape[0]

    voting_dists_all = dists[:, :k]  # [M, K]
    norm_dists = dists[:, k:]  # [M, Knorm]

    voting_annot_all = np.full((n_qfxs, k), -1, dtype=np.int32)
    voting_feat_all = np.full((n_qfxs, k), -1, dtype=np.int32)
    for j in range(k):
        col = labels[:, j]
        valid = (col >= 0) & (col < n_total)
        voting_annot_all[valid, j] = annot_of_desc[col[valid]]
        voting_feat_all[valid, j] = feat_of_desc[col[valid]]

    dlog.stage_voting_cols(dists, labels, annot_of_desc, k, kpad, query_annot_index)

    # ---- 3. Baseline-neighbour filter (self + same-name) ----
    qname = database[query_annot_index].name_uuid
    if hs.can_match_samename or qname is None:
        same_name_set: set[int] = set()
    else:
        same_name_set = {
            i
            for i, a in enumerate(database)
            if i != query_annot_index and a.name_uuid == qname
        }

    invalid = (voting_annot_all == query_annot_index) | np.isin(
        voting_annot_all, list(same_name_set)
    )  # [M, K], bool

    dlog.stage_filter_counts(
        dists, labels, annot_of_desc, k, kpad, query_annot_index, same_name_set
    )

    if ctx is not None:
        ctx.trace_baseline_filter(query_annot_index + 1, ~invalid)

    # ---- 3b. Name-normalizer validity (WBIA normalizer_rule='name') ----
    normalizer_valid: np.ndarray | None = None
    if hs.normalizer_rule == "name":
        norm_col_labels = labels[:, -1]
        norm_ok = (norm_col_labels >= 0) & (norm_col_labels < n_total)
        norm_annots = np.full(n_qfxs, -1, dtype=np.int32)
        norm_annots[norm_ok] = annot_of_desc[norm_col_labels[norm_ok]]
        db_names = np.array([a.name_uuid for a in database], dtype=object)
        norm_names = np.full(n_qfxs, None, dtype=object)
        norm_names[norm_ok] = db_names[norm_annots[norm_ok]]
        voting_names = np.full_like(voting_annot_all, None, dtype=object)
        voting_ok = voting_annot_all >= 0
        voting_names[voting_ok] = db_names[voting_annot_all[voting_ok]]
        normalizer_valid = np.ones(n_qfxs, dtype=bool)
        for j in range(k):
            conflict = np.zeros(n_qfxs, dtype=bool)
            vn = voting_names[:, j]
            has_name = (norm_names != None) & (vn != None)  # noqa: E711
            conflict[has_name] = norm_names[has_name] == vn[has_name]
            normalizer_valid &= ~conflict
        if qname is not None:
            qname_conflict = np.zeros(n_qfxs, dtype=bool)
            has_qn = norm_names != None  # noqa: E711
            qname_conflict[has_qn] = norm_names[has_qn] == qname
            normalizer_valid &= ~qname_conflict
        normalizer_valid &= norm_ok

    # ---- 4. FG weights ----
    if hs.fg_on:
        fg_weights = [per_feature_fg(ann) for ann in database]
        q_fgw = fg_weights[query_annot_index]
    else:
        fg_weights = None
        q_fgw = None

    # ---- 5. WBIA filter chain → flat match list ----
    matches: list[Match] = []
    lnbnn_weights_list: list[float] = []
    for qfx in range(n_qfxs):
        if normalizer_valid is not None and not normalizer_valid[qfx]:
            continue
        for j in range(k):
            vdist = float(
                norm_dists[qfx, 0] if hs.normonly_on else voting_dists_all[qfx, j]
            )
            ndist = float(norm_dists[qfx, 0])
            w = ndist - vdist
            db_idx = int(voting_annot_all[qfx, j])
            if db_idx < 0 or invalid[qfx, j]:
                continue
            dfx = int(voting_feat_all[qfx, j])
            if dfx < 0:
                continue

            if hs.fg_on and q_fgw is not None:
                w *= np.sqrt(float(q_fgw[qfx]) * float(fg_weights[db_idx][dfx]))

            if hs.bar_l2_on:
                w *= 1.0 - vdist

            if hs.ratio_thresh is not None:
                ratio = vdist / ndist if ndist > 0 else 1.0
                if ratio > hs.ratio_thresh:
                    continue
                w *= 1.0 - ratio

            if hs.const_on:
                w *= 1.0

            lnbnn_weights_list.append(w)
            matches.append(
                Match(
                    qfx=qfx,
                    daid=db_idx,
                    dfx=dfx,
                    dist=w,
                    name_uuid=database[db_idx].name_uuid,
                )
            )

    dlog.stage_lnbnn_weights(lnbnn_weights_list)

    if ctx is not None:
        ctx.trace_neighbor_weights(query_annot_index + 1, lnbnn_weights_list)

    # ---- 6. Score ----
    _wbia_methods = {"csum_wbia", "nsum_wbia", "sumamech"}
    if hs.score_method in _wbia_methods:
        annot_uuids = [a.annot_uuid for a in database]
        annot_name_map = {
            a.annot_uuid: a.name_uuid for a in database if a.name_uuid is not None
        }
        qk: np.ndarray | None = (
            query_features.keypoints if hs.rotation_invariance else None
        )
        csum_annot, canonical = score_matches_with_names(
            matches, annot_uuids, annot_name_map, hs.score_method, qk
        )
        scored = _canonical_to_scoredmatches(canonical, matches, database, annot_uuids)
        _annot_to_matches: dict[int, tuple[float, int]] = {}
        for sm in scored:
            daid = next(
                i for i, a in enumerate(database) if a.annot_uuid == sm.annot_uuid
            )
            c = csum_annot.get(sm.annot_uuid, 0.0)
            _annot_to_matches[daid] = (c, sm.num_matches)
        dlog.stage_match_to_annot(_annot_to_matches)
    else:
        scored = score_matches(matches, database, hs.score_method)
        _annot_to_matches = {}
        for sm in scored:
            daid = next(
                i for i, a in enumerate(database) if a.annot_uuid == sm.annot_uuid
            )
            csum_val = (
                sm.score if hs.score_method == "csum" else sm.score * sm.num_matches
            )
            _annot_to_matches[daid] = (csum_val, sm.num_matches)
        dlog.stage_match_to_annot(_annot_to_matches)

    _uuid_to_daid = {ann.annot_uuid: i + 1 for i, ann in enumerate(database)}
    _name_to_nid: dict = {}
    _next_nid = 1
    for ann in database:
        if ann.name_uuid is not None and ann.name_uuid not in _name_to_nid:
            _name_to_nid[ann.name_uuid] = _next_nid
            _next_nid += 1

    _trace_qaid = query_annot_index + 1
    _trace_qnid = (
        _name_to_nid.get(database[query_annot_index].name_uuid, -1)
        if database[query_annot_index].name_uuid
        else -1
    )

    if ctx is not None:
        _pre_daids = np.array(
            [_uuid_to_daid[sm.annot_uuid] for sm in scored], dtype=np.int64
        )
        _pre_dnids = np.array(
            [
                _name_to_nid.get(sm.name_uuid, -1) if sm.name_uuid else -1
                for sm in scored
            ],
            dtype=np.int64,
        )
        _pre_sort = np.argsort(_pre_daids)
        _pre_daids = _pre_daids[_pre_sort]
        _pre_dnids = _pre_dnids[_pre_sort]
        _fm_list_pre = _fm_arrays_for_scored(
            [scored[i] for i in _pre_sort], matches, database
        )
        ctx.trace_chipmatches(
            "chipmatches_pre_sv",
            _trace_qaid,
            _trace_qnid,
            _pre_daids,
            None,
            None,
            None,
            None,
            _fm_list_pre,
            fsv_list=None,
        )

    # ---- 7. Spatial verification ----
    if hs.sv_on:
        _prescored = scored
        if hs.prescore_method != hs.score_method:
            _prescored = _prescore_candidates(matches, database, hs.prescore_method)
        sver_candidates = make_sver_shortlist(
            _prescored,
            n_names=hs.sv_n_name_shortlist,
            n_annots_per_name=hs.sv_n_annot_per_name,
        )
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
        )

        if sv_results:
            match_lookup: dict[tuple[int, int, int], Match] = {}
            for m in matches:
                match_lookup[(m.daid, m.qfx, m.dfx)] = m

            sv_matches: list[Match] = []
            for annot_uuid, (
                inlier_qfxs,
                inlier_dfxs,
                weights,
            ) in sv_results.items():
                daid = next(
                    i for i, a in enumerate(database) if a.annot_uuid == annot_uuid
                )
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
            if hs.score_method in _wbia_methods:
                qk = query_features.keypoints if hs.rotation_invariance else None
                _, sv_canonical = score_matches_with_names(
                    sv_matches, annot_uuids, annot_name_map, hs.score_method, qk
                )
                sv_scored = _canonical_to_scoredmatches(
                    sv_canonical, sv_matches, database, annot_uuids
                )
            else:
                sv_scored = score_matches(sv_matches, database, hs.score_method)

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

    if ctx is not None:
        _post_daids = np.array(
            [_uuid_to_daid[sm.annot_uuid] for sm in scored], dtype=np.int64
        )
        _post_dnids = np.array(
            [
                _name_to_nid.get(sm.name_uuid, -1) if sm.name_uuid else -1
                for sm in scored
            ],
            dtype=np.int64,
        )
        _post_ascores = np.array([sm.score for sm in scored], dtype=np.float64)
        _post_slist = np.array([sm.score for sm in scored], dtype=np.float64)
        _post_sort = np.argsort(_post_daids)
        _post_daids = _post_daids[_post_sort]
        _post_dnids = _post_dnids[_post_sort]
        _post_ascores = _post_ascores[_post_sort]
        _post_slist = _post_slist[_post_sort]
        _fm_list_post = _fm_arrays_for_scored(
            [scored[i] for i in _post_sort], matches, database
        )
        ctx.trace_chipmatches(
            "chipmatches_post_sv",
            _trace_qaid,
            _trace_qnid,
            _post_daids,
            _post_dnids,
            None,
            None,
            None,
            _fm_list_post,
        )

    dlog.stage_final(scored)

    if ctx is not None:
        _daids = np.array(
            [_uuid_to_daid[sm.annot_uuid] for sm in scored], dtype=np.int64
        )
        _dnids = np.array(
            [
                _name_to_nid.get(sm.name_uuid, -1) if sm.name_uuid else -1
                for sm in scored
            ],
            dtype=np.int64,
        )
        _ascores = np.array([sm.score for sm in scored], dtype=np.float64)
        _slist = np.array([sm.score for sm in scored], dtype=np.float64)
        _final_sort = np.argsort(_daids)
        _daids = _daids[_final_sort]
        _dnids = _dnids[_final_sort]
        _ascores = _ascores[_final_sort]
        _slist = _slist[_final_sort]
        _fm_list_final = _fm_arrays_for_scored(
            [scored[i] for i in _final_sort], matches, database
        )
        ctx.trace_final_scores(
            qaid=_trace_qaid,
            qnid=_trace_qnid,
            score_method=_trace_score_method(hs.score_method),
            daid_list=_daids,
            dnid_list=_dnids,
            annot_score_list=_ascores,
            name_score_list=_ascores,
            score_list=_slist,
            fm_list=_fm_list_final,
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
        _, canonical = score_matches_with_names(
            matches, annot_uuids, annot_name_map, score_method
        )
        return _canonical_to_scoredmatches(canonical, matches, database, annot_uuids)
    else:
        return score_matches(matches, database, score_method)
