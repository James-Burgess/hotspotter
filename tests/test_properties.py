"""Property-based invariants and edge-case tests for scoring/pipeline.

Tests mathematical invariants that must hold for any input, plus
edge cases around feature filtering and pipeline robustness.
"""

import uuid

import numpy as np
import pytest

from hotspotter.config import HotSpotterConfig
from hotspotter.data import AnnotatedImage, FeatureSet, Match, ScoredMatch
from hotspotter.name_scoring import compute_fmech_score, group_matches_by_name
from hotspotter.pipeline import _filter_query_features
from hotspotter.scoring import (
    baseline_filter,
    build_matches,
    score_matches,
    weight_neighbors_lnbnn,
)


def _make_features(n: int = 10) -> FeatureSet:
    return FeatureSet(
        keypoints=np.column_stack(
            [
                np.random.randn(n).astype(np.float32),
                np.random.randn(n).astype(np.float32),
                np.ones(n, dtype=np.float32) * 2.0,
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            ]
        ),
        descriptors=np.random.randint(0, 256, (n, 128), dtype=np.uint8),
    )


def _make_annot(
    name_uuid: uuid.UUID | None,
    n_feats: int = 5,
    image_uuid: uuid.UUID | None = None,
    scales: np.ndarray | None = None,
) -> AnnotatedImage:
    nf = len(scales) if scales is not None else n_feats
    kpts = np.column_stack(
        [
            np.random.randn(nf).astype(np.float32),
            np.random.randn(nf).astype(np.float32),
            np.ones(nf, dtype=np.float32) * 2.0 if scales is None else scales,
            np.zeros(nf, dtype=np.float32),
            np.zeros(nf, dtype=np.float32),
            np.zeros(nf, dtype=np.float32),
        ]
    )
    return AnnotatedImage(
        annot_uuid=uuid.uuid4(),
        name_uuid=name_uuid,
        image=np.zeros((50, 50), dtype=np.uint8),
        features=FeatureSet(
            keypoints=kpts,
            descriptors=np.zeros((nf, 128), dtype=np.uint8),
        ),
        bbox=(0, 0, 50, 50),
        image_uuid=image_uuid,
    )


# ---- weight_neighbors_lnbnn invariants ----


class TestLnbnnInvariants:
    @pytest.mark.parametrize("n_qfxs", [1, 5, 20])
    @pytest.mark.parametrize("k_total", [2, 5, 10])
    def test_weights_always_nonnegative(self, n_qfxs, k_total):
        voting = np.abs(np.random.randn(n_qfxs, k_total)).astype(np.float64) * 0.5
        norm = np.abs(np.random.randn(n_qfxs, 3)).astype(np.float64) * 0.5 + 0.3
        w = weight_neighbors_lnbnn(voting, norm)
        assert np.all(w >= 0.0)

    @pytest.mark.parametrize("n_qfxs", [1, 5, 20])
    @pytest.mark.parametrize("k_total", [2, 5, 10])
    def test_weights_zero_when_vdist_gt_ndist(self, n_qfxs, k_total):
        voting = np.abs(np.random.randn(n_qfxs, k_total)).astype(np.float64) * 0.5 + 0.8
        norm = np.abs(np.random.randn(n_qfxs, 3)).astype(np.float64) * 0.1 + 0.05
        w = weight_neighbors_lnbnn(voting, norm)
        assert np.allclose(w, 0.0)

    def test_lnbnn_ratio_clamps(self):
        voting = np.array([[0.2, 0.6]], dtype=np.float64)
        norm = np.array([[0.4]], dtype=np.float64)
        w = weight_neighbors_lnbnn(voting, norm, lnbnn_ratio=0.5)
        assert w[0, 0] > 0
        assert w[0, 1] == 0.0

    def test_const_mode(self):
        voting = np.abs(np.random.randn(3, 4)).astype(np.float64)
        norm = np.abs(np.random.randn(3, 2)).astype(np.float64) + 0.5
        w = weight_neighbors_lnbnn(voting, norm, const_on=True)
        assert np.all((w == 0.0) | (w == 1.0))

    def test_weights_non_increasing_per_row(self):
        voting = np.abs(np.random.randn(5, 4)).astype(np.float64)
        voting.sort(axis=1)
        norm = np.abs(np.random.randn(5, 1)).astype(np.float64) + 1.0
        w = weight_neighbors_lnbnn(voting, norm)
        for row in w:
            for j in range(len(row) - 1):
                assert row[j] >= row[j + 1] - 1e-12


# ---- nsum <= csum invariant ----


class TestNsumLeCsum:
    def test_nsum_le_total_csum(self):
        name = uuid.uuid4()
        matches = []
        for qfx in range(5):
            for daid in range(3):
                matches.append(
                    Match(
                        qfx=qfx,
                        daid=daid,
                        dfx=qfx,
                        dist=float(qfx + 1),
                        name_uuid=name,
                    )
                )
        by_name = group_matches_by_name(matches)
        name_scores = compute_fmech_score(by_name)
        nsum_val = name_scores[name]
        csum_total = sum(m.dist for m in matches)
        assert nsum_val <= csum_total + 1e-10


# ---- baseline_filter invariants ----


class TestBaselineFilterInvariants:
    def test_always_marks_self_invalid(self):
        db = [_make_annot(uuid.uuid4()) for _ in range(5)]
        voting = np.array([[0, 1, 2]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting, db, 0)
        assert invalid[0, 0]

    def test_never_marks_different_name_invalid(self):
        name_a = uuid.uuid4()
        name_b = uuid.uuid4()
        db = [_make_annot(name_a), _make_annot(name_b)]
        voting = np.array([[1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting, db, 0)
        assert not invalid[0, 0]

    def test_marks_same_name_invalid_when_disabled(self):
        shared = uuid.uuid4()
        db = [_make_annot(shared), _make_annot(shared)]
        voting = np.array([[1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting, db, 0, can_match_samename=False)
        assert invalid[0, 0]

    def test_keeps_same_name_when_enabled(self):
        shared = uuid.uuid4()
        db = [_make_annot(shared), _make_annot(shared)]
        voting = np.array([[1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting, db, 0, can_match_samename=True)
        assert not invalid[0, 0]

    def test_marks_same_image_invalid_when_disabled(self):
        img = uuid.uuid4()
        db = [
            _make_annot(uuid.uuid4(), image_uuid=img),
            _make_annot(uuid.uuid4(), image_uuid=img),
        ]
        voting = np.array([[1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting, db, 0, can_match_sameimg=False)
        assert invalid[0, 0]

    def test_keeps_same_image_when_enabled(self):
        img = uuid.uuid4()
        db = [
            _make_annot(uuid.uuid4(), image_uuid=img),
            _make_annot(uuid.uuid4(), image_uuid=img),
        ]
        voting = np.array([[1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting, db, 0, can_match_sameimg=True)
        assert not invalid[0, 0]


# ---- score_matches invariants ----


class TestScoreMatchesInvariants:
    def test_scores_are_nonnegative(self):
        name = uuid.uuid4()
        db = [_make_annot(name, n_feats=3), _make_annot(name, n_feats=3)]
        matches = [
            Match(qfx=0, daid=0, dfx=0, dist=1.0, name_uuid=name),
            Match(qfx=1, daid=0, dfx=1, dist=0.5, name_uuid=name),
            Match(qfx=2, daid=1, dfx=0, dist=0.0, name_uuid=name),
        ]
        for method in ("nsum", "csum"):
            scored = score_matches(matches, db, score_method=method)
            for sm in scored:
                assert sm.score >= 0.0

    def test_nsum_score_between_zero_and_max_weight(self):
        name = uuid.uuid4()
        db = [_make_annot(name, n_feats=3)]
        max_w = 5.0
        matches = [
            Match(qfx=0, daid=0, dfx=0, dist=max_w, name_uuid=name),
            Match(qfx=1, daid=0, dfx=1, dist=1.0, name_uuid=name),
        ]
        scored = score_matches(matches, db, score_method="nsum")
        assert scored[0].score >= 0.0
        assert scored[0].score <= max_w


# ---- _filter_query_features edge cases ----


class TestFilterQueryFeatures:
    def test_noop_when_all_thresholds_none(self):
        db = [_make_annot(uuid.uuid4(), n_feats=5)]
        hs = HotSpotterConfig(
            minscale_thresh=None, maxscale_thresh=None, fgw_thresh=None
        )
        result = _filter_query_features(db, 0, hs)
        assert len(result) == 5
        np.testing.assert_array_equal(result.keypoints, db[0].features.keypoints)

    def test_minscale_filters_small_scales(self):
        scales = np.array([0.5, 1.0, 3.0, 5.0, 0.2], dtype=np.float32)
        db = [_make_annot(uuid.uuid4(), n_feats=5, scales=scales)]
        hs = HotSpotterConfig(
            minscale_thresh=2.0, maxscale_thresh=None, fgw_thresh=None
        )
        result = _filter_query_features(db, 0, hs)
        assert len(result) == 2

    def test_maxscale_filters_large_scales(self):
        scales = np.array([0.5, 1.0, 3.0, 5.0, 0.2], dtype=np.float32)
        db = [_make_annot(uuid.uuid4(), n_feats=5, scales=scales)]
        hs = HotSpotterConfig(
            minscale_thresh=None, maxscale_thresh=2.0, fgw_thresh=None
        )
        result = _filter_query_features(db, 0, hs)
        assert len(result) == 3

    def test_combining_thresholds(self):
        scales = np.array([0.5, 2.0, 3.0, 5.0, 0.2], dtype=np.float32)
        db = [_make_annot(uuid.uuid4(), n_feats=5, scales=scales)]
        hs = HotSpotterConfig(minscale_thresh=1.0, maxscale_thresh=4.0, fgw_thresh=None)
        result = _filter_query_features(db, 0, hs)
        assert len(result) == 2

    def test_preserves_keypoint_descriptor_alignment(self):
        scales = np.array([1.0, 0.5, 3.0], dtype=np.float32)
        kpts = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                [2.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        descs = np.zeros((3, 128), dtype=np.uint8)
        descs[0, 0] = 1
        descs[1, 0] = 2
        descs[2, 0] = 3
        ann = AnnotatedImage(
            annot_uuid=uuid.uuid4(),
            name_uuid=uuid.uuid4(),
            image=np.zeros((50, 50), dtype=np.uint8),
            features=FeatureSet(keypoints=kpts, descriptors=descs),
            bbox=(0, 0, 50, 50),
        )
        db = [ann]
        hs = HotSpotterConfig(
            minscale_thresh=1.0, maxscale_thresh=None, fgw_thresh=None
        )
        result = _filter_query_features(db, 0, hs)
        assert len(result) == 2
        np.testing.assert_array_equal(result.keypoints[0], kpts[0])
        np.testing.assert_array_equal(result.keypoints[1], kpts[2])
        assert result.descriptors[0, 0] == 1
        assert result.descriptors[1, 0] == 3

    def test_filter_all_features_raises_assertion(self):
        scales = np.array([0.1, 0.2], dtype=np.float32)
        db = [_make_annot(uuid.uuid4(), scales=scales)]
        hs = HotSpotterConfig(
            minscale_thresh=10.0, maxscale_thresh=None, fgw_thresh=None
        )
        with pytest.raises(AssertionError, match="All query features filtered"):
            _filter_query_features(db, 0, hs)

    def test_fgw_thresh_filters_low_fg_features(self):
        kpts = np.array(
            [
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        ann = AnnotatedImage(
            annot_uuid=uuid.uuid4(),
            name_uuid=uuid.uuid4(),
            image=np.zeros((50, 50), dtype=np.uint8),
            features=FeatureSet(
                keypoints=kpts,
                descriptors=np.zeros((3, 128), dtype=np.uint8),
            ),
            bbox=(0, 0, 50, 50),
        )
        db = [ann]
        hs = HotSpotterConfig(
            minscale_thresh=None, maxscale_thresh=None, fgw_thresh=0.5
        )
        result = _filter_query_features(db, 0, hs)
        assert len(result) == 3


# ---- Python-level invariants cross-checked ----


class TestCrossModuleInvariants:
    def test_scoredmatch_correspondences_are_unique(self):
        from hotspotter.data import ScoredMatch

        sm = ScoredMatch(
            annot_uuid=uuid.uuid4(),
            name_uuid=None,
            score=1.0,
            num_matches=5,
            correspondences=[(0, 1), (1, 2), (2, 3)],
        )
        pairs = set(sm.correspondences)
        assert len(pairs) == len(sm.correspondences)

    def test_feature_set_length(self):
        fs = _make_features(7)
        assert len(fs) == 7
        assert fs.keypoints.shape[0] == 7
        assert fs.descriptors.shape[0] == 7

    def test_zero_weight_build_matches_produces_empty(self):
        ann = _make_annot(uuid.uuid4(), n_feats=3)
        db = [ann]
        weights = np.zeros((1, 2), dtype=np.float64)
        voting_annot = np.array([[0, 0]], dtype=np.int32)
        voting_feat = np.array([[0, 1]], dtype=np.int32)
        invalid = np.zeros((1, 2), dtype=bool)
        matches = build_matches(
            weights, voting_annot, voting_feat, invalid, db, k=1, kpad=1
        )
        assert len(matches) == 0

    def test_invalid_entries_produce_no_matches(self):
        ann = _make_annot(uuid.uuid4(), n_feats=3)
        db = [ann]
        weights = np.ones((1, 2), dtype=np.float64)
        voting_annot = np.array([[0, 0]], dtype=np.int32)
        voting_feat = np.array([[0, 1]], dtype=np.int32)
        invalid = np.ones((1, 2), dtype=bool)
        matches = build_matches(
            weights, voting_annot, voting_feat, invalid, db, k=1, kpad=1
        )
        assert len(matches) == 0

    def test_voting_columns_size_matches_weights(self):
        k, kpad = 3, 1
        voting = np.random.randn(10, k + kpad)
        norm = np.random.randn(10, 3) + 1.0
        w = weight_neighbors_lnbnn(voting, norm)
        assert w.shape == (10, k + kpad)

    def test_build_matches_distinct_columns_produce_distinct_matches(self):
        ann = _make_annot(uuid.uuid4(), n_feats=5)
        db = [ann]
        weights = np.ones((1, 2), dtype=np.float64)
        voting_annot = np.array([[0, 0]], dtype=np.int32)
        voting_feat = np.array([[0, 3]], dtype=np.int32)
        invalid = np.zeros((1, 2), dtype=bool)
        matches = build_matches(
            weights, voting_annot, voting_feat, invalid, db, k=1, kpad=1
        )
        assert len(matches) == 2
        assert matches[0].dfx != matches[1].dfx
