import uuid

import numpy as np
import pytest

from hotspotter.data import AnnotatedImage, FeatureSet
from hotspotter.scoring import (
    baseline_filter,
    build_matches,
    compute_normalizer_validity,
    score_matches,
    weight_neighbors_lnbnn,
)


def _make_annot(
    name_uuid: uuid.UUID | None, n_feats: int = 5, image_uuid: uuid.UUID | None = None
) -> AnnotatedImage:
    return AnnotatedImage(
        annot_uuid=uuid.uuid4(),
        name_uuid=name_uuid,
        image=np.zeros((50, 50), dtype=np.uint8),
        features=FeatureSet(
            keypoints=np.zeros((n_feats, 6), dtype=np.float32),
            descriptors=np.zeros((n_feats, 128), dtype=np.uint8),
        ),
        bbox=(0, 0, 50, 50),
        image_uuid=image_uuid,
    )


class TestBaselineFilter:
    def test_removes_self(self):
        name = uuid.uuid4()
        ann = _make_annot(name)
        db = [ann]
        voting_annot = np.array([[0, 1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting_annot, db, 0)
        assert invalid[0, 0]

    def test_removes_same_name(self):
        name = uuid.uuid4()
        ann1 = _make_annot(name)
        ann2 = _make_annot(name)
        db = [ann1, ann2]
        voting_annot = np.array([[0, 1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting_annot, db, 0, can_match_samename=False)
        assert invalid[0, 0]  # self
        assert invalid[0, 1]  # same name

    def test_different_name_kept(self):
        name_a = uuid.uuid4()
        name_b = uuid.uuid4()
        ann1 = _make_annot(name_a)
        ann2 = _make_annot(name_b)
        db = [ann1, ann2]
        voting_annot = np.array([[0, 1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting_annot, db, 0)
        assert invalid[0, 0]  # self
        assert not invalid[0, 1]  # different name

    def test_same_image_excluded(self):
        img = uuid.uuid4()
        ann1 = _make_annot(uuid.uuid4(), image_uuid=img)
        ann2 = _make_annot(uuid.uuid4(), image_uuid=img)
        db = [ann1, ann2]
        voting_annot = np.array([[0, 1]], dtype=np.int32)
        invalid, _, _ = baseline_filter(voting_annot, db, 0, can_match_sameimg=False)
        assert invalid[0, 0]  # self
        assert invalid[0, 1]  # same image


class TestWeightNeighborsLnbnn:
    def test_basic(self):
        voting = np.array([[0.2, 0.4, 0.6]], dtype=np.float64)
        norm = np.array([[0.8]], dtype=np.float64)
        w = weight_neighbors_lnbnn(voting, norm)
        assert np.allclose(w[0], [0.6, 0.4, 0.2])

    def test_ratio_clamp(self):
        voting = np.array([[0.2, 0.9, 1.0]], dtype=np.float64)
        norm = np.array([[1.0]], dtype=np.float64)
        w = weight_neighbors_lnbnn(voting, norm, lnbnn_ratio=0.8)
        assert w[0, 0] > 0
        assert w[0, 1] == pytest.approx(0)
        assert w[0, 2] == pytest.approx(0)

    def test_zero_dist(self):
        voting = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        norm = np.array([[0.0]], dtype=np.float64)
        w = weight_neighbors_lnbnn(voting, norm)
        assert np.allclose(w, 0.0)

    def test_bar_l2(self):
        voting = np.array([[0.2, 0.4]], dtype=np.float64)
        norm = np.array([[0.8]], dtype=np.float64)
        w = weight_neighbors_lnbnn(voting, norm, bar_l2_on=True)
        assert w[0, 0] == pytest.approx(0.6 * 0.8)
        assert w[0, 1] == pytest.approx(0.4 * 0.6)

    def test_ratio_thresh(self):
        voting = np.array([[0.2, 0.6]], dtype=np.float64)
        norm = np.array([[0.5]], dtype=np.float64)
        w = weight_neighbors_lnbnn(voting, norm, ratio_thresh=0.9)
        assert w[0, 1] == pytest.approx(0.0)

    def test_normonly(self):
        voting = np.array([[0.2, 0.4]], dtype=np.float64)
        norm = np.array([[0.8]], dtype=np.float64)
        w = weight_neighbors_lnbnn(voting, norm, normonly_on=True)
        assert np.allclose(w, 0.0)

    def test_max_clamp(self):
        voting = np.array([[1.5, 0.3]], dtype=np.float64)
        norm = np.array([[1.0]], dtype=np.float64)
        w = weight_neighbors_lnbnn(voting, norm)
        assert w[0, 0] == pytest.approx(0.0)
        assert w[0, 1] == pytest.approx(0.7)


class TestBuildMatches:
    def test_basic(self):
        name = uuid.uuid4()
        db = [_make_annot(name, n_feats=2), _make_annot(name, n_feats=2)]
        weights = np.array([[0.0, 0.5]], dtype=np.float64)
        voting_annot = np.array([[0, 1]], dtype=np.int32)
        voting_feat = np.array([[0, 0]], dtype=np.int32)
        invalid = np.zeros((1, 2), dtype=bool)
        matches = build_matches(
            weights, voting_annot, voting_feat, invalid, db, k=1, kpad=1
        )
        assert len(matches) == 1
        assert matches[0].dfx == 0

    def test_all_columns_processed(self):
        name = uuid.uuid4()
        db = [_make_annot(name, n_feats=3), _make_annot(name, n_feats=3)]
        weights = np.array([[0.5, 0.3, 0.1]], dtype=np.float64)
        voting_annot = np.array([[0, 0, 0]], dtype=np.int32)
        voting_feat = np.array([[0, 1, 2]], dtype=np.int32)
        invalid = np.zeros((1, 3), dtype=bool)
        matches = build_matches(
            weights, voting_annot, voting_feat, invalid, db, k=2, kpad=1
        )
        assert len(matches) == 3
        assert {m.dfx for m in matches} == {0, 1, 2}


class TestScoreMatches:
    def test_nsum(self):
        db = [
            _make_annot(uuid.uuid4(), n_feats=5),
            _make_annot(uuid.uuid4(), n_feats=5),
        ]
        name = uuid.uuid4()
        from hotspotter.data import Match

        matches = [
            Match(qfx=0, daid=0, dfx=1, dist=1.0, name_uuid=name),
            Match(qfx=1, daid=0, dfx=2, dist=0.5, name_uuid=name),
            Match(qfx=2, daid=1, dfx=0, dist=0.2, name_uuid=uuid.uuid4()),
        ]
        scored = score_matches(matches, db, score_method="nsum")
        assert len(scored) == 2
        for s in scored:
            if s.annot_uuid == db[0].annot_uuid:
                assert s.score == 0.75
                assert len(s.correspondences) == 2
            else:
                assert s.score == 0.2
                assert len(s.correspondences) == 1

    def test_csum(self):
        db = [_make_annot(uuid.uuid4(), n_feats=5)]
        from hotspotter.data import Match

        matches = [
            Match(qfx=0, daid=0, dfx=0, dist=1.0, name_uuid=None),
            Match(qfx=1, daid=0, dfx=1, dist=0.5, name_uuid=None),
        ]
        scored = score_matches(matches, db, score_method="csum")
        assert len(scored) == 1
        assert scored[0].score == 1.5
        assert len(scored[0].correspondences) == 2


class TestComputeNormalizerValidity:
    def test_all_valid_when_unique_names(self):
        name_a = uuid.uuid4()
        name_b = uuid.uuid4()
        name_c = uuid.uuid4()
        name_d = uuid.uuid4()
        db = [
            _make_annot(name_a),
            _make_annot(name_b),
            _make_annot(name_c),
        ]
        n_total = 6
        annot_of_desc = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        voting_annot = np.array([[1, 2]], dtype=np.int32)
        labels = np.array([[2, 4, 0]], dtype=np.int32)
        result = compute_normalizer_validity(
            voting_annot, labels, annot_of_desc, n_total, db, k=1, kpad=1, qname=name_d
        )
        assert result[0]

    def test_invalid_when_normalizer_shares_name_with_voting(self):
        shared = uuid.uuid4()
        db = [
            _make_annot(shared),
            _make_annot(shared),
            _make_annot(uuid.uuid4()),
        ]
        n_total = 6
        annot_of_desc = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        voting_annot = np.array([[1, 2]], dtype=np.int32)
        labels = np.array([[0, 4, 3]], dtype=np.int32)
        result = compute_normalizer_validity(
            voting_annot, labels, annot_of_desc, n_total, db, k=1, kpad=1, qname=None
        )
        assert not result[0]

    def test_invalid_when_normalizer_shares_name_with_query(self):
        shared = uuid.uuid4()
        db = [
            _make_annot(shared),
            _make_annot(uuid.uuid4()),
        ]
        n_total = 4
        annot_of_desc = np.array([0, 0, 1, 1], dtype=np.int32)
        voting_annot = np.array([[1, -1]], dtype=np.int32)
        labels = np.array([[2, -1, 0]], dtype=np.int32)
        result = compute_normalizer_validity(
            voting_annot, labels, annot_of_desc, n_total, db, k=1, kpad=1, qname=shared
        )
        assert not result[0]

    def test_valid_when_normalizer_has_different_name(self):
        db = [
            _make_annot(uuid.uuid4()),
            _make_annot(uuid.uuid4()),
        ]
        n_total = 4
        annot_of_desc = np.array([0, 0, 1, 1], dtype=np.int32)
        voting_annot = np.array([[1]], dtype=np.int32)
        labels = np.array([[2, 0]], dtype=np.int32)
        result = compute_normalizer_validity(
            voting_annot,
            labels,
            annot_of_desc,
            n_total,
            db,
            k=1,
            kpad=0,
            qname=uuid.uuid4(),
        )
        assert result[0]

    def test_invalid_when_normalizer_out_of_range(self):
        db = [_make_annot(uuid.uuid4())]
        n_total = 2
        annot_of_desc = np.array([0, 0], dtype=np.int32)
        voting_annot = np.array([[0]], dtype=np.int32)
        labels = np.array([[0, 99]], dtype=np.int32)
        result = compute_normalizer_validity(
            voting_annot, labels, annot_of_desc, n_total, db, k=1, kpad=0, qname=None
        )
        assert not result[0]

    def test_conflict_in_second_voting_column(self):
        shared = uuid.uuid4()
        other = uuid.uuid4()
        db = [
            _make_annot(uuid.uuid4()),
            _make_annot(shared),
            _make_annot(other),
            _make_annot(shared),
        ]
        n_total = 8
        annot_of_desc = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        voting_annot = np.array([[2, 3]], dtype=np.int32)
        labels = np.array([[4, 6, 2]], dtype=np.int32)
        result = compute_normalizer_validity(
            voting_annot,
            labels,
            annot_of_desc,
            n_total,
            db,
            k=1,
            kpad=1,
            qname=uuid.uuid4(),
        )
        assert not result[0]

    def test_multiple_features_batch(self):
        db = [
            _make_annot(uuid.uuid4()),
            _make_annot(uuid.uuid4()),
            _make_annot(uuid.uuid4()),
            _make_annot(uuid.uuid4()),
        ]
        n_total = 12
        annot_of_desc = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32)
        voting_annot = np.array([[1, 2], [2, 3]], dtype=np.int32)
        labels = np.array([[3, 6, 9], [6, 9, 3]], dtype=np.int32)
        result = compute_normalizer_validity(
            voting_annot, labels, annot_of_desc, n_total, db, k=2, kpad=0, qname=None
        )
        assert result[0]
        assert result[1]
