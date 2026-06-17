"""Tests for wbia_core.scoring (focused, no faiss needed)."""

import uuid

import numpy as np
import pytest

from wbia_core.data import AnnotatedImage, FeatureSet
from wbia_core.scoring import (
    build_matches,
    filter_self_matches,
    score_matches,
    weight_neighbors_lnbnn,
)


def _make_annot(name_uuid: uuid.UUID | None, n_feats: int = 5) -> AnnotatedImage:
    return AnnotatedImage(
        annot_uuid=uuid.uuid4(),
        name_uuid=name_uuid,
        image=np.zeros((50, 50), dtype=np.uint8),
        features=FeatureSet(
            keypoints=np.zeros((n_feats, 6), dtype=np.float32),
            descriptors=np.zeros((n_feats, 128), dtype=np.uint8),
        ),
        bbox=(0, 0, 50, 50),
    )


class TestFilterSelfMatches:
    def test_removes_self(self):
        name = uuid.uuid4()
        ann = _make_annot(name)
        db = [ann]

        distances = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
        labels = np.array([[0, 1], [0, 1]], dtype=np.int64)

        d, l = filter_self_matches(distances, labels, db, 0)
        # self (column 0) is set to inf/-1 but columns NOT re-sorted
        assert np.isinf(d[0, 0])
        assert l[0, 0] == -1

    def test_removes_same_name(self):
        name = uuid.uuid4()
        ann1 = _make_annot(name)
        ann2 = _make_annot(name)
        db = [ann1, ann2]

        distances = np.array([[0.5, 0.3]], dtype=np.float32)
        labels = np.array([[0, 1]], dtype=np.int64)

        d, l = filter_self_matches(distances, labels, db, 0)
        assert l[0, 0] == -1  # ann1 is self
        assert l[0, 1] == -1  # ann2 has same name
        assert np.all(np.isinf(d[0]))

    def test_different_name_kept(self):
        name_a = uuid.uuid4()
        name_b = uuid.uuid4()
        ann1 = _make_annot(name_a)
        ann2 = _make_annot(name_b)
        db = [ann1, ann2]

        distances = np.array([[0.5, 0.3]], dtype=np.float32)
        labels = np.array([[0, 1]], dtype=np.int64)

        d, l = filter_self_matches(distances, labels, db, 0)
        # Self (col 0) removed, other name (col 1) kept
        assert l[0, 0] == -1
        assert l[0, 1] == 1
        assert np.isinf(d[0, 0])
        assert not np.isinf(d[0, 1])


class TestWeightNeighborsLnbnn:
    def test_basic(self):
        distances = np.array(
            [[0.2, 0.4, 0.6, 0.8]], dtype=np.float32
        )  # k=3, normalizer = col 3
        labels = np.zeros((1, 4), dtype=np.int64)
        w = weight_neighbors_lnbnn(distances, labels, k=3)
        # WBIA formula: norm - nn_dist
        assert np.allclose(w[0], [0.6, 0.4, 0.2])

    def test_ratio_clamp(self):
        distances = np.array([[0.2, 0.9, 1.0, 1.0]], dtype=np.float32)
        labels = np.zeros((1, 4), dtype=np.int64)
        w = weight_neighbors_lnbnn(distances, labels, k=3, lnbnn_ratio=0.8)
        assert w[0, 0] > 0
        assert w[0, 1] == pytest.approx(0)
        assert w[0, 2] == pytest.approx(0)

    def test_zero_dist(self):
        distances = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        labels = np.zeros((1, 4), dtype=np.int64)
        w = weight_neighbors_lnbnn(distances, labels, k=3)
        # All identical: norm - nn = 0
        assert np.allclose(w, 0.0)


class TestBuildMatches:
    def test_basic(self):
        name = uuid.uuid4()
        db = [_make_annot(name, n_feats=2)]

        weights = np.array([[0.5, 0.0], [0.3, 0.0]], dtype=np.float64)
        labels = np.array([[0, 1], [0, 1]], dtype=np.int64)
        # local_labels: each match points to feature index 0
        local_labels = np.array([[0, 0], [0, 0]], dtype=np.int64)

        matches = build_matches(weights, labels[:, :1], local_labels[:, :1], db)
        assert len(matches) == 2
        assert all(m.dfx == 0 for m in matches)


class TestScoreMatches:
    def test_nsum(self):
        db = [
            _make_annot(uuid.uuid4(), n_feats=5),
            _make_annot(uuid.uuid4(), n_feats=5),
        ]
        name = uuid.uuid4()
        from wbia_core.data import Match

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
        from wbia_core.data import Match

        matches = [
            Match(qfx=0, daid=0, dfx=0, dist=1.0, name_uuid=None),
            Match(qfx=1, daid=0, dfx=1, dist=0.5, name_uuid=None),
        ]
        scored = score_matches(matches, db, score_method="csum")
        assert len(scored) == 1
        assert scored[0].score == 1.5
        assert len(scored[0].correspondences) == 2
