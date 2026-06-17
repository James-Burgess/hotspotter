"""Tests for wbia_core.data."""

import uuid

import numpy as np
import pytest

from wbia_core.data import AnnotatedImage, FeatureSet, Match, ScoredMatch


def _make_features(n: int = 10) -> FeatureSet:
    return FeatureSet(
        keypoints=np.random.randn(n, 6).astype(np.float32),
        descriptors=np.random.randint(0, 256, (n, 128), dtype=np.uint8),
    )


class TestFeatureSet:
    def test_valid(self):
        fs = _make_features(5)
        assert len(fs) == 5

    def test_keypoints_wrong_shape(self):
        with pytest.raises(ValueError, match="keypoints must be"):
            FeatureSet(
                keypoints=np.zeros((5, 3), dtype=np.float32),
                descriptors=np.zeros((5, 128), dtype=np.uint8),
            )

    def test_descriptors_wrong_shape(self):
        with pytest.raises(ValueError, match="descriptors must be"):
            FeatureSet(
                keypoints=np.zeros((5, 6), dtype=np.float32),
                descriptors=np.zeros((5, 64), dtype=np.uint8),
            )

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="must agree"):
            FeatureSet(
                keypoints=np.zeros((5, 6), dtype=np.float32),
                descriptors=np.zeros((3, 128), dtype=np.uint8),
            )


class TestAnnotatedImage:
    def test_create(self):
        uid = uuid.uuid4()
        ann = AnnotatedImage(
            annot_uuid=uid,
            name_uuid=None,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            features=_make_features(10),
            bbox=(0, 0, 100, 100),
        )
        assert ann.annot_uuid == uid
        assert ann.name_uuid is None


class TestMatch:
    def test_create(self):
        m = Match(qfx=0, daid=1, dfx=2, dist=0.5, name_uuid=None)
        assert m.qfx == 0
        assert m.daid == 1
        assert m.dfx == 2
        assert m.dist == 0.5


class TestScoredMatch:
    def test_create(self):
        uid = uuid.uuid4()
        sm = ScoredMatch(annot_uuid=uid, name_uuid=None, score=0.95, num_matches=5)
        assert sm.score == 0.95
        assert sm.num_matches == 5
        assert sm.correspondences == []
