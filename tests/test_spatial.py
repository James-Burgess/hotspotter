"""Tests for hotspotter.spatial — uses exact correspondences."""

import uuid

import numpy as np

from hotspotter.data import AnnotatedImage, FeatureSet, ScoredMatch
from hotspotter.spatial import spatial_verify


def _make_features(n: int) -> FeatureSet:
    return FeatureSet(
        keypoints=np.column_stack(
            [
                np.random.randn(n).astype(np.float32),
                np.random.randn(n).astype(np.float32),
                np.ones(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            ]
        ),
        descriptors=np.random.randint(0, 256, (n, 128), dtype=np.uint8),
    )


class TestSpatialVerify:
    def test_passthrough_below_min(self):
        """No-op when there are fewer correspondences than min_inliers."""
        db = [
            AnnotatedImage(
                annot_uuid=uuid.uuid4(),
                name_uuid=None,
                image=np.zeros((50, 50), dtype=np.uint8),
                features=_make_features(10),
                bbox=(0, 0, 50, 50),
            )
        ]
        query_features = _make_features(10)

        scored = [
            ScoredMatch(
                annot_uuid=db[0].annot_uuid,
                name_uuid=None,
                score=0.5,
                num_matches=2,
                correspondences=[(0, 0)],
            ),
        ]
        result, sv_results = spatial_verify(scored, query_features, db, min_inliers=3)
        assert result[0].sv_inliers == 0
        assert result[0].sv_homography is None
        assert len(sv_results) == 0

    def test_exact_correspondences_used(self):
        """Build known keypoint pairs and verify they pass through."""
        db = [
            AnnotatedImage(
                annot_uuid=uuid.uuid4(),
                name_uuid=None,
                image=np.zeros((50, 50), dtype=np.uint8),
                features=_make_features(10),
                bbox=(0, 0, 50, 50),
            )
        ]
        xs = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.5, 1.5], dtype=np.float32)
        ys = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        q_kp = np.column_stack(
            [
                xs,
                ys,
                np.ones(8, dtype=np.float32),
                np.zeros(8, dtype=np.float32),
                np.ones(8, dtype=np.float32),
                np.zeros(8, dtype=np.float32),
            ]
        )
        query_features = FeatureSet(
            keypoints=q_kp,
            descriptors=np.random.randint(0, 256, (8, 128), dtype=np.uint8),
        )

        # db_kp at indices 2..9 — same keypoints as query 0..7
        db_kp = db[0].features.keypoints
        db_kp[2:10] = q_kp

        scored = [
            ScoredMatch(
                annot_uuid=db[0].annot_uuid,
                name_uuid=None,
                score=0.5,
                num_matches=8,
                correspondences=[
                    (0, 2),
                    (1, 3),
                    (2, 4),
                    (3, 5),
                    (4, 6),
                    (5, 7),
                    (6, 8),
                    (7, 9),
                ],
            ),
        ]
        result, sv_results = spatial_verify(scored, query_features, db, min_inliers=4)
        assert result[0].sv_inliers > 0
        assert result[0].sv_homography is not None
        assert len(sv_results) == 1
        _, _, weights = next(iter(sv_results.values()))
        assert len(weights) == result[0].sv_inliers
        assert np.all(weights >= 0.0) and np.all(weights <= 1.0)

    def test_out_of_range_correspondences_skipped(self):
        """Correspondences with out-of-range indices are silently skipped."""
        db = [
            AnnotatedImage(
                annot_uuid=uuid.uuid4(),
                name_uuid=None,
                image=np.zeros((50, 50), dtype=np.uint8),
                features=_make_features(10),
                bbox=(0, 0, 50, 50),
            )
        ]
        query_features = _make_features(10)

        scored = [
            ScoredMatch(
                annot_uuid=db[0].annot_uuid,
                name_uuid=None,
                score=0.5,
                num_matches=20,
                correspondences=[
                    (999, 0),
                    (0, 999),
                ],
            ),
        ]
        result, sv_results = spatial_verify(scored, query_features, db, min_inliers=3)
        assert result[0].sv_inliers == 0
        assert result[0].sv_homography is None
        assert len(sv_results) == 0
