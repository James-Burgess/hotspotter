"""Basic tests for wbia_core.knn (requires pyflann or faiss)."""

import numpy as np

from wbia_core.data import FeatureSet
from wbia_core.knn import build_index, query_index


def _make_features(n: int = 50) -> FeatureSet:
    return FeatureSet(
        keypoints=np.random.randn(n, 6).astype(np.float32),
        descriptors=np.random.randint(0, 256, (n, 128), dtype=np.uint8),
    )


class TestBuildIndex:
    def test_build_and_query(self):
        db_feats = _make_features(50)
        q_feats = _make_features(10)

        index = build_index(db_feats)

        distances, labels = query_index(index, q_feats, k=3)
        assert distances.shape == (10, 3)
        assert labels.shape == (10, 3)
        assert distances.dtype == np.float32

    def test_nearest_is_self(self):
        """When the query features are in the index, the nearest neighbour
        should be itself (distance == 0)."""
        feats = _make_features(30)
        index = build_index(feats)
        distances, labels = query_index(index, feats, k=1)
        # since the query is in the index the top result should be
        # itself with distance ≈ 0.
        assert np.allclose(distances.min(), 0.0, atol=1e-3)
