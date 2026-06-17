"""Integration test for the full :func:`identify` pipeline."""

import uuid

import numpy as np
import pytest

from wbia_core.config import HotSpotterConfig, IdentificationConfig
from wbia_core.data import AnnotatedImage, FeatureSet
from wbia_core.pipeline import identify

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_annot(name_uuid: uuid.UUID | None, n_feats: int) -> AnnotatedImage:
    return AnnotatedImage(
        annot_uuid=uuid.uuid4(),
        name_uuid=name_uuid,
        image=np.zeros((100, 100), dtype=np.uint8),
        features=_make_features(n_feats),
        bbox=(0, 0, 100, 100),
    )


# ---------------------------------------------------------------------------
# Integration test: full identify() with synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_database(
    n_annotations: int = 5,
    feats_per_annot: int = 20,
    same_name_pairs: list[tuple[int, int]] | None = None,
) -> list[AnnotatedImage]:
    """Build a database with controllable name sharing."""
    names = [uuid.uuid4() for _ in range(n_annotations)]
    if same_name_pairs:
        for i, j in same_name_pairs:
            names[j] = names[i]  # share name

    return [_make_annot(names[i], feats_per_annot) for i in range(n_annotations)]


class TestIdentify:
    """Integration-level tests that exercise the full pipeline end-to-end.

    These tests verify that the pipeline runs without error and returns
    sensible output shapes.  They do *not* verify correctness of the
    ranking — that requires shadow-mode comparison against WBIA.
    """

    def test_returns_correct_shape(self):
        db = _make_synthetic_database(5, 20)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=3)
        )
        results = identify(0, db, config)
        assert len(results) <= 3
        for r in results:
            assert isinstance(r.annot_uuid, uuid.UUID)
            assert isinstance(r.score, float)
            assert r.score >= 0

    def test_query_self_excluded(self):
        """The query annotation should not appear in its own results."""
        db = _make_synthetic_database(3, 20)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=5)
        )
        results = identify(0, db, config)
        uuids = [r.annot_uuid for r in results]
        assert db[0].annot_uuid not in uuids

    def test_same_name_excluded(self):
        """Annotations sharing the query's name should be filtered out."""
        db = _make_synthetic_database(3, 20, same_name_pairs=[(0, 1)])
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(
                sv_on=False, num_return=5, can_match_samename=False
            )
        )
        results = identify(0, db, config)
        uuids = [r.annot_uuid for r in results]
        assert db[0].annot_uuid not in uuids
        assert db[1].annot_uuid not in uuids  # same name
        assert db[2].annot_uuid in uuids

    def test_with_spatial_verification(self):
        """Full pipeline with SV enabled (smoke test)."""
        db = _make_synthetic_database(3, 30)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=True, num_return=3)
        )
        results = identify(0, db, config)
        assert len(results) > 0

    def test_correspondences_present(self):
        """ScoredMatches from the pipeline should have correspondences."""
        db = _make_synthetic_database(3, 20)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=5)
        )
        results = identify(0, db, config)
        for r in results:
            assert len(r.correspondences) > 0
            for qfx, dfx in r.correspondences:
                assert isinstance(qfx, int)
                assert isinstance(dfx, int)

    def test_different_pipeline_name(self):
        """Should raise because only HotSpotter is implemented."""
        config = IdentificationConfig(pipeline="MiewId")
        with pytest.raises(NotImplementedError):
            identify(0, [_make_annot(uuid.uuid4(), 10)], config)

    @pytest.mark.slow
    def test_large_database(self):
        """Stress test with many annotations (marked slow)."""
        db = _make_synthetic_database(50, 15)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=10)
        )
        results = identify(0, db, config)
        assert len(results) <= 10
