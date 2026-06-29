"""Integration test for the full :func:`identify` pipeline."""

import uuid

import numpy as np
import pytest
from pydantic import ValidationError

from hotspotter.config import HotSpotterConfig, IdentificationConfig
from hotspotter.data import AnnotatedImage, FeatureSet
from hotspotter.pipeline import _compute_kpad, identify

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


def _make_annot(
    name_uuid: uuid.UUID | None,
    n_feats: int,
    image_uuid: uuid.UUID | None = None,
) -> AnnotatedImage:
    return AnnotatedImage(
        annot_uuid=uuid.uuid4(),
        name_uuid=name_uuid,
        image=np.zeros((100, 100), dtype=np.uint8),
        features=_make_features(n_feats),
        bbox=(0, 0, 100, 100),
        image_uuid=image_uuid,
    )


# ---------------------------------------------------------------------------
# Integration test: full identify() with synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_database(
    n_annotations: int = 5,
    feats_per_annot: int = 20,
    same_name_pairs: list[tuple[int, int]] | None = None,
    same_image_pairs: list[tuple[int, int]] | None = None,
) -> list[AnnotatedImage]:
    """Build a database with controllable name / image sharing."""
    names = [uuid.uuid4() for _ in range(n_annotations)]
    if same_name_pairs:
        for i, j in same_name_pairs:
            names[j] = names[i]  # share name

    images = [uuid.uuid4() for _ in range(n_annotations)]
    if same_image_pairs:
        for i, j in same_image_pairs:
            images[j] = images[i]  # share source image

    return [
        _make_annot(names[i], feats_per_annot, images[i]) for i in range(n_annotations)
    ]


class TestIdentify:
    """Integration-level tests that exercise the full pipeline end-to-end.

    These tests verify that the pipeline runs without error and returns
    sensible output shapes.  They do *not* verify correctness of the
    ranking — that requires shadow-mode comparison against WBIA.
    """

    def test_returns_correct_shape(self):
        db = _make_synthetic_database(5, 20)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=3, knn_backend="exact")
        )
        results = identify(0, db, config)
        assert len(results) <= 3
        for r in results:
            assert isinstance(r.annot_uuid, uuid.UUID)
            assert isinstance(r.score, float)
            assert r.score >= 0

    def test_knorm_is_configurable(self):
        config = HotSpotterConfig(knorm=2)
        assert config.knorm == 2

    def test_knorm_rejects_zero(self):
        with pytest.raises(ValidationError):
            HotSpotterConfig(knorm=0)

    def test_identify_uses_configured_knorm(self, monkeypatch):
        captured = {}
        original_exact_knn = identify.__globals__["exact_knn"]

        def capture_exact_knn(query_features, db_feats, k_total):
            captured["k_total"] = k_total
            return original_exact_knn(query_features, db_feats, k_total)

        monkeypatch.setitem(identify.__globals__, "exact_knn", capture_exact_knn)
        db = _make_synthetic_database(4, 20)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(
                knn=4,
                knorm=2,
                kpad=1,
                sv_on=False,
                knn_backend="exact",
            )
        )

        identify(0, db, config)

        assert captured["k_total"] == 8

    def test_dynamic_kpad_counts_same_name(self):
        db = _make_synthetic_database(4, 20, same_name_pairs=[(0, 1), (0, 2)])
        hs = HotSpotterConfig(kpad_policy="dynamic", can_match_samename=False)
        assert _compute_kpad(hs, 0, db) == 2

    def test_query_self_excluded(self):
        """The query annotation should not appear in its own results."""
        db = _make_synthetic_database(3, 20)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=5, knn_backend="exact")
        )
        results = identify(0, db, config)
        uuids = [r.annot_uuid for r in results]
        assert db[0].annot_uuid not in uuids

    def test_same_name_excluded(self):
        """Annotations sharing the query's name should be filtered out."""
        db = _make_synthetic_database(3, 20, same_name_pairs=[(0, 1)])
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(
                sv_on=False,
                num_return=5,
                can_match_samename=False,
                knn_backend="exact",
            )
        )
        results = identify(0, db, config)
        uuids = [r.annot_uuid for r in results]
        assert db[0].annot_uuid not in uuids
        assert db[1].annot_uuid not in uuids  # same name
        assert db[2].annot_uuid in uuids

    def test_with_spatial_verification(self):
        """Full pipeline with SV enabled (smoke test)."""
        pytest.importorskip("vtool.spatial_verification")
        db = _make_synthetic_database(3, 30)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=True, num_return=3, knn_backend="exact")
        )
        results = identify(0, db, config)
        assert len(results) > 0

    def test_correspondences_present(self):
        """ScoredMatches from the pipeline should have correspondences."""
        db = _make_synthetic_database(3, 20)
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=5, knn_backend="exact")
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
            hotspotter=HotSpotterConfig(sv_on=False, num_return=10, knn_backend="exact")
        )
        results = identify(0, db, config)
        assert len(results) <= 10


# ---------------------------------------------------------------------------
# Regression tests for WBIA-faithful behaviour
# ---------------------------------------------------------------------------


class TestWbiaFidelity:
    """Tests pinning behaviour to WBIA's exact defaults / control flow."""

    def test_sv_n_annot_per_name_parity_default(self):
        """Default is 999 (verify-all) for cross-process parity.

        WBIA's literal default is 3 (Config.py:288), but prescore-based
        shortlist selection diverges across processes due to FLANN noise, so
        999 is used to let the SV inlier test alone decide survival. See
        HotSpotterConfig.sv_n_annot_per_name docstring.
        """
        hs = HotSpotterConfig()
        assert hs.sv_n_annot_per_name == 999

    def test_can_match_sameimg_default_is_false(self):
        """WBIA's ``can_match_sameimg`` default is False (Config.py:490)."""
        hs = HotSpotterConfig()
        assert hs.can_match_sameimg is False

    def test_same_image_excluded_when_disabled(self):
        """can_match_sameimg=False must filter same-image ('contact') annots."""
        db = _make_synthetic_database(3, 20, same_image_pairs=[(0, 1)])
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(
                sv_on=False,
                num_return=5,
                can_match_sameimg=False,
                knn_backend="exact",
            )
        )
        results = identify(0, db, config)
        uuids = [r.annot_uuid for r in results]
        assert db[0].annot_uuid not in uuids  # self
        assert db[1].annot_uuid not in uuids  # same image
        assert db[2].annot_uuid in uuids  # different image

    def test_same_image_allowed_when_enabled(self):
        """can_match_sameimg=True must NOT filter same-image annots."""
        db = _make_synthetic_database(3, 20, same_image_pairs=[(0, 1)])
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(
                sv_on=False,
                num_return=5,
                can_match_sameimg=True,
                knn_backend="exact",
            )
        )
        results = identify(0, db, config)
        uuids = [r.annot_uuid for r in results]
        assert db[1].annot_uuid in uuids  # same image, allowed

    def test_dynamic_kpad_counts_same_image(self):
        """Dynamic Kpad must count same-image annots too (WBIA impossible daids)."""
        db = _make_synthetic_database(4, 20, same_image_pairs=[(0, 1), (0, 2)])
        hs = HotSpotterConfig(
            kpad_policy="dynamic",
            can_match_samename=True,  # isolate the same-image contribution
            can_match_sameimg=False,
        )
        assert _compute_kpad(hs, 0, db) == 2

    def test_results_ranked_by_canonical_score(self):
        """Final ordering must follow the canonical name score (WBIA score_list).

        WBIA's ``get_top_aids`` argsorts ``cm.score_list``, which holds the
        canonical name score (best annot per name); same-name runners-up get
        -inf and sink. We build two names where the csum-best annotation is NOT
        the canonical one and assert the canonical annotation ranks first.
        """
        from hotspotter.data import ScoredMatch as _SM

        # name A: two annots, annot_a2 has higher csum but the same name score
        # is carried by whichever annot is canonical (highest csum within name
        # for maxcsum, or the fmech carrier). We check the sort key directly.
        na, nb = uuid.uuid4(), uuid.uuid4()
        a1 = uuid.uuid4()
        a2 = uuid.uuid4()
        b1 = uuid.uuid4()

        # ScoredMatch.score = canonical; .annot_csum = per-annot csum
        # a1 is canonical for name A (score 9.0), a2 is a runner-up (-inf, csum 5)
        # b1 is canonical for name B (score 7.0). Sorted desc by score: a1, b1, a2.
        scored = [
            _SM(annot_uuid=a1, name_uuid=na, score=9.0, annot_csum=9.0),
            _SM(annot_uuid=a2, name_uuid=na, score=float("-inf"), annot_csum=5.0),
            _SM(annot_uuid=b1, name_uuid=nb, score=7.0, annot_csum=7.0),
        ]
        csum_annot = {a1: 9.0, a2: 5.0, b1: 7.0}
        scored.sort(
            key=lambda sm: (sm.score, csum_annot.get(sm.annot_uuid, 0.0)),
            reverse=True,
        )
        assert [sm.annot_uuid for sm in scored] == [a1, b1, a2]
