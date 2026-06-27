"""Tests for hotspotter.spatial — uses exact correspondences."""

import uuid

import numpy as np

from hotspotter.data import AnnotatedImage, FeatureSet, ScoredMatch
from hotspotter.spatial import make_sver_shortlist, spatial_verify


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


class TestMakeSverShortlist:
    """Pin WBIA-faithful shortlist selection (scoring.py:96-117)."""

    @staticmethod
    def _sm(aid, nid, canonical, csum):
        return ScoredMatch(
            annot_uuid=aid, name_uuid=nid, score=canonical, annot_csum=csum
        )

    def test_within_name_ranked_by_csum_not_canonical(self):
        """Within a name, annots are ranked by annot score (csum), not canonical.

        Mirrors WBIA ``annot_score_group.argsort()[::-1]`` (scoring.py:108-109):
        even a non-canonical (-inf) annotation with high csum must be selected
        over a canonical one with low csum.
        """
        na = uuid.uuid4()
        # name na: a_lowcsum is canonical (score=10) but has csum 1;
        #          a_highcsum is non-canonical (-inf) but has csum 9.
        a_lowcsum = uuid.uuid4()
        a_highcsum = uuid.uuid4()
        scored = [
            self._sm(a_lowcsum, na, canonical=10.0, csum=1.0),
            self._sm(a_highcsum, na, canonical=float("-inf"), csum=9.0),
        ]
        shortlist = make_sver_shortlist(scored, n_names=1, n_annots_per_name=1)
        # The high-csum annot must be the one kept for SV.
        assert shortlist[0].annot_uuid == a_highcsum

    def test_names_ranked_by_name_score(self):
        """Names are ranked by their name (canonical) score, descending."""
        na, nb = uuid.uuid4(), uuid.uuid4()
        a = uuid.uuid4()
        b = uuid.uuid4()
        scored = [
            self._sm(a, na, canonical=3.0, csum=3.0),
            self._sm(b, nb, canonical=8.0, csum=8.0),
        ]
        shortlist = make_sver_shortlist(
            scored, n_names=1, n_annots_per_name=1, score_method="nsum"
        )
        assert [sm.annot_uuid for sm in shortlist] == [b]

    def test_csum_score_method_uses_flat_shortlist(self):
        """csum score_method takes a flat top-(n_names*n_annots) by csum."""
        na = uuid.uuid4()
        a1, a2, a3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        scored = [
            self._sm(a1, na, canonical=1.0, csum=1.0),
            self._sm(a2, na, canonical=2.0, csum=5.0),
            self._sm(a3, na, canonical=3.0, csum=9.0),
        ]
        shortlist = make_sver_shortlist(
            scored, n_names=1, n_annots_per_name=2, score_method="csum"
        )
        # Flat top-2 by csum: a3 (9), a2 (5)
        assert [sm.annot_uuid for sm in shortlist] == [a3, a2]
