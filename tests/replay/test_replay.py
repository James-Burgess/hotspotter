"""Replay tests — compare wbia-core against recorded WBIA fixtures.

These tests are parametrized over NPZ fixtures in testdata/fixtures/.
Each fixture contains a WBIA identification result for a synthetic
image set.  The test replays the same identification through
wbia-core and compares the rankings.

Skipped by default (marker ``replay``).  Run with::

    pytest tests/replay/ -m replay

Or to record new fixtures first::

    cd tests/replay && docker compose up -d
    python record_fixtures.py
    cd ../.. && pytest tests/replay/ -m replay

Environment variables:
    WBIA_URL                — WBIA endpoint (default http://localhost:5000)
    WBIA_RECORD_FIXTURES=1  — record fixtures instead of using cached
"""

from __future__ import annotations

import json
import pathlib
import uuid as uuid_mod

import numpy as np
import pytest

from wbia_core.config import HotSpotterConfig, IdentificationConfig
from wbia_core.data import AnnotatedImage, FeatureSet
from wbia_core.pipeline import identify

from .conftest import FIXTURES_DIR

UUID_KEY = "__UUID__"


def _unwrap_uuid(val) -> str:
    if isinstance(val, dict):
        return str(val.get(UUID_KEY, val))
    return str(val)


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def _load_fixture(path: pathlib.Path) -> dict:
    data = np.load(path, allow_pickle=True)
    raw = dict(data["raw_result"].item()) if data.get("raw_result") else {}
    return {
        "species": str(data.get("species", "")),
        "seed": int(data.get("seed", 0)),
        "query_idx": int(data.get("query_idx", 0)),
        "annot_uuids": [str(u) for u in data["annot_uuids"]],
        "name_uuids": [
            str(v) if v is not None else None for v in data.get("name_uuids", [])
        ],
        "bboxes": [tuple(b) for b in data.get("bboxes", [])],
        "image_bytes": list(data["image_bytes"]),
        "raw_result": raw,
        "config": dict(data.get("config", {}).item() if data.get("config") else {}),
    }


def _parse_wbia_scores(raw_result: dict) -> dict[str, float]:
    json_result = raw_result.get("json_result", raw_result)
    scores: dict[str, float] = {}

    cm_dict = json_result.get("cm_dict")
    if cm_dict:
        for _qauuid, data in cm_dict.items():
            dauuids = [_unwrap_uuid(u) for u in data.get("dannot_uuid_list", [])]
            score_list = data.get("annot_score_list", [])
            for duuid, score in zip(dauuids, score_list):
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        continue
                if np.isfinite(score):
                    scores[duuid] = float(score)
        return scores

    if isinstance(json_result, list):
        for entry in json_result:
            dauuids = [_unwrap_uuid(u) for u in entry.get("dauuid_list", [])]
            score_list = entry.get("score_list", [])
            for duuid, score in zip(dauuids, score_list):
                if np.isfinite(score):
                    scores[duuid] = float(score)
        return scores

    return scores


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------


def _discover_fixtures() -> list[pathlib.Path]:
    if not FIXTURES_DIR.exists():
        return []
    return sorted(FIXTURES_DIR.glob("*.npz"))


@pytest.fixture
def fixture_path(request) -> pathlib.Path:
    return request.param


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

_all_fixtures = _discover_fixtures()


# ---------------------------------------------------------------------------
# Tests that validate fixture structure (no pyhesaff needed)
# ---------------------------------------------------------------------------


class TestReplayFixtureLoading:

    @pytest.mark.skipif(not _all_fixtures, reason="No replay fixtures found")
    @pytest.mark.parametrize("fixture_path", _all_fixtures, indirect=True)
    def test_fixture_loads(self, fixture_path: pathlib.Path):
        fx = _load_fixture(fixture_path)
        assert len(fx["annot_uuids"]) >= 2
        assert len(fx["image_bytes"]) >= 2
        assert fx["species"]
        assert fx["raw_result"]

    @pytest.mark.skipif(not _all_fixtures, reason="No replay fixtures found")
    @pytest.mark.parametrize("fixture_path", _all_fixtures, indirect=True)
    def test_wbia_scores_parsable(self, fixture_path: pathlib.Path):
        fx = _load_fixture(fixture_path)
        scores = _parse_wbia_scores(fx["raw_result"])
        assert len(scores) > 0
        for uid, score in scores.items():
            assert isinstance(uid, str)
            assert isinstance(score, float)

    @pytest.mark.skipif(not _all_fixtures, reason="No replay fixtures found")
    @pytest.mark.parametrize("fixture_path", _all_fixtures, indirect=True)
    def test_image_decodable(self, fixture_path: pathlib.Path):
        import cv2 as _cv2

        fx = _load_fixture(fixture_path)
        for i, blob in enumerate(fx["image_bytes"]):
            buf = np.frombuffer(blob, dtype=np.uint8)
            img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
            assert img is not None, f"Image {i} failed to decode"
            assert img.ndim == 3


# ---------------------------------------------------------------------------
# Tests that require pyhesaff + recorded fixtures
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _all_fixtures, reason="No replay fixtures found")
class TestReplayAgainstWbiaCore:

    @pytest.mark.parametrize("fixture_path", _discover_fixtures(), indirect=True)
    def test_replay_rankings(self, fixture_path: pathlib.Path):
        """Top-N annotations from wbia-core should overlap with WBIA's."""
        fx = _load_fixture(fixture_path)
        wbia_scores = _parse_wbia_scores(fx["raw_result"])
        if len(wbia_scores) == 0:
            pytest.skip("No WBIA scores to compare")

        wbia_top = sorted(wbia_scores, key=lambda k: wbia_scores[k], reverse=True)[:5]

        database = _build_database_from_fixture(fx)
        query_idx = fx["query_idx"]

        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=10)
        )
        wbia_core_results = identify(query_idx, database, config)
        core_top = [str(r.annot_uuid) for r in wbia_core_results]

        overlap = set(wbia_top) & set(core_top)
        assert (
            len(overlap) > 0
        ), f"No overlap between WBIA top-5 and wbia-core top-10 for {fixture_path.name}\n  WBIA: {wbia_top}\n  Core: {core_top}"

    @pytest.mark.replay
    @pytest.mark.parametrize("fixture_path", _discover_fixtures(), indirect=True)
    def test_replay_self_excluded(self, fixture_path: pathlib.Path):
        fx = _load_fixture(fixture_path)
        database = _build_database_from_fixture(fx)
        query_idx = fx["query_idx"]
        query_uuid = fx["annot_uuids"][query_idx]

        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=10)
        )
        results = identify(query_idx, database, config)
        result_uuids = [str(r.annot_uuid) for r in results]
        assert query_uuid not in result_uuids

    @pytest.mark.replay
    @pytest.mark.parametrize("fixture_path", _discover_fixtures(), indirect=True)
    def test_replay_correspondences(self, fixture_path: pathlib.Path):
        fx = _load_fixture(fixture_path)
        database = _build_database_from_fixture(fx)
        query_idx = fx["query_idx"]

        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=10)
        )
        results = identify(query_idx, database, config)
        for r in results:
            assert len(r.correspondences) > 0

    @pytest.mark.replay
    @pytest.mark.slow
    @pytest.mark.parametrize("fixture_path", _discover_fixtures(), indirect=True)
    def test_replay_with_spatial_verification(self, fixture_path: pathlib.Path):
        fx = _load_fixture(fixture_path)
        database = _build_database_from_fixture(fx)
        query_idx = fx["query_idx"]

        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=True, num_return=10)
        )
        results = identify(query_idx, database, config)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Live end-to-end comparison: same image sent to WBIA and wbia-core
# ---------------------------------------------------------------------------


@pytest.mark.replay
class TestLiveWbiaComparison:
    """Generate images, send them to both WBIA (via REST) and wbia-core
    (via pipeline), and compare the result rankings."""

    def test_live_rankings_overlap(self, wbia_url: str):
        from .record_fixtures import (
            IMAGES_DIR,
            _add_annots,
            _add_images,
            _generate_images,
            _image_uri,
            _poll_job,
            _start_identify,
            _start_image_server,
            _unwrap_uuid,
            _wbia_healthy,
        )
        import uuid as uuid_mod
        from wbia_core.features import extract_features
        from wbia_core.config import SiftConfig
        import cv2 as _cv2

        # --- 1. Generate a small test set ---
        cfg = {
            "name": "zebra_grevys",
            "seed": 42,
            "n_annots": 4,
            "spots_per_annot": [25, 30, 20, 35],
        }
        query_idx = 0
        images_bytes = _generate_images(cfg)

        # --- 2. Start image server and write images to its directory ---
        import os as _os

        _os.makedirs(str(IMAGES_DIR), exist_ok=True)
        filenames = []
        for i, blob in enumerate(images_bytes):
            fname = f"live_test_a{i}.png"
            (IMAGES_DIR / fname).write_bytes(blob)
            filenames.append(fname)

        server, port = _start_image_server()
        try:

            # --- 3. Ensure WBIA is up ---
            assert _wbia_healthy(wbia_url, timeout=60), "WBIA not healthy"

            # --- 4. Add images to WBIA ---
            uris = [_image_uri(port, fn) for fn in filenames]
            image_uuids = _add_images(wbia_url, uris)
            image_uuid_strs = [_unwrap_uuid(u) for u in image_uuids]
            assert len(image_uuid_strs) == cfg["n_annots"]

            # --- 5. Add annotations ---
            bboxes = [[20, 10, 260, 180]] * cfg["n_annots"]
            annot_uuids = _add_annots(wbia_url, image_uuid_strs, bboxes, cfg["name"])
            annot_uuid_strs = [_unwrap_uuid(u) for u in annot_uuids]

            # --- 6. Run WBIA identification ---
            jobid = _start_identify(
                wbia_url,
                [annot_uuid_strs[query_idx]],
                annot_uuid_strs,
            )
            raw_result = _poll_job(wbia_url, jobid)
            assert raw_result["status"] == "completed"

            wbia_scores = _parse_wbia_scores(raw_result)
            assert len(wbia_scores) > 0, "No scores from WBIA"
            wbia_top = sorted(wbia_scores, key=lambda k: wbia_scores[k], reverse=True)[
                :5
            ]
        finally:
            server.shutdown()

        # --- 7. Run wbia-core on the same images ---
        sift_cfg = SiftConfig()
        database: list[AnnotatedImage] = []
        for i, blob in enumerate(images_bytes):
            buf = np.frombuffer(blob, dtype=np.uint8)
            img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
            features = extract_features(img, sift_cfg)
            database.append(
                AnnotatedImage(
                    annot_uuid=uuid_mod.UUID(annot_uuid_strs[i]),
                    name_uuid=None,
                    image=img,
                    features=features,
                    bbox=(20, 10, 260, 180),
                )
            )

        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=False, num_return=10)
        )
        core_results = identify(query_idx, database, config)
        core_top = [str(r.annot_uuid) for r in core_results]

        # --- 8. Compare ---
        overlap = set(wbia_top) & set(core_top)
        assert len(overlap) > 0, (
            f"No top-N overlap between WBIA and wbia-core.\n"
            f"  WBIA top-5: {wbia_top}\n"
            f"  wbia-core top-10: {core_top}"
        )


def _build_database_from_fixture(fx: dict) -> list[AnnotatedImage]:
    """Build an AnnotatedImage list from a fixture by extracting features.

    Requires ``wbia-pyhesaff`` (hard dependency of wbia-core).
    """
    from wbia_core.features import extract_features
    from wbia_core.config import SiftConfig
    import cv2 as _cv2

    sift_cfg = SiftConfig()
    database: list[AnnotatedImage] = []

    for i, img_bytes in enumerate(fx["image_bytes"]):
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        img = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode image {i}")

        features = extract_features(img, sift_cfg)

        auuid = uuid_mod.UUID(fx["annot_uuids"][i])
        nuuid = uuid_mod.UUID(fx["name_uuids"][i]) if fx["name_uuids"][i] else None
        bbox = fx["bboxes"][i] if fx["bboxes"] else (0, 0, img.shape[1], img.shape[0])
        bbox_int = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        database.append(
            AnnotatedImage(
                annot_uuid=auuid,
                name_uuid=nuuid,
                image=img,
                features=features,
                bbox=bbox_int,
            )
        )

    return database
