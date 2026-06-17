"""Tests for the wbia-core sidecar Flask app."""

import base64

import cv2
import numpy as np
import pytest

from sidecar.api import app


def _make_jpeg_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/api/health/")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["service"] == "wbia-core"


class TestIdentify:
    def test_identify_single_shot(self, client):
        rng = np.random.RandomState(42)
        query_img = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        db_img = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        payload = {
            "query_image_b64": _make_jpeg_b64(query_img),
            "query_bbox": [10, 20, 80, 60],
            "query_theta": 0.0,
            "database": [
                {
                    "aid": "coco-annot-1",
                    "image_b64": _make_jpeg_b64(db_img),
                    "bbox": [10, 20, 80, 60],
                    "theta": 0.0,
                    "name_uuid": None,
                }
            ],
            "config": {
                "pipeline_root": "vsmany",
                "K": 4,
                "Knorm": 1,
                "Kpad": 0,
                "fg_on": False,
                "sv_on": False,
            },
        }

        resp = client.post("/api/v1/identify/", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "completed"

        response = data["response"]
        assert "annot_scores" in response
        assert "timing_ms" in response
        assert isinstance(response["timing_ms"], (int, float))

        scores = response["annot_scores"]
        assert len(scores) > 0
        assert scores[0]["aid"] == "coco-annot-1"
        assert isinstance(scores[0]["score"], (int, float))
        assert isinstance(scores[0]["num_matches"], int)
