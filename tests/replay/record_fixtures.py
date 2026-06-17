#!/usr/bin/env python3
"""Record WBIA identification fixtures for replay tests.

Usage:
    # Start WBIA first (or use existing):
    cd tests/replay && docker compose up -d

    # Record fixtures (waits for WBIA to be healthy):
    python tests/replay/record_fixtures.py

    # Or with explicit URL:
    WBIA_URL=http://wbia:5000 python tests/replay/record_fixtures.py

What it does:
1. Generates synthetic spot-pattern images (via OpenCV)
2. Starts a temporary HTTP server to serve those images
3. Adds images + annotations to WBIA
4. Runs identification via the HotSpotter pipeline (async)
5. Polls for completion and saves the result as an NPZ fixture

Output: tests/replay/testdata/fixtures/<species>_<seed>.npz

Each NPZ contains:
    - annot_uuids: list[str]  — annotation UUIDs (query first)
    - name_uuids: list[str | None]
    - scores: dict[str, float] — per-annotation scores from WBIA
    - num_matches: dict[str, int]
    - config: dict — pipeline config used
    - raw_result: dict — full WBIA job result JSON
    - image_bytes: list[bytes] — PNG-encoded images
    - bboxes: list[tuple[int,int,int,int]]
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import socketserver
import subprocess
import time
from http.server import SimpleHTTPRequestHandler
from threading import Thread
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPLAY_DIR = pathlib.Path(__file__).resolve().parent
TESTDATA = REPLAY_DIR / "testdata"
IMAGES_DIR = TESTDATA / "images"
FIXTURES_DIR = TESTDATA / "fixtures"

WBIA_URL = os.environ.get("WBIA_URL", "http://localhost:5000")
IMAGE_SERVER_PORT = int(os.environ.get("IMAGE_SERVER_PORT", "8899"))
# How the *WBIA container* reaches the host.  With extra_hosts in
# docker-compose.yml this resolves to the Docker host gateway.
HOST_ALIAS = os.environ.get("HOST_ALIAS", "host.docker.internal")

FIXTURES = [
    {
        "name": "zebra_grevys",
        "seed": 42,
        "n_annots": 5,
        "spots_per_annot": [25, 30, 20, 35, 28],
    },
    {
        "name": "giraffe_reticulated",
        "seed": 99,
        "n_annots": 4,
        "spots_per_annot": [40, 35, 45, 38],
    },
    {"name": "whale_shark", "seed": 17, "n_annots": 3, "spots_per_annot": [15, 12, 18]},
]

# Flatten into per-query test cases
TEST_CASES: list[dict] = []
for cfg in FIXTURES:
    for qidx in range(cfg["n_annots"]):
        TEST_CASES.append(
            {
                "species": cfg["name"],
                "seed": cfg["seed"],
                "n_annots": cfg["n_annots"],
                "spots_per_annot": cfg["spots_per_annot"],
                "query_idx": qidx,
            }
        )

_POLL_INTERVAL = 2.0
_POLL_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def _generate_spot_image(
    width: int,
    height: int,
    num_spots: int,
    seed: int,
    annot_index: int,
) -> np.ndarray:
    """Generate a synthetic spot-pattern image.

    Each annotation gets a distinct pattern by combining the base seed
    with the annotation index.  Spots are drawn as filled dark circles
    on a light-gray background.
    """
    rng = np.random.RandomState(seed * 1000 + annot_index)
    img = np.full((height, width, 3), 210, dtype=np.uint8)

    existing: list[tuple[int, int, int]] = []
    attempts = 0
    while len(existing) < num_spots and attempts < num_spots * 5:
        x = int(rng.randint(15, width - 15))
        y = int(rng.randint(15, height - 15))
        r = int(rng.randint(4, 12))
        # Simple non-overlap check
        ok = True
        for ex, ey, er in existing:
            if (x - ex) ** 2 + (y - ey) ** 2 < (r + er + 5) ** 2:
                ok = False
                break
        if ok:
            existing.append((x, y, r))
        attempts += 1

    for x, y, r in existing:
        cv2.circle(img, (x, y), r, (40, 40, 40), -1)
        cv2.circle(img, (x, y), r + 1, (160, 160, 160), 1)

    return img


def _generate_images(cfg: dict) -> list[bytes]:
    """Generate PNG-encoded spot images for one fixture."""
    images: list[bytes] = []
    for i in range(cfg["n_annots"]):
        img = _generate_spot_image(300, 200, cfg["spots_per_annot"][i], cfg["seed"], i)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for annot {i}")
        images.append(buf.tobytes())
    return images


# ---------------------------------------------------------------------------
# HTTP server (serve images to WBIA container)
# ---------------------------------------------------------------------------


class _SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass


def _start_image_server() -> tuple[socketserver.TCPServer, int]:
    os.makedirs(IMAGES_DIR, exist_ok=True)

    class _ImageHandler(_SilentHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(IMAGES_DIR), **kwargs)

    server = socketserver.TCPServer(("0.0.0.0", 0), _ImageHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


# ---------------------------------------------------------------------------
# WBIA API client (stdlib urllib only)
# ---------------------------------------------------------------------------


def _wbia_healthy(url: str, timeout: int = 120) -> bool:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{url}/api/test/heartbeat/")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("response") is True:
                    return True
        except Exception:
            pass
        print("  waiting for WBIA heartbeat...")
        time.sleep(3)
    return False


UUID_KEY = "__UUID__"


def _wrap_uuid(val: str) -> dict:
    """Wrap a UUID string in WBIA's ``__UUID__`` format."""
    return {UUID_KEY: val}


def _wrap_uuid_list(vals: list[str]) -> list[dict]:
    return [_wrap_uuid(v) for v in vals]


def _unwrap_uuid(val: Any) -> str:
    if isinstance(val, dict):
        return str(val.get(UUID_KEY, val))
    return str(val)


# ---------------------------------------------------------------------------
# WBIA API client (stdlib urllib only)
# ---------------------------------------------------------------------------
# Fixture recording
# ---------------------------------------------------------------------------


POST_HEADERS = {"Content-Type": "application/json"}


POST_TIMEOUT = 120


def _post(url: str, data: dict) -> Any:
    import urllib.request

    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=POST_HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=POST_TIMEOUT) as resp:
        return json.loads(resp.read())


def _post_and_unwrap(url: str, data: dict) -> Any:
    result = _post(url, data)
    # WBIA wraps responses in {"status": {...}, "response": ...}
    return result["response"]


def _add_images(wbia_url: str, image_uri_list: list[str]) -> Any:
    return _post_and_unwrap(
        f"{wbia_url}/api/image/json/",
        {"image_uri_list": image_uri_list},
    )


def _add_annots(
    wbia_url: str,
    image_uuid_strs: list[str],
    bboxes: list[list[int]],
    species: str,
) -> Any:
    return _post_and_unwrap(
        f"{wbia_url}/api/annot/json/",
        {
            "image_uuid_list": _wrap_uuid_list(image_uuid_strs),
            "annot_bbox_list": bboxes,
            "annot_theta_list": [0.0] * len(image_uuid_strs),
            "annot_species_list": [species] * len(image_uuid_strs),
        },
    )


def _start_identify(
    wbia_url: str,
    query_uuid_strs: list[str],
    db_uuid_strs: list[str],
) -> Any:
    jobid = _post_and_unwrap(
        f"{wbia_url}/api/engine/query/graph/",
        {
            "query_annot_uuid_list": _wrap_uuid_list(query_uuid_strs),
            "database_annot_uuid_list": _wrap_uuid_list(db_uuid_strs),
            "query_config_dict": {
                "pipeline": "vsmany",
                "pipeline_root": "vsmany",
            },
        },
    )
    return jobid


def _poll_job(wbia_url: str, jobid: str | int) -> Any:
    deadline = time.monotonic() + _POLL_TIMEOUT
    while time.monotonic() < deadline:
        result = _post_and_unwrap(
            f"{wbia_url}/api/engine/job/result/",
            {"jobid": jobid},
        )
        if result.get("status") == "completed":
            return result
        if result.get("status") == "error":
            raise RuntimeError(f"Job {jobid} failed: {result.get('message', '')}")
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(f"Job {jobid} not complete after {_POLL_TIMEOUT}s")


def _image_uri(port: int, filename: str) -> str:
    """URI for the WBIA container to fetch an image."""
    return f"http://{HOST_ALIAS}:{port}/{filename}"


def _record_one(
    wbia_url: str,
    image_server_port: int,
    tc: dict,
) -> pathlib.Path:
    """Record a single fixture and return its path."""
    species = tc["species"]
    query_idx = tc["query_idx"]
    print(f"\n=== {species} query_idx={query_idx} ===")

    # 1. Generate images
    images_bytes = _generate_images(tc)
    filenames: list[str] = []
    for i, blob in enumerate(images_bytes):
        fname = f"{species}_q{query_idx}_a{i}.png"
        (IMAGES_DIR / fname).write_bytes(blob)
        filenames.append(fname)

    # 2. Add images to WBIA
    uris = [_image_uri(image_server_port, fn) for fn in filenames]
    image_uuids = _add_images(wbia_url, uris)
    assert len(image_uuids) == len(
        images_bytes
    ), f"Expected {len(images_bytes)} image UUIDs, got {len(image_uuids)}"
    print(f"  added {len(image_uuids)} images")
    # WBIA returns UUID dicts like {"__UUID__": "..."} — flatten
    image_uuid_strs = [_unwrap_uuid(u) for u in image_uuids]

    # 3. Add annotations
    bboxes = [[20, 10, 260, 180]] * len(image_uuid_strs)
    annot_uuids = _add_annots(wbia_url, image_uuid_strs, bboxes, species)
    annot_uuid_strs = [_unwrap_uuid(u) for u in annot_uuids]
    print(f"  added {len(annot_uuid_strs)} annotations")

    # 4. Run identification
    jobid = _start_identify(wbia_url, [annot_uuid_strs[query_idx]], annot_uuid_strs)
    print(f"  jobid={jobid}")

    # 5. Poll and get result
    result = _poll_job(wbia_url, jobid)
    print(f"  completed")

    # 6. Save fixture
    fixture = {
        "species": species,
        "seed": tc["seed"],
        "query_idx": query_idx,
        "annot_uuids": annot_uuid_strs,
        "name_uuids": [None] * len(annot_uuid_strs),
        "bboxes": bboxes,
        "image_bytes": images_bytes,
        "raw_result": result,
        "config": {"pipeline": "vsmany", "pipeline_root": "vsmany"},
    }

    os.makedirs(FIXTURES_DIR, exist_ok=True)
    fixture_path = FIXTURES_DIR / f"{species}_q{query_idx}.npz"
    np.savez_compressed(fixture_path, **fixture)
    print(f"  → {fixture_path}")
    return fixture_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    print(f"WBIA_URL={WBIA_URL}")
    print(f"Waiting for WBIA to become healthy...")
    if not _wbia_healthy(WBIA_URL):
        print("ERROR: WBIA not healthy.  Start it with:")
        print(f"  cd {REPLAY_DIR} && docker compose up -d")
        raise SystemExit(1)

    server, port = _start_image_server()
    print(f"Image server on port {port}")

    try:
        for tc in TEST_CASES:
            fixture_path = _record_one(WBIA_URL, port, tc)
            print(f"  saved {fixture_path}")
    finally:
        server.shutdown()

    print(f"\nDone.  {len(TEST_CASES)} fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
