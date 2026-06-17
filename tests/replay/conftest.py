"""pytest fixtures for replay tests — manages Docker compose lifecycle."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

import pytest

REPLAY_DIR = pathlib.Path(__file__).parent
TESTDATA = REPLAY_DIR / "testdata"
FIXTURES_DIR = TESTDATA / "fixtures"
IMAGES_DIR = TESTDATA / "images"

WBIA_URL = os.environ.get("WBIA_URL", "http://localhost:5000")
DOCKER_COMPOSE = os.environ.get("DOCKER_COMPOSE", "docker compose")
RECORD = os.environ.get("WBIA_RECORD_FIXTURES", "0") == "1"


def _wbia_healthy(url: str, timeout: int = 300) -> bool:
    """Poll the WBIA heartbeat endpoint until healthy or timeout."""
    heartbeat = f"{url}/api/test/heartbeat/"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(heartbeat)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("response") is True:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Session-scoped: spin up Docker compose, tear down on exit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def wbia_url() -> str:
    return WBIA_URL


@pytest.fixture(scope="session")
def docker_services(wbia_url: str):
    """Start WBIA + PostgreSQL via docker compose, yield, then tear down."""
    compose_file = REPLAY_DIR / "docker-compose.yml"
    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found — replay tests disabled")

    up = subprocess.run(
        [*DOCKER_COMPOSE.split(), "-f", str(compose_file), "up", "-d"],
        capture_output=True,
        text=True,
        cwd=REPLAY_DIR,
    )
    if up.returncode != 0:
        pytest.fail(f"docker compose up failed:\n{up.stderr}")

    if not _wbia_healthy(wbia_url):
        subprocess.run(
            [*DOCKER_COMPOSE.split(), "-f", str(compose_file), "down", "-v"],
            cwd=REPLAY_DIR,
        )
        pytest.fail("WBIA did not become healthy within timeout")

    yield

    down = subprocess.run(
        [*DOCKER_COMPOSE.split(), "-f", str(compose_file), "down", "-v"],
        capture_output=True,
        text=True,
        cwd=REPLAY_DIR,
    )
    if down.returncode != 0:
        print(f"WARN: docker compose down failed:\n{down.stderr}")


# ---------------------------------------------------------------------------
# Module-scoped: fixture listing
# ---------------------------------------------------------------------------


def _discover_fixtures() -> list[pathlib.Path]:
    if not FIXTURES_DIR.exists():
        return []
    return sorted(FIXTURES_DIR.glob("*.npz"))


@pytest.fixture(scope="session")
def fixture_paths() -> list[pathlib.Path]:
    return _discover_fixtures()


@pytest.fixture
def fixture_path(request) -> pathlib.Path:
    """Parametrized fixture path — see test_replay.py for parametrization."""
    return request.param
