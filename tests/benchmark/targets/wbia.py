"""WbiaTargetRunner — multi-step WBIA REST dance."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import shutil
import socket
import socketserver
import subprocess
import tempfile
import time
import uuid as uuid_mod
from http.server import SimpleHTTPRequestHandler
from threading import Thread
from typing import Any

from .base import QueryResult, TargetConfig, TargetRunner

UUID_KEY = "__UUID__"
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 600.0
HOST_ALIAS = "host.docker.internal"


# ---------------------------------------------------------------------------
# WBIA helpers (stdlib urllib)
# ---------------------------------------------------------------------------


def _wrap_uuid(val: str) -> dict:
    return {UUID_KEY: val}


def _unwrap_uuid(val: Any) -> str:
    if isinstance(val, dict):
        return str(val.get(UUID_KEY, val))
    return str(val)


def _post(url: str, data: dict, timeout: int = 300) -> Any:
    import urllib.request

    body = json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_and_unwrap(url: str, data: dict) -> Any:
    return _post(url, data)["response"]


def _wbia_healthy(wbia_url: str, timeout: int = 300) -> bool:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{wbia_url}/api/test/heartbeat/")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("response") is True:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _poll_job(wbia_url: str, jobid: str | int) -> Any:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        result = _post_and_unwrap(
            f"{wbia_url}/api/engine/job/result/",
            {"jobid": jobid},
        )
        if result.get("status") == "completed":
            return result
        if result.get("status") == "error":
            raise RuntimeError(f"Job {jobid} failed: {result.get('message', '')}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {jobid} not complete after {POLL_TIMEOUT}s")


# ---------------------------------------------------------------------------
# WBIA normalisation
# ---------------------------------------------------------------------------


def normalise_wbia_result(raw: dict, db_aid_list: list[str]) -> list[dict]:
    """Parse WBIA's ``json_result.cm_dict`` into canonical annot_scores.

    WBIA returns scores nested under each query annotation UUID inside
    ``cm_dict``.  The lists inside match one-to-one with ``db_aid_list``
    (the DB annotation UUIDs in the order they were sent to the query
    endpoint).

    Reads ``score_list`` (post-name-scoring canonical scores) rather than
    ``annot_score_list`` (raw per-annotation csum pre-name-scoring) so
    that name-level methods (nsum / fmech, csum_wbia, sumamech) are
    correctly compared against wbia-core's name-aligned output.

    Non-canonical annotations receive ``-inf`` from WBIA's
    ``align_name_scores_with_annots`` and are filtered out here (only
    the top annotation per unique name carries the name-level score).
    """
    assert (
        raw.get("status") == "completed"
    ), f"Expected completed, got {raw.get('status')}"
    json_result = raw.get("json_result", raw)
    cm_dict = json_result.get("cm_dict", {})
    if not cm_dict:
        return []

    # Take the first (and only) query annotation
    data = next(iter(cm_dict.values()))
    score_list = data.get("score_list", data.get("annot_score_list", []))
    num_match_list = data.get("num_matches_list", [])

    if num_match_list and len(num_match_list) != len(score_list):
        num_match_list = []

    result: list[dict] = []
    for i, score in enumerate(score_list):
        if i >= len(db_aid_list):
            break
        try:
            s = float(score)
        except (ValueError, TypeError):
            continue
        if s == float("-inf"):
            continue
        n = int(num_match_list[i]) if num_match_list else 0
        result.append({"aid": db_aid_list[i], "score": s, "num_matches": n})

    result.sort(key=lambda x: x["score"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Image server
# ---------------------------------------------------------------------------


class _SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def _start_image_server(
    image_dir: str | pathlib.Path,
) -> tuple[socketserver.TCPServer, int]:
    class _ImageHandler(_SilentHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(image_dir), **kwargs)

    server = socketserver.TCPServer(("0.0.0.0", 0), _ImageHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class WbiaTargetRunner(TargetRunner):
    def __init__(self, config: TargetConfig):
        super().__init__(config)
        self._container_name = f"{config.name}-{uuid_mod.uuid4().hex[:8]}"
        self._container_id: str | None = None
        self._image_dir: pathlib.Path | None = None
        self._image_server: socketserver.TCPServer | None = None
        self._image_server_port: int = 0
        self._wbia_url: str = ""

    def start(self) -> dict:
        port = self.config.port
        image = self.config.image
        name = self._container_name
        self._wbia_url = f"http://localhost:{port}"

        # Locate patch script (relative to this file)
        patch_host = (
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "replay"
            / "patch_wbia_schema.py"
        )
        patch_mount = f"{patch_host}:/patch_wbia_schema.py:ro"
        volume_name = f"wbia-data-{name}"
        entrypoint = (
            "/bin/sh -c '/virtualenv/env3/bin/python /patch_wbia_schema.py "
            "&& exec /bin/entrypoint /virtualenv/env3/bin/python -m wbia.dev "
            "--dbdir /data/db --logdir /data/db/logs --web --containerized --production'"
        )

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"{port}:5000",
            "-v",
            volume_name + ":/data/db",
            "-v",
            patch_mount,
        ]

        # Locate WBIA source for debug code mounts
        _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent
        _wbia_src = _repo_root / "wildbook-ia" / "wbia" / "algo" / "hots"

        # Mount debug log file directly (avoids gunicorn import ordering issues)
        if self.config.debug_log_file:
            log_file = self.config.debug_log_file
            cmd.extend(["-v", f"{log_file}:/app/debug.log"])
            cmd.extend(["-e", "WBIA_DEBUG=1"])
            cmd.extend(["-e", "WBIA_DEBUG_FILE=/app/debug.log"])
            cmd.extend(
                [
                    "-v",
                    f"{_wbia_src / 'debug_log.py'}:"
                    f"/virtualenv/env3/lib/python3.10/site-packages/wbia/algo/hots/debug_log.py:ro",
                ]
            )
            cmd.extend(
                [
                    "-v",
                    f"{_wbia_src / 'pipeline.py'}:"
                    f"/virtualenv/env3/lib/python3.10/site-packages/wbia/algo/hots/pipeline.py:ro",
                ]
            )

        cmd.extend(
            [
                "--add-host",
                "host.docker.internal:host-gateway",
                "-e",
                "EXEC_PRIVILEGED=true",
                "--entrypoint",
                "/bin/sh",
                image,
                "-c",
                entrypoint,
            ]
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container {name}: {result.stderr.strip()}"
            )

        self._container_id = result.stdout.strip()

        self._image_dir = pathlib.Path(
            tempfile.mkdtemp(prefix=f"wbia-img-{self.config.name}-")
        )
        self._image_server, self._image_server_port = _start_image_server(
            self._image_dir
        )

        if not _wbia_healthy(self._wbia_url, timeout=600):
            self.stop()
            raise TimeoutError(f"WBIA container {name} not healthy after 600s")

        return {
            "target": self.config.name,
            "image": self.config.image,
            "container_id": self._container_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_server_port": self._image_server_port,
        }

    def run_query(self, query_index: int, request_body: dict) -> QueryResult:
        wbia_url = self._wbia_url
        image_dir = self._image_dir
        image_server_port = self._image_server_port

        # --- 0. Start timer ---
        _start_ts = time.monotonic()

        # --- 1. Extract images from request body ---
        if self._image_dir is None or self._image_server_port == 0:
            return QueryResult(
                query_index=query_index, error="Image server not started"
            )

        image_dir = self._image_dir
        image_server_port = self._image_server_port

        all_entries: list[dict] = []
        query_b64 = request_body.get("query_image_b64", "")
        query_bbox = request_body.get("query_bbox", [0, 0, 0, 0])
        query_species = request_body.get("query_species", "")
        query_aid = f"q_{query_index}"

        db_entries = request_body.get("database", [])

        filenames: list[str] = []
        entry_metas: list[dict] = []
        annot_names: list[str | None] = []

        # Query image first
        q_bytes = base64.b64decode(query_b64)
        q_fname = f"annot_q{query_index}.png"
        (image_dir / q_fname).write_bytes(q_bytes)
        filenames.append(q_fname)
        entry_metas.append(
            {
                "aid": query_aid,
                "bbox": query_bbox,
                "species": query_species,
                "is_query": True,
            }
        )
        annot_names.append(None)
        all_entries.append(
            {
                "aid": query_aid,
                "bbox": query_bbox,
                "species": query_species,
                "is_query": True,
            }
        )

        # Database images
        for i, db_entry in enumerate(db_entries):
            img_b64 = db_entry.get("image_b64", "")
            bbox = db_entry.get("bbox", [0, 0, 0, 0])
            species = db_entry.get("species", "")
            aid = db_entry.get("aid", f"db_{i}")
            name_uuid = db_entry.get("name_uuid")
            fname = f"annot_q{query_index}_db{i}.png"
            img_bytes = base64.b64decode(img_b64)
            (image_dir / fname).write_bytes(img_bytes)
            filenames.append(fname)
            entry_metas.append(
                {
                    "aid": aid,
                    "bbox": bbox,
                    "species": species,
                    "is_query": False,
                }
            )
            annot_names.append(name_uuid)
            all_entries.append(
                {
                    "aid": aid,
                    "bbox": bbox,
                    "species": species,
                    "is_query": False,
                }
            )

        # Make a list of annot UUIDs in the same order as scores
        uid_map: dict[str, str] = {}
        for e in all_entries:
            uid_map[e["aid"]] = e["aid"]

        try:
            # --- 2. Upload images to WBIA ---
            uris = [f"http://{HOST_ALIAS}:{image_server_port}/{fn}" for fn in filenames]
            image_uuids = _post_and_unwrap(
                f"{wbia_url}/api/image/json/",
                {"image_uri_list": uris},
            )
            image_uuid_strs = [_unwrap_uuid(u) for u in image_uuids]

            # --- 3. Create annotations ---
            bboxes = [e["bbox"] for e in entry_metas]
            species_list = [e["species"] for e in entry_metas]

            annot_uuids = _post_and_unwrap(
                f"{wbia_url}/api/annot/json/",
                {
                    "image_uuid_list": [_wrap_uuid(u) for u in image_uuid_strs],
                    "annot_bbox_list": bboxes,
                    "annot_theta_list": [0.0] * len(image_uuid_strs),
                    "annot_species_list": species_list,
                    "annot_name_list": annot_names,
                },
            )
            annot_uuid_strs = [_unwrap_uuid(u) for u in annot_uuids]

            # Build map: entry_metas index → actual WBIA annot UUID
            wbia_annot_uuids = [uid_map[e["aid"]] for e in entry_metas]

            # --- 4. Run identification ---
            query_annot_uuid = annot_uuid_strs[0]
            db_annot_uuids = annot_uuid_strs[1:]

            jobid = _post_and_unwrap(
                f"{wbia_url}/api/engine/query/graph/",
                {
                    "query_annot_uuid_list": [_wrap_uuid(query_annot_uuid)],
                    "database_annot_uuid_list": [_wrap_uuid(u) for u in db_annot_uuids],
                    "query_config_dict": {
                        "pipeline": "vsmany",
                        "pipeline_root": "vsmany",
                        **request_body.get("config", {}),
                    },
                },
            )

            # --- 5. Poll ---
            raw_result = _poll_job(wbia_url, jobid)

            # --- 6. Normalise ---
            elapsed_ms = (time.monotonic() - _start_ts) * 1000
            annot_scores = normalise_wbia_result(raw_result, wbia_annot_uuids[1:])

            return QueryResult(
                query_index=query_index,
                annot_scores=annot_scores,
                timing_ms=elapsed_ms,
                raw_response=raw_result,
            )

        except Exception as exc:
            return QueryResult(
                query_index=query_index,
                error=str(exc),
            )

    def stop(self) -> None:
        if self._image_server is not None:
            self._image_server.shutdown()
            self._image_server = None

        name = self._container_name
        subprocess.run(
            ["docker", "stop", name],
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["docker", "rm", name],
            capture_output=True,
            timeout=60,
        )

        if self._image_dir is not None and self._image_dir.exists():
            shutil.rmtree(self._image_dir, ignore_errors=True)
            self._image_dir = None
