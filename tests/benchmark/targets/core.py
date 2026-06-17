"""CoreTargetRunner — sends requests to a wbia-core sidecar via single-shot POST."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid as uuid_mod
from pathlib import Path

from .base import QueryResult, TargetConfig, TargetRunner


class CoreTargetRunner(TargetRunner):
    def __init__(self, config: TargetConfig):
        super().__init__(config)
        self._container_name = f"{config.name}-{uuid_mod.uuid4().hex[:8]}"
        self._container_id: str | None = None

    def start(self) -> dict:
        port = self.config.port
        image = self.config.image
        name = self._container_name

        cmd = ["docker", "run", "-d", "--name", name, "-p", f"{port}:5000"]
        cmd.extend(["-e", "WBIA_CORE_DEBUG=1"])
        cmd.append(image)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=500,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container {name}: {result.stderr.strip()}"
            )

        self._container_id = result.stdout.strip()

        deadline = time.monotonic() + 500
        while time.monotonic() < deadline:
            try:
                req = urllib.request.Request(f"http://localhost:{port}/api/health/")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("status") == "ok":
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            self.stop()
            raise TimeoutError(f"Container {name} not healthy after 500s")

        return {
            "target": self.config.name,
            "image": self.config.image,
            "container_id": self._container_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run_query(self, query_index: int, request_body: dict) -> QueryResult:
        port = self.config.port
        url = f"http://localhost:{port}/api/v1/identify/"
        body = json.dumps(request_body).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=1200) as resp:
                raw = json.loads(resp.read())
            elapsed = (time.monotonic() - t0) * 1000

            if raw.get("status") == "completed":
                resp_data = raw["response"]
                return QueryResult(
                    query_index=query_index,
                    annot_scores=resp_data.get("annot_scores", []),
                    timing_ms=resp_data.get("timing_ms", elapsed),
                    raw_response=raw,
                )
            else:
                return QueryResult(
                    query_index=query_index,
                    raw_response=raw,
                    error=raw.get("message", "Unknown error"),
                )
        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                err = json.loads(body)
                msg = err.get("message", str(exc))
            except Exception:
                msg = str(exc)
            return QueryResult(query_index=query_index, error=msg)
        except Exception as exc:
            return QueryResult(query_index=query_index, error=str(exc))

    def stop(self) -> None:
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

    def capture_logs(self, log_file: str) -> None:
        """Save container logs (combined stdout+stderr) to *log_file* before stopping."""
        name = self._container_name
        result = subprocess.run(
            ["docker", "logs", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            combined = (result.stdout or "") + (result.stderr or "")
            Path(log_file).write_text(combined)
