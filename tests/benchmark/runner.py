"""Benchmark orchestrator — start targets, run queries, save results."""

from __future__ import annotations

import base64
import json
import subprocess
import time
import uuid as uuid_mod
from pathlib import Path
from typing import Any

from coco.loader import CocoSubset
from targets.base import QueryResult, TargetRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_images(body: dict) -> dict:
    """Return a copy of *body* without base64 image data for disk logging."""
    result: dict[str, Any] = {}
    for k, v in body.items():
        if k in ("query_image_b64",):
            result[k] = f"<base64 {len(v)} bytes>"
        elif k == "database" and isinstance(v, list):
            cleaned = []
            for entry in v:
                e = dict(entry)
                if "image_b64" in e:
                    e["image_b64"] = f"<base64 {len(e['image_b64'])} bytes>"
                cleaned.append(e)
            result[k] = cleaned
        else:
            result[k] = v
    return result


def _get_git_info(project_root: Path) -> dict[str, Any]:
    """Get git commit, branch, and dirty status from the project root."""
    info: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["commit"] = result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["branch"] = result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = bool(result.stdout.strip())
    except Exception:
        pass
    return info


def _build_annot_meta(subset: CocoSubset) -> list[dict[str, Any]]:
    """Build annotation metadata list from the subset."""
    query_set = set(subset.query_indices)
    meta = []
    for idx, ann in enumerate(subset.annotations):
        meta.append(
            {
                "annot_id": ann.annot_id,
                "image_id": ann.image_id,
                "bbox": [int(v) for v in ann.bbox],
                "species": ann.species,
                "individual_ids": ann.individual_ids,
                "width": ann.width,
                "height": ann.height,
                "is_query": idx in query_set,
                "query_index": (
                    query_set_idx
                    if (query_set_idx := _find_query_index(subset.query_indices, idx))
                    is not None
                    else None
                ),
            }
        )
    return meta


def _find_query_index(query_indices: list[int], idx: int) -> int | None:
    for qi, qi_val in enumerate(query_indices):
        if qi_val == idx:
            return qi
    return None


def _make_name_uuid(individual_ids: list[int]) -> str | None:
    """Create a deterministic name UUID from the first individual_id."""
    if not individual_ids:
        return None
    return str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"ind-{individual_ids[0]}"))


# ---------------------------------------------------------------------------
# Metadata writing
# ---------------------------------------------------------------------------


def write_run_metadata(
    results_dir: Path,
    subset: CocoSubset,
    targets: list[TargetRunner],
    config: dict,
    cli_args: dict[str, Any] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    errors: list[str] | None = None,
) -> None:
    """Write run.json and annotations.json into the results directory."""
    if started_at is None:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if finished_at is None:
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    started_epoch = time.mktime(time.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ"))
    finished_epoch = time.mktime(time.strptime(finished_at, "%Y-%m-%dT%H:%M:%SZ"))

    git_info = _get_git_info(results_dir.resolve())
    target_names = [t.config.name for t in targets]

    run_id = results_dir.name.replace("test-run-results-", "")

    run_meta = {
        "run_id": run_id,
        "created_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(max(0, finished_epoch - started_epoch), 1),
        "targets": target_names,
        "n_annotations": len(subset.annotations),
        "n_queries": len(subset.query_indices),
        "seed": subset.config.get("seed"),
        "species": subset.config.get("species"),
        "config": config,
        "cli_args": cli_args or {},
        "git": git_info,
        "status": "completed" if not errors else "partial" if errors else "completed",
        "errors": errors or [],
    }

    (results_dir / "run.json").write_text(json.dumps(run_meta, indent=2))

    # annotations.json
    annot_meta = _build_annot_meta(subset)
    (results_dir / "annotations.json").write_text(json.dumps(annot_meta, indent=2))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_benchmark(
    subset: CocoSubset,
    targets: list[TargetRunner],
    results_dir: str | Path,
    config: dict,
    cli_args: dict[str, Any] | None = None,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write config
    (results_dir / "config.json").write_text(
        json.dumps({**subset.config, **config}, indent=2)
    )

    aggregate: dict[str, Any] = {
        "targets": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    debug_logs_dir = results_dir / "debug-logs"
    debug_logs_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        name = target.config.name
        target.config.debug_logs_dir = str(debug_logs_dir.resolve())

        log_name = name.replace("-", "_") + ".log"
        log_file = debug_logs_dir / log_name
        log_file.touch()
        target.config.debug_log_file = str(log_file.resolve())
        target_dir = results_dir / f"target-{name}"
        target_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "target": name,
            "image": target.config.image,
            "container_id": None,
            "started_at": None,
            "finished_at": None,
            "n_queries": len(subset.query_indices),
            "total_timing_ms": 0,
            "errors": [],
        }

        try:
            info = target.start()
            manifest["container_id"] = info.get("container_id", "")
            manifest["started_at"] = info.get("started_at", "")
        except Exception as exc:
            manifest["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            manifest["errors"].append(f"Failed to start: {exc}")
            (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            aggregate["targets"][name] = manifest
            continue

        total_timing = 0.0
        has_target_errors = False

        for qi, query_index in enumerate(subset.query_indices):
            query_dir = target_dir / f"query_{qi:03d}"
            query_dir.mkdir(parents=True, exist_ok=True)

            query_annot = subset.annotations[query_index]
            db_indices = [i for i in range(len(subset.annotations)) if i != query_index]

            database_body = []
            for db_idx in db_indices:
                db_annot = subset.annotations[db_idx]
                database_body.append(
                    {
                        "aid": f"coco-annot-{db_annot.annot_id}",
                        "image_b64": base64.b64encode(db_annot.image).decode("utf-8"),
                        "bbox": list(db_annot.bbox),
                        "theta": 0.0,
                        "name_uuid": _make_name_uuid(db_annot.individual_ids),
                        "species": db_annot.species,
                    }
                )

            request_body = {
                "query_image_b64": base64.b64encode(query_annot.image).decode("utf-8"),
                "query_bbox": list(query_annot.bbox),
                "query_theta": 0.0,
                "query_species": query_annot.species,
                "database": database_body,
                "config": config,
            }

            (query_dir / "request.json").write_text(
                json.dumps(_strip_images(request_body), indent=2)
            )

            result: QueryResult = target.run_query(query_index, request_body)

            response_data = {
                "query_index": query_index,
                "error": result.error,
                "response": {
                    "annot_scores": result.annot_scores,
                    "timing_ms": result.timing_ms,
                },
                "raw_response": result.raw_response,
            }
            (query_dir / "response.json").write_text(
                json.dumps(response_data, indent=2)
            )

            if result.error:
                has_target_errors = True
                manifest["errors"].append(f"query_{qi:03d}: {result.error}")
            else:
                total_timing += result.timing_ms

        manifest["total_timing_ms"] = total_timing
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        if not getattr(target.config, "keep_containers", False):
            if hasattr(target, "capture_logs") and target.config.debug_log_file:
                target.capture_logs(target.config.debug_log_file)
            try:
                target.stop()
            except Exception as exc:
                manifest["errors"].append(f"Failed to stop: {exc}")

        aggregate["targets"][name] = manifest

    aggregate["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Write run metadata
    all_errors = []
    for m in aggregate["targets"].values():
        all_errors.extend(m.get("errors", []))
    write_run_metadata(
        results_dir=results_dir,
        subset=subset,
        targets=targets,
        config={**subset.config, **config},
        cli_args=cli_args,
        started_at=aggregate.get("started_at"),
        finished_at=aggregate.get("finished_at"),
        errors=all_errors,
    )

    return aggregate
