#!/usr/bin/env python3
"""Bottle web app for viewing benchmark results."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import bottle
from bottle import get, run, static_file

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
RESULTS_DIR = PROJECT_ROOT / "test-results"
COCO_JSON_PATH = (
    PROJECT_ROOT / "tests/test-dataset/annotations/instances_train2020.json"
)
COCO_IMAGES_DIR = PROJECT_ROOT / "tests/test-dataset/images/train2020"
CHIP_CACHE_DIR = HERE / ".chip_cache"

bottle.TEMPLATE_PATH.insert(0, str(HERE / "templates"))

TARGET_COLORS = {
    "wbia-core": "#6366f1",
    "wbia-latest": "#f59e0b",
    "wbia-nightly": "#a855f7",
    "wbia-develop": "#14b8a6",
    "wbia-slim": "#ae54fe",
}


def _target_color(name: str) -> str:
    return TARGET_COLORS.get(name, "#6b7280")


def _target_rgb(name: str) -> tuple[int, int, int]:
    h = _target_color(name).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ---------------------------------------------------------------------------
# COCO annotation → image mapping
# ---------------------------------------------------------------------------

_COCO_ANNOT_MAP: dict[int, dict[str, Any]] | None = None


def _load_coco_annot_map() -> dict[int, dict[str, Any]]:
    global _COCO_ANNOT_MAP
    if _COCO_ANNOT_MAP is not None:
        return _COCO_ANNOT_MAP

    _COCO_ANNOT_MAP = {}
    if not COCO_JSON_PATH.exists():
        return _COCO_ANNOT_MAP

    coco = json.loads(COCO_JSON_PATH.read_text())
    image_map = {img["id"]: img for img in coco["images"]}

    for ann in coco["annotations"]:
        img = image_map.get(ann["image_id"])
        if img is None:
            continue
        _COCO_ANNOT_MAP[ann["id"]] = {
            "file_name": img["file_name"],
            "bbox": [int(v) for v in ann["bbox"]],
            "image_id": ann["image_id"],
            "width": img["width"],
            "height": img["height"],
        }

    return _COCO_ANNOT_MAP


_IMAGE_NAME_MAP: dict[int, str] | None = None


def _load_image_name_map() -> dict[int, str]:
    global _IMAGE_NAME_MAP
    if _IMAGE_NAME_MAP is not None:
        return _IMAGE_NAME_MAP
    _IMAGE_NAME_MAP = {}
    if not COCO_JSON_PATH.exists():
        return _IMAGE_NAME_MAP
    coco = json.loads(COCO_JSON_PATH.read_text())
    for img in coco["images"]:
        _IMAGE_NAME_MAP[img["id"]] = img["file_name"]
    return _IMAGE_NAME_MAP


def _get_annot_info(annot_id: int) -> dict[str, Any] | None:
    return _load_coco_annot_map().get(annot_id)


def _parse_aid(aid: str) -> int | None:
    if not aid or not aid.startswith("coco-annot-"):
        return None
    try:
        return int(aid.replace("coco-annot-", ""))
    except ValueError:
        return None


def _chip_image_path(annot_id: int, size: int = 200) -> Path | None:
    info = _get_annot_info(annot_id)
    if info is None:
        return None
    key = f"{annot_id}_{size}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return CHIP_CACHE_DIR / f"{h}.jpg"


def _chip_image(annot_id: int, size: int = 300) -> bytes | None:
    info = _get_annot_info(annot_id)
    if info is None:
        return None

    cache_path = _chip_image_path(annot_id, size)
    if cache_path and cache_path.exists():
        return cache_path.read_bytes()

    img_path = COCO_IMAGES_DIR / info["file_name"]
    if not img_path.exists():
        return None

    from PIL import Image as PILImage

    img = PILImage.open(img_path)
    bx, by, bw, bh = info["bbox"]
    chip = img.crop((bx, by, bx + bw, by + bh))
    chip.thumbnail((size, size), PILImage.LANCZOS)

    buf = io.BytesIO()
    chip.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()

    if cache_path:
        CHIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(jpeg_bytes)

    return jpeg_bytes


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def _discover_runs() -> list[dict[str, Any]]:
    runs = []

    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir()):
            if not d.is_dir():
                continue
            if not (d / "config.json").exists():
                continue
            try:
                run_id = d.name
                run_json_path = d / "run.json"
                summary_path = d / "summary.json"
                config_path = d / "config.json"

                run_meta = (
                    json.loads(run_json_path.read_text())
                    if run_json_path.exists()
                    else {}
                )
                summary = (
                    json.loads(summary_path.read_text())
                    if summary_path.exists()
                    else {}
                )
                config = (
                    json.loads(config_path.read_text()) if config_path.exists() else {}
                )
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                continue

            date_str = (
                run_meta.get("created_at")
                or summary.get("run_id", "")
                or run_id.replace("test-run-results-", "")
            )

            duration = run_meta.get("duration_seconds")
            git_info = run_meta.get("git", {})
            status = run_meta.get("status", "unknown")

            runs.append(
                {
                    "id": run_id,
                    "name": run_id.replace("test-run-results-", ""),
                    "path": str(d),
                    "date": date_str,
                    "targets": run_meta.get("targets") or summary.get("targets", []),
                    "config": run_meta.get("config") or config,
                    "agreement": summary.get("agreement", {}),
                    "n_queries": run_meta.get("n_queries")
                    or len(summary.get("per_query", [])),
                    "n_annotations": run_meta.get("n_annotations"),
                    "n_errors": run_meta.get("n_errors", 0)
                    or len(summary.get("errors", [])),
                    "has_summary": summary_path.exists(),
                    "has_run_json": run_json_path.exists(),
                    "duration": duration,
                    "git": git_info,
                    "status": status,
                    "seed": run_meta.get("seed") or config.get("seed"),
                    "species": run_meta.get("species") or config.get("species"),
                }
            )

    # Sort by date descending — use run.json created_at or directory timestamp
    def _sort_key(r):
        ts = r.get("date", "")
        d = r.get("name", "")
        if ts:
            return ts
        return d

    runs.sort(key=_sort_key, reverse=True)

    return runs


def _get_compare_data(run_name: str) -> dict[str, Any] | None:
    run_path = RESULTS_DIR / run_name
    if not run_path.exists():
        return None

    summary_path = run_path / "summary.json"
    annot_path = run_path / "annotations.json"

    if not summary_path.exists() and not annot_path.exists():
        return {"status": "in_progress", "run_id": run_name}

    if not summary_path.exists():
        return {"error": "summary.json not found"}

    try:
        return json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": f"summary.json: {exc}"}


def _fmt_timestamp(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts


def _pass_fail(val: bool) -> str:
    return (
        '<span class="badge pass">PASS</span>'
        if val
        else '<span class="badge fail">FAIL</span>'
    )


def _get_log_files(run_name: str) -> list[dict[str, Any]]:
    run_path = RESULTS_DIR / run_name
    logs_dir = run_path / "debug-logs"
    if not logs_dir.exists():
        return []
    results = []
    for log_file in sorted(logs_dir.iterdir()):
        if log_file.suffix in (".log", ".txt") or log_file.is_file():
            size = log_file.stat().st_size
            results.append(
                {
                    "name": log_file.name,
                    "size": size,
                    "size_str": f"{size:,} bytes",
                }
            )
    return results


# ---------------------------------------------------------------------------
# Overlay image generation
# ---------------------------------------------------------------------------


def _draw_overlay(
    run_name: str, query_index: int, max_width: int = 700
) -> bytes | None:
    """Draw query bbox (white) + each target's top-1 match bbox (target color)."""
    run_path = RESULTS_DIR / run_name
    if not run_path.exists():
        return None

    annot_path = run_path / "annotations.json"
    summary_path = run_path / "summary.json"
    if not annot_path.exists() or not summary_path.exists():
        return None

    annots = json.loads(annot_path.read_text())
    summary = json.loads(summary_path.read_text())

    target_names = summary.get("targets", [])

    query_annot = None
    for a in annots:
        if a.get("is_query") and a.get("query_index") == query_index:
            query_annot = a
            break
    if query_annot is None:
        return None

    image_id = query_annot["image_id"]
    image_name_map = _load_image_name_map()
    file_name = image_name_map.get(image_id)
    if not file_name:
        return None

    img_path = COCO_IMAGES_DIR / file_name
    if not img_path.exists():
        return None

    targets_key = "-".join(sorted(target_names))
    cache_key = f"{run_name}_q{query_index}_{targets_key}_{max_width}"
    h = hashlib.md5(cache_key.encode()).hexdigest()[:16]
    cache_path = CHIP_CACHE_DIR / f"overlay_{h}.jpg"
    if cache_path.exists():
        return cache_path.read_bytes()

    from PIL import Image as PILImage, ImageDraw, ImageFont

    img = PILImage.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    annot_by_id = {a["annot_id"]: a for a in annots}

    per_query = {
        q["query_index"]: q.get("top1_aids", {}) for q in summary.get("per_query", [])
    }
    top1_by_target = per_query.get(query_index, {})

    for name in target_names:
        aid = top1_by_target.get(name)
        if aid is None:
            continue
        aid_int = _parse_aid(aid)
        if aid_int is None:
            continue
        entry = annot_by_id.get(aid_int)
        if entry is None:
            continue
        bx, by, bw, bh = entry["bbox"]
        color = _target_rgb(name)
        draw.rectangle([bx, by, bx + bw, by + bh], outline=color, width=5)
        _draw_label(draw, font, bx, by, f"{name}: #{aid_int}", color)

    qx, qy, qw, qh = query_annot["bbox"]
    draw.rectangle([qx, qy, qx + qw, qy + qh], outline=(255, 255, 255), width=5)
    _draw_label(draw, font, qx, qy, f"Q{query_index}", (200, 200, 200))

    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), PILImage.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    jpeg_bytes = buf.getvalue()

    CHIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(jpeg_bytes)

    return jpeg_bytes


def _draw_label(draw, font, bx: float, by: float, label: str, color: tuple):
    try:
        bbox_label = draw.textbbox((bx + 3, by - 18), label, font=font)
    except Exception:
        bbox_label = (bx + 3, by - 18, bx + 3 + len(label) * 7, by - 4)
    draw.rectangle(bbox_label, fill=(0, 0, 0, 180))
    draw.text((bx + 4, by - 17), label, fill=color, font=font)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@get("/")
def home():
    runs = _discover_runs()
    return bottle.template("home", runs=runs, pf=_pass_fail)


@get("/run/<run_name>")
def run_detail(run_name: str):
    data = _get_compare_data(run_name)
    if data is None:
        return bottle.template("error", message=f"Run '{run_name}' not found.")
    if "status" in data and data.get("status") == "in_progress":
        return bottle.template("in_progress", run_name=run_name)
    if "error" in data:
        return bottle.template("error", message=data["error"])
    log_files = _get_log_files(run_name)
    return bottle.template(
        "run",
        run_id=run_name,
        data=data,
        log_files=log_files,
        fmt_ts=_fmt_timestamp,
        pf=_pass_fail,
        tc=_target_color,
    )


@get("/run/<run_name>/data")
def run_data_json(run_name: str):
    data = _get_compare_data(run_name)
    if data is None:
        return {"error": "not found"}
    return data


@get("/run/<run_name>/logs/<log_name>")
def run_log(run_name: str, log_name: str):
    log_path = RESULTS_DIR / run_name / "debug-logs" / log_name
    if not log_path.exists():
        return bottle.HTTPResponse(status=404, body="Log not found")
    bottle.response.set_header("Content-Type", "text/plain; charset=utf-8")
    return log_path.read_text()


@get("/annot_image/<annot_id>")
def annot_image(annot_id: str):
    try:
        aid_int = int(annot_id)
    except ValueError:
        return bottle.HTTPResponse(status=404, body="Invalid annotation ID")

    jpeg = _chip_image(aid_int, size=300)
    if jpeg is None:
        return bottle.HTTPResponse(status=404, body="Image not found")

    bottle.response.set_header("Content-Type", "image/jpeg")
    bottle.response.set_header("Cache-Control", "public, max-age=86400")
    return jpeg


@get(r"/coco_image/<aid:re:coco-annot-\d+>")
def coco_annot_image(aid: str):
    annot_id = _parse_aid(aid)
    if annot_id is None:
        return bottle.HTTPResponse(status=404, body="Invalid annotation ID")
    return annot_image(str(annot_id))


@get("/overlay/<run_name>/<query_index:int>")
def overlay_image(run_name: str, query_index: int):
    jpeg = _draw_overlay(run_name, query_index)
    if jpeg is None:
        return bottle.HTTPResponse(status=404, body="Overlay not available")
    bottle.response.set_header("Content-Type", "image/jpeg")
    bottle.response.set_header("Cache-Control", "public, max-age=86400")
    return jpeg


@get("/static/<filename:path>")
def static(filename):
    return static_file(filename, root=str(HERE / "static"))


@get("/api/runs")
def runs_json():
    runs = _discover_runs()
    return {"runs": runs}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark web viewer")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--no-coco", action="store_true", help="Skip COCO image loading"
    )
    args = parser.parse_args()

    if not args.no_coco:
        n = len(_load_coco_annot_map())
        if n:
            print(f"Loaded {n} COCO annotations for image serving")
        else:
            print("Warning: COCO annotations not found, images will not be available")

    print(f"Starting benchmark viewer at http://{args.host}:{args.port}/")
    run(host=args.host, port=args.port, debug=args.debug, reloader=args.debug)


if __name__ == "__main__":
    main()
