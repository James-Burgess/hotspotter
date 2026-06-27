#!/usr/bin/env python3
"""Run WBIA Docker images on a 50-image batch and collect traces.

For each WBIA image (nightly, latest, latest-local, develop):
  1. Launch a container with patches, batch, images, and output mounts
  2. Run the in-container trace recorder
  3. Collect the resulting parquet traces

Usage:
  python scripts/run_wbia_on_batch50.py [--images nightly latest develop]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFRA_ROOT = ROOT.parent
PATCHES_DIR = INFRA_ROOT / "patches"
DEFAULT_BATCH_JSON = INFRA_ROOT / "batches" / "zebra_coco.json"
DEFAULT_BATCH_IMAGE_DIR = INFRA_ROOT / "batches" / "images"
ARTIFACT_ROOT = INFRA_ROOT / "artifacts" / "wbia-oracle"

DEFAULT_IMAGES = [
    "wildme/wbia:nightly",
    "wildme/wbia:latest",
    "wildme/wbia:latest-local",
    "wildme/wbia:develop",
]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-")


def _write_incontainer_script(run_dir: Path, batch_path: str, image_dir: str) -> Path:
    """Copy the recorder script and patch paths for the 50-image batch."""
    src = PATCHES_DIR / "wbia_record_oracle_incontainer.py"
    dst = run_dir / "recorder.py"
    content = src.read_text()
    content = content.replace(
        'REFERENCE_BATCH = pathlib.Path("/input/pipeline-tests/reference_batch.json")',
        f'REFERENCE_BATCH = pathlib.Path("{batch_path}")',
    )
    content = content.replace(
        'IMAGE_DIR = pathlib.Path("/input/pipeline-tests/assets/images")',
        f'IMAGE_DIR = pathlib.Path("{image_dir}")',
    )
    dst.write_text(content)
    return dst


def _record_one_image(
    image: str, run_id: str, run_dir: Path, batch_json: Path, image_dir: Path
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    db_dir = run_dir / "db"
    db_dir.mkdir(exist_ok=True)

    recorder = _write_incontainer_script(
        run_dir,
        batch_path="/input/batch50.json",
        image_dir="/input/batch50_images",
    )

    manifest = {
        "run_id": run_id,
        "wbia_image": image,
        "batch_json": str(batch_json),
        "image_dir": str(image_dir),
        "started_unix": time.time(),
    }
    (run_dir / "manifest.host.json").write_text(json.dumps(manifest, indent=2))

    install_deps = (
        '/virtualenv/env3/bin/python3 -c "'
        "import importlib.util, subprocess, sys; "
        "missing=[pkg for pkg in ('pandas','pyarrow') if importlib.util.find_spec(pkg) is None]; "
        "subprocess.check_call([sys.executable,'-m','pip','install','--quiet',*missing]) if missing else None"
        '"'
    )

    cmd_parts = [
        'if [ "${WBIA_TRACE_INSTALL_DEPS:-1}" = "1" ]; then',
        install_deps + ";",
        "fi;",
        "/virtualenv/env3/bin/python3 /patches/patch_wbia_schema.py",
        "&& /virtualenv/env3/bin/python3 /patches/recorder.py",
    ]
    entrypoint_cmd = " ".join(cmd_parts)

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"wbia-oracle-{_safe_name(run_id)}",
        "-v",
        f"{PATCHES_DIR}/wbia_parquet_trace.py:/patches/wbia_parquet_trace.py:ro",
        "-v",
        f"{PATCHES_DIR}/patch_wbia_schema.py:/patches/patch_wbia_schema.py:ro",
        "-v",
        f"{recorder}:/patches/recorder.py:ro",
        "-v",
        f"{batch_json}:/input/batch50.json:ro",
        "-v",
        f"{image_dir}:/input/batch50_images:ro",
        "-v",
        f"{ARTIFACT_ROOT}:/artifacts/wbia-oracle",
        "-v",
        f"{db_dir}:/data/db",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        f"WBIA_TRACE_RUN_ID={run_id}",
        "-e",
        "WBIA_TRACE_DIR=/artifacts/wbia-oracle",
        "-e",
        "WBIA_TRACE_INSTALL_DEPS=1",
        "-e",
        "WBIA_ORACLE_DB_DIR=/data/db",
        "--entrypoint",
        "/bin/bash",
        image,
        "-lc",
        entrypoint_cmd,
    ]

    print(f"\n{'='*60}")
    print(f"Running {image} → {run_id}")
    print(f"{'='*60}")
    print("$", " ".join(docker_cmd[:16]), "...")
    sys.stdout.flush()

    result = subprocess.run(docker_cmd, cwd=ROOT, timeout=3600)

    manifest["exit_code"] = result.returncode
    manifest["completed_unix"] = time.time()
    (run_dir / "manifest.host.json").write_text(json.dumps(manifest, indent=2))

    if result.returncode != 0:
        print(f"FAILED: {image} (exit {result.returncode})", flush=True)
    else:
        print(f"SUCCESS: {image} → {run_dir}", flush=True)

    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    parser.add_argument(
        "--batch-json",
        type=Path,
        default=DEFAULT_BATCH_JSON,
        help="batch.json to run (default: zebra_coco.json)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_BATCH_IMAGE_DIR,
        help="flat image directory referenced by the batch",
    )
    parser.add_argument(
        "--run-id-prefix",
        default="batch50",
        help="prefix for the run-id/artifact dir (default: batch50)",
    )
    parser.add_argument("--keep-containers", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args(argv)

    batch_json = args.batch_json.resolve()
    image_dir = args.image_dir.resolve()

    if not batch_json.exists():
        print(f"ERROR: batch JSON not found: {batch_json}", file=sys.stderr)
        return 1

    if not image_dir.exists():
        print(f"ERROR: image dir not found: {image_dir}", file=sys.stderr)
        return 1

    print(f"Batch: {batch_json.name} ({batch_json.stat().st_size} bytes)")
    print(f"Images: {len(list(image_dir.glob('*.jpg')))} files")
    print(f"Artifacts → {ARTIFACT_ROOT}")
    print(f"Images to run: {len(args.images)}")
    for img in args.images:
        print(f"  {img}")
    sys.stdout.flush()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    failed = []
    for image in args.images:
        safe_img = _safe_name(image)
        run_id = f"{safe_img}-{args.run_id_prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = ARTIFACT_ROOT / run_id
        rc = _record_one_image(image, run_id, run_dir, batch_json, image_dir)
        if rc != 0:
            failed.append(image)

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED: {len(failed)}/{len(args.images)} images")
        for img in failed:
            print(f"  {img}")
        return 1
    else:
        print(f"ALL {len(args.images)} images completed successfully")
        print(f"Traces: {ARTIFACT_ROOT}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
