#!/usr/bin/env python3
"""Compare hotspotter pipeline output against a WBIA oracle dump.

Runs ``hotspotter.identify()`` with all 7 canonical HotSpotter parameter
permutations against the same test images used by the WBIA oracle recorder,
writes parquet checkpoints to a temporary directory, then runs
``compare_wbia_oracles.py`` to produce a terminal summary and HTML report.

Usage::

    # Compare against the canonical nightly oracle
    python3 scripts/compare_to_wbia.py \
        ../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-105210

    # Point to different images / batch
    python3 scripts/compare_to_wbia.py <oracle_dir> \
        --image-dir ../pipeline/tests/assets/images \
        --batch ../pipeline/tests/reference_batch.json

    # Only run one config
    python3 scripts/compare_to_wbia.py <oracle_dir> --configs sv_on_true

Output::

    Terminal summary + HTML report in ---out (default: oracles/comparisons/<a>__vs__<b>.html).
    Exit code 0 = parity PASS (\u03c1 \u2265 0.97), 2 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

QUERY_CONFIGS: dict[str, dict] = {
    "sv_on_true": {"sv_on": True},
    "sv_on_false": {"sv_on": False},
    "sv_on_n20": {"sv_on": True, "num_return": 20},
    "K2": {"sv_on": True, "knn": 2},
    "K6": {"sv_on": True, "knn": 6},
    "score_csum": {"sv_on": True, "score_method": "csum_wbia"},
    "pre_csum": {"sv_on": True, "prescore_method": "csum_wbia"},
    "Knorm2": {"sv_on": True, "knorm": 2},
    "kpad_fixed_0": {"sv_on": True, "kpad_policy": "fixed", "kpad": 0},
}

ALL_CONFIGS = list(QUERY_CONFIGS.keys())


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_compare_script() -> Path:
    here = _find_repo_root()
    script = here.parent / "scripts" / "compare_wbia_oracles.py"
    if not script.exists():
        sys.exit(f"compare_wbia_oracles.py not found at {script}")
    return script


def _find_batch(batch_arg: str | None) -> Path:
    if batch_arg:
        return Path(batch_arg)
    here = _find_repo_root()
    default = here.parent / "pipeline" / "tests" / "reference_batch.json"
    if not default.exists():
        sys.exit(
            "Cannot find reference_batch.json. "
            "Pass --batch path or place it at ../pipeline/tests/reference_batch.json"
        )
    return default


def _find_image_dir(image_dir_arg: str | None, batch_path: Path) -> Path:
    if image_dir_arg:
        return Path(image_dir_arg)
    default = batch_path.parent / "assets" / "images"
    if not default.is_dir():
        sys.exit(f"Image directory not found: {default}. Pass --image-dir.")
    return default


def run_traces(
    trace_dir: Path,
    image_dir: Path,
    batch_path: Path,
    config_names: list[str],
) -> None:
    """Run hotspotter via Docker for each config, writing parquet traces."""
    repo_root = _find_repo_root()
    mount_image = f"{image_dir.resolve()}:/app/pipeline/tests/assets/images"
    mount_batch = f"{batch_path.resolve()}:/app/pipeline/tests/reference_batch.json"
    mount_trace = f"{trace_dir.resolve()}:/app/artifacts/hotspotter-trace"

    total = len(config_names)
    for idx, cfg_name in enumerate(config_names):
        cfg = dict(QUERY_CONFIGS[cfg_name])
        cfg.setdefault("fg_on", False)
        cfg.setdefault("kpad_policy", "dynamic")
        cfg.setdefault("knorm", 1)
        cfg_json = json.dumps(cfg)

        print(f"[{idx + 1}/{total}] {cfg_name:20s}  {cfg_json}", flush=True)
        t0 = time.monotonic()

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                mount_image,
                "-v",
                mount_batch,
                "-v",
                mount_trace,
                "-e",
                f"HOTSPOTTER_TRACE_DIR=/app/artifacts/hotspotter-trace",
                "-e",
                f"HOTSPOTTER_TRACE_RUN_ID=hotspotter-{cfg_name}",
                "-e",
                f"HOTSPOTTER_TRACE_CONFIG_LABEL={cfg_name}",
                "--entrypoint",
                "bash",
                "hotspotter:latest",
                "-c",
                (
                    "python scripts/run_fixture.py"
                    " /app/pipeline/tests/reference_batch.json"
                    " --image-dir /app/pipeline/tests/assets/images"
                    f" --config '{cfg_json}'"
                    " > /dev/null"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.monotonic() - t0
        if result.returncode == 0:
            print(f"        OK  ({elapsed:.1f}s)")
        else:
            print(f"        FAIL ({elapsed:.1f}s)")
            if result.stderr:
                lines = result.stderr.strip().splitlines()
                for line in lines[-5:]:
                    print(f"        {line}")
            sys.exit(1)


def compare(
    oracle_dir: Path, trace_dir: Path, passing_rho: float, out: Path | None
) -> int:
    """Run the comparison script and print the terminal output."""
    compare_script = _find_compare_script()

    cmd = [
        sys.executable,
        str(compare_script),
        str(oracle_dir),
        str(trace_dir),
        "--passing-rho",
        str(passing_rho),
    ]
    if out:
        cmd.extend(["--out", str(out)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run hotspotter with all configs, compare against WBIA oracle."
    )
    parser.add_argument("oracle_dir", help="Path to WBIA oracle dump directory")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=ALL_CONFIGS,
        help=f"Config names to run (default: all {len(ALL_CONFIGS)})",
    )
    parser.add_argument("--batch", help="Path to reference_batch.json")
    parser.add_argument("--image-dir", help="Directory containing test images")
    parser.add_argument("--out", help="HTML report output path (default: auto)")
    parser.add_argument(
        "--passing-rho",
        type=float,
        default=0.97,
        help="Minimum final name score Spearman ρ for parity PASS (default: 0.97)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip the trace run; only compare existing trace_dir",
    )
    parser.add_argument(
        "--trace-dir",
        help="Custom trace output directory (default: temp dir under /tmp)",
    )
    args = parser.parse_args()

    oracle_dir = Path(args.oracle_dir)
    if not oracle_dir.is_dir():
        sys.exit(f"Oracle directory not found: {oracle_dir}")

    batch_path = _find_batch(args.batch)
    image_dir = _find_image_dir(args.image_dir, batch_path)

    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
    else:
        import tempfile

        trace_dir = Path(tempfile.mkdtemp(prefix="hotspotter-trace-"))

    trace_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        run_traces(trace_dir, image_dir, batch_path, args.configs)

    print(f"\nComparing {oracle_dir.name} vs {trace_dir.name} ...\n", flush=True)

    return compare(oracle_dir, trace_dir, args.passing_rho, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
