#!/usr/bin/env python3
"""N-way parity: N WBIA images vs hotspotter.

Records the *baseline* config (``sv_on_true``) against every WBIA image,
runs hotspotter once, then compares all pairs:

  Apple-Apple:  every pair of WBIA oracles (proves WBIA determinism)
  Apple-Orange: every WBIA oracle vs hotspotter (parity gate)

Environment:
  ORACLE_DIR — parent for recorded oracles (default: ../artifacts/wbia-oracle)
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_PARENT = ROOT.parent / "scripts"
ORACLE_DIR = Path(
    __import__("os").environ.get(
        "ORACLE_DIR", str(ROOT.parent / "artifacts" / "wbia-oracle")
    )
)

BASELINE_CONFIG = "sv_on_true"


def _run(cmd: list[str], timeout: int = 1800) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, timeout=timeout)


def record_oracle(image: str, algorithm: str = "linear") -> Path:
    """Record a WBIA oracle for *image* with the baseline config."""
    record_script = SCRIPTS_PARENT / "record_wbia_oracle.py"
    if not record_script.exists():
        sys.exit(f"recorder not found: {record_script}")

    _run(
        [
            sys.executable,
            str(record_script),
            "--images",
            image,
            "--algorithm",
            algorithm,
            "--configs",
            BASELINE_CONFIG,
        ],
        timeout=3600,
    )

    dirs = sorted(
        [
            d
            for d in ORACLE_DIR.iterdir()
            if d.is_dir() and image.replace(":", "-") in d.name
        ],
        reverse=True,
    )
    if not dirs:
        sys.exit(f"Oracle recording failed — no output directory found for {image}")
    latest = dirs[0]
    print(f"Recorded {image} → {latest.name}")
    return latest


def run_hotspotter(oracle_dir: Path) -> Path:
    """Run hotspotter traces and compare against *oracle_dir*."""
    import tempfile

    compare_script = ROOT / "scripts" / "compare_to_wbia.py"
    trace_dir = Path(tempfile.mkdtemp(prefix="hotspotter-parity-"))

    _run(
        [
            sys.executable,
            str(compare_script),
            str(oracle_dir),
            "--backends",
            "linear",
            "--configs",
            BASELINE_CONFIG,
            "--trace-dir",
            str(trace_dir),
            "--passing-rho",
            "0.999",
        ],
        timeout=600,
    )

    return trace_dir / "linear"


def compare_pair(label: str, oracle_a: Path, oracle_b: Path) -> int:
    """Run comparer on two oracles; return exit code (0=PASS, 2=FAIL)."""
    compare_script = SCRIPTS_PARENT / "compare_wbia_oracles.py"
    out = ORACLE_DIR / "comparisons" / f"{oracle_a.name}__vs__{oracle_b.name}.html"

    result = subprocess.run(
        [
            sys.executable,
            str(compare_script),
            str(oracle_a),
            str(oracle_b),
            "--passing-fm-jaccard",
            "0.999",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    verdict = "PASS" if result.returncode == 0 else "FAIL"
    print(f"  [{label}] {verdict}  {out}")
    return result.returncode


def _short_name(oracle: Path) -> str:
    name = oracle.name
    for prefix in ("wildme-wbia-", "wildme-", "wbia-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-record",
        action="store_true",
        help="Skip oracle recording (use existing oracles).",
    )
    parser.add_argument(
        "--wbia-images",
        nargs="*",
        default=[
            "wildme/wbia:nightly",
            "wildme/wbia:latest",
            "wildme/wbia:develop",
        ],
        help="WBIA images to record (default: nightly latest develop).",
    )
    args = parser.parse_args()

    t0 = time.monotonic()

    # --- Phase 1: Record all WBIA oracles ---
    oracles: list[tuple[str, Path]] = []
    if not args.skip_record:
        for i, image in enumerate(args.wbia_images):
            short = image.split("/")[-1].split(":")[-1]
            print(f"\n=== Phase 1.{i+1}: Record {image} ===", flush=True)
            oracle_path = record_oracle(image)
            oracles.append((short, oracle_path))
    else:
        dirs = sorted(
            [
                d
                for d in ORACLE_DIR.iterdir()
                if d.is_dir() and d.name.startswith("wildme-wbia-")
            ],
        )
        if len(dirs) < 2:
            sys.exit(
                f"Need at least 2 WBIA oracle dirs in {ORACLE_DIR}; found {len(dirs)}"
            )
        matched = []
        for image in args.wbia_images:
            tag = image.split(":")[-1]
            for d in reversed(dirs):
                name = d.name
                if (
                    name.startswith(f"wildme-wbia-{tag}-")
                    and name[len(f"wildme-wbia-{tag}-")].isdigit()
                ):
                    if d not in [p for _, p in matched]:
                        matched.append((tag, d))
                        break
        if not matched:
            for d in reversed(dirs):
                matched.append((_short_name(d), d))
                if len(matched) >= len(args.wbia_images):
                    break
        oracles = matched
        print(f"Using existing oracles:")
        for name, path in oracles:
            print(f"  {name}: {path.name}")

    # --- Phase 2: Apple-Apple (WBIA vs WBIA pairwise) ---
    results: dict[str, int] = {}
    for (name_a, path_a), (name_b, path_b) in itertools.combinations(oracles, 2):
        label = f"WBIA:{name_a} vs WBIA:{name_b}"
        print(f"\n=== Apple-Apple: {label} ===", flush=True)
        results[label] = compare_pair(label, path_a, path_b)

    # --- Phase 3: Run hotspotter once ---
    print(f"\n=== Apple-Orange: hotspotter vs {oracles[0][0]} ===", flush=True)
    hs_trace = run_hotspotter(oracles[0][1])

    # --- Phase 4: Apple-Orange (every WBIA vs hotspotter) ---
    for name, path in oracles:
        label = f"WBIA:{name} vs hotspotter"
        print(f"\n=== Apple-Orange: {label} ===", flush=True)
        results[label] = compare_pair(label, path, hs_trace)

    # --- Summary ---
    elapsed = time.monotonic() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("\n========== PARITY SUMMARY ==========")
    for label in sorted(results.keys()):
        code = results[label]
        verdict = "PASS" if code == 0 else "FAIL"
        print(f"  {label:45s}: {verdict}")

    return max(results.values()) if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
