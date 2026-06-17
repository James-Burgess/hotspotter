#!/usr/bin/env python3
"""CLI entry point for running COCO subset benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from coco.loader import CocoLoader
from compare import compare_results, print_detailed_report
from runner import run_benchmark
from targets.base import TargetConfig
from targets.core import CoreTargetRunner
from targets.wbia import WbiaTargetRunner

TARGET_MAP = {
    "wbia-core": (CoreTargetRunner, "wbia-core:latest"),
    "wbia-slim": (CoreTargetRunner, "wbia-slim:latest"),
    "wbia-latest": (WbiaTargetRunner, "wildme/wbia:latest"),
    "wbia-nightly": (WbiaTargetRunner, "wildme/wbia:nightly"),
    "wbia-develop": (WbiaTargetRunner, "wildme/wbia:develop"),
}

DEFAULT_CONFIG = {
    "pipeline_root": "vsmany",
    "K": 4,
    "Knorm": 1,
    "Kpad": 0,
    "kpad_policy": "dynamic",
    "score_method": "nsum",
    "normalizer_rule": "last",
    "can_match_samename": True,
    "fg_on": False,
    "bar_l2_on": False,
    "sv_on": False,
    "sv_n_name_shortlist": 40,
    "sv_n_annot_per_name": 3,
}


def _build_targets(
    target_names: list[str],
    base_port: int = 5000,
    keep_containers: bool = False,
):
    targets = []
    for i, name in enumerate(target_names):
        if name not in TARGET_MAP:
            print(f"Unknown target {name!r}, skipping", file=sys.stderr)
            continue
        runner_cls, image = TARGET_MAP[name]
        config = TargetConfig(name=name, image=image, port=base_port + i)
        config.keep_containers = keep_containers
        targets.append(runner_cls(config))
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Run COCO subset against one or more identification backends."
    )
    parser.add_argument(
        "--n-annots",
        type=int,
        default=100,
        help="Total annotations in subset (default: 100)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=10,
        help="Number of query annotations (default: 10)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Filter: giraffe_masai, zebra_plains, or all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic shuffle seed (default: 42)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["wbia-core"],
        choices=list(TARGET_MAP),
        metavar="TARGET",
        help="Targets to test (default: wbia-core)",
    )
    parser.add_argument(
        "--coco-json",
        type=str,
        default="tests/test-dataset/annotations/instances_train2020.json",
        help="Path to COCO JSON (default: tests/test-dataset/...)",
    )
    parser.add_argument(
        "--coco-images",
        type=str,
        default="tests/test-dataset/images/train2020",
        help="Path to COCO images directory (default: tests/test-dataset/...)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Output directory (default: test-run-results-<timestamp>)",
    )
    parser.add_argument(
        "--keep-containers",
        action="store_true",
        help="Don't stop containers after run",
    )
    parser.add_argument(
        "--flann-algorithm",
        type=str,
        default="kdtree",
        choices=["kdtree", "exact"],
        help="FLANN search algorithm (default: kdtree)",
    )

    args = parser.parse_args()

    # Load subset
    loader = CocoLoader(args.coco_json, args.coco_images)
    subset = loader.select_subset(
        n_annots=args.n_annots,
        species=args.species,
        seed=args.seed,
        n_queries=args.n_queries,
    )

    # Build targets
    targets = _build_targets(
        args.targets,
        keep_containers=args.keep_containers,
    )

    if not targets:
        print("No valid targets specified", file=sys.stderr)
        raise SystemExit(1)

    # Results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        ts = time.strftime("%Y%m%dT%H%M%S")
        results_dir = Path(f"test-run-results-{ts}")

    # Collect CLI args for metadata
    cli_args = {
        "n_annots": args.n_annots,
        "n_queries": args.n_queries,
        "species": args.species,
        "seed": args.seed,
        "targets": args.targets,
        "flann_algorithm": args.flann_algorithm,
        "results_dir": str(results_dir),
    }

    # Run
    print(
        f"Running {len(targets)} targets with {len(subset.annotations)} annotations, "
        f"{len(subset.query_indices)} queries → {results_dir}"
    )
    run_config = {**DEFAULT_CONFIG, "flann_algorithm": args.flann_algorithm}
    aggregate = run_benchmark(
        subset, targets, results_dir, run_config, cli_args=cli_args
    )

    # Debug logs are captured directly into results_dir/debug-logs/ by the targets
    debug_logs_dir = results_dir / "debug-logs"
    if debug_logs_dir.exists():
        for f in sorted(debug_logs_dir.iterdir()):
            print(f"Debug log: {f} ({f.stat().st_size:,} bytes)")

    # Write summary and print detailed report
    summary = compare_results(results_dir)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary → {summary_path}")
    print()
    print_detailed_report(results_dir)


if __name__ == "__main__":
    main()
