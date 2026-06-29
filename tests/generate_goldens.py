#!/usr/bin/env python3
"""Generate golden traces for all pipeline configs.

Run inside Docker:
    docker run --rm -v $(pwd):/app hotspotter:latest \
        python tests/generate_goldens.py

Writes pre-SV stage traces to tests/assets/golden_traces/{config_name}/
for each config in CONFIGS. Only query 0 is traced (sufficient for
bit-exact regression — the pipeline is deterministic per-query).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_fixture import build_database, load_batch
from hotspotter.config import HotSpotterConfig, IdentificationConfig
from hotspotter.pipeline import identify

BATCH = Path(__file__).resolve().parent / "assets" / "batch" / "reference_batch.json"
IMAGE_DIR = Path(__file__).resolve().parent / "assets" / "batch" / "images"
OUTPUT = Path(__file__).resolve().parent / "assets" / "golden_traces"

CONFIGS: dict[str, dict] = {
    "default": {},
    "fg_on": {"fg_on": True},
    "bar_l2": {"bar_l2_on": True},
    "ratio": {"ratio_thresh": 0.8},
    "normonly": {"normonly_on": True},
    "normalizer_name": {"normalizer_rule": "name"},
    "sqrd_dist": {"sqrd_dist_on": True},
    "no_samename": {"can_match_samename": False},
    "no_sameimg": {"can_match_sameimg": True},
    "csum": {"score_method": "csum"},
    "nsum_wbia": {"score_method": "nsum_wbia"},
    "csum_wbia": {"score_method": "csum_wbia"},
    "sumamech": {"score_method": "sumamech"},
    "rot_invariance": {"rotation_invariance": True, "score_method": "nsum_wbia"},
    "sv_off": {"sv_on": False},
    "all_filters": {
        "fg_on": True,
        "bar_l2_on": True,
        "ratio_thresh": 0.8,
    },
}

PRE_SV_STAGES = (
    "nearest_neighbors",
    "baseline_neighbor_filter",
    "neighbor_weights",
    "chipmatches_pre_sv",
)


def generate_one(database, qidx, name: str, overrides: dict) -> None:
    trace_dir = OUTPUT / name
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HOTSPOTTER_TRACE_DIR"] = str(trace_dir)
    os.environ["HOTSPOTTER_TRACE_RUN_ID"] = "golden"
    os.environ["HOTSPOTTER_TRACE_CONFIG_LABEL"] = name

    hs_kwargs = {"fg_on": False, "knn_backend": "exact"}
    hs_kwargs.update(overrides)
    config = IdentificationConfig(hotspotter=HotSpotterConfig(**hs_kwargs))

    identify(qidx[0], database, config, trace_query_index=0)

    non_golden_dirs = {d for d in trace_dir.iterdir() if d.name not in PRE_SV_STAGES}
    for d in non_golden_dirs:
        if d.is_dir():
            shutil.rmtree(d)
        else:
            d.unlink()

    size = sum(f.stat().st_size for f in trace_dir.rglob("*") if f.is_file())
    print(f"  {name}: {size / 1e6:.1f} MB")


def main():
    print("Loading reference batch...")
    db, qidx, _ = build_database(load_batch(BATCH), IMAGE_DIR)
    print(f"  {len(db)} annots, {len(qidx)} queries")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, overrides in CONFIGS.items():
        t0 = time.time()
        print(f"Generating golden: {name} ({overrides})")
        generate_one(db, qidx, name, overrides)
        print(f"  done in {time.time() - t0:.1f}s")

    print(f"\nAll goldens written to {OUTPUT}/")


if __name__ == "__main__":
    main()
