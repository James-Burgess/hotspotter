#!/usr/bin/env python3
"""Generate golden traces for all pipeline configs.

Run inside Docker:
    docker run --rm -v $(pwd):/app hotspotter:latest \
        python tests/generate_goldens.py

Writes full-stage traces to tests/assets/golden_traces/{config_name}/
for each config in CONFIGS. Only query 0 is traced (sufficient for
bit-exact regression — the pipeline is deterministic per-query).

Post-SV stages are now traced because SV is deterministic (serial,
no OpenMP). Configs with sv_on=False will produce identical pre-SV
and post-SV data (SV is a no-op).
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_fixture import build_database, load_batch
from hotspotter.config import HotSpotterConfig, IdentificationConfig
from hotspotter.pipeline import identify

_TESTS = Path(__file__).resolve().parent
BATCH = _TESTS / "test-dataset" / "reference_batch.json"
IMAGE_DIR = _TESTS / "test-dataset" / "images"
OUTPUT = _TESTS / "assets" / "golden_traces"

CONFIGS: dict[str, dict] = {
    # --- base / scoring ---
    "default": {},
    "fg_on": {"fg_on": True},
    "bar_l2": {"bar_l2_on": True},
    "ratio": {"ratio_thresh": 0.8},
    "normonly": {"normonly_on": True},
    "sqrd_dist": {"sqrd_dist_on": True},
    "no_samename": {"can_match_samename": False},
    "no_sameimg": {"can_match_sameimg": True},
    "all_filters": {"fg_on": True, "bar_l2_on": True, "ratio_thresh": 0.8},
    "const_on": {"const_on": True},
    "lograt_on": {"lograt_on": True},
    "cos_on": {"cos_on": True},
    # --- normalizer ---
    "normalizer_name": {"normalizer_rule": "name"},
    "lnbnn_normer": {"lnbnn_normer": "dummy", "lnbnn_norm_thresh": 0.01},
    "lnbnn_normer_05": {"lnbnn_normer": "dummy", "lnbnn_norm_thresh": 0.5},
    # --- score methods ---
    "csum": {"score_method": "csum"},
    "nsum_wbia": {"score_method": "nsum_wbia"},
    "csum_wbia": {"score_method": "csum_wbia"},
    "sumamech": {"score_method": "sumamech"},
    "rot_invariance": {"rotation_invariance": True, "score_method": "nsum_wbia"},
    # --- KNN / Kpad ---
    "knn_8": {"knn": 8},
    "knorm_2": {"knorm": 2},
    "kpad_3": {"kpad": 3, "kpad_policy": "fixed"},
    "kpad_dynamic": {"kpad_policy": "dynamic"},
    "lnbnn_ratio_08": {"lnbnn_ratio": 0.8},
    # --- query feature filters ---
    "minscale": {"minscale_thresh": 2.0},
    "maxscale": {"maxscale_thresh": 10.0},
    "fgw": {"fgw_thresh": 0.5},
    # --- spatial verification ---
    "sv_off": {"sv_on": False},
    "sv_xy_loose": {"sv_xy_thresh": 0.05},
    "sv_inliers_10": {"sv_min_n_inliers": 10},
    "sv_refine_affine": {"sv_refine_method": "affine"},
    "sv_no_full_checks": {"sv_full_homog_checks": False},
    "sv_abstain": {"sv_abstain_on_fail": True},
    "sv_no_weight": {"sv_weight_inliers": False},
    "sv_sver_weight": {"sv_sver_output_weighting": True},
    "sv_shortlist": {
        "sv_verify_all": False,
        "sv_n_name_shortlist": 5,
        "sv_n_annot_per_name": 2,
    },
    "prescore_csum": {"prescore_method": "csum"},
    # --- KNN backends ---
    "backend_linear": {"knn_backend": "linear"},
    "backend_faiss": {"knn_backend": "faiss"},
}

ALL_STAGES = (
    "nearest_neighbors",
    "baseline_neighbor_filter",
    "neighbor_weights",
    "chipmatches_pre_sv",
    "chipmatches_post_sv",
    "final_scores",
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

    non_golden_dirs = {d for d in trace_dir.iterdir() if d.name not in ALL_STAGES}
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
