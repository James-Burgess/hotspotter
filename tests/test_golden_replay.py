"""Bit-exact replay of committed golden traces across all pipeline configs.

Each config in CONFIGS has a committed golden trace at
``tests/assets/golden_traces/{name}/``.  This test re-runs
:func:`identify` with the same config and asserts every pre-SV stage
matches bit-for-bit.

Pre-SV stages are deterministic (exact KNN + deterministic SIFT).
Post-SV is non-deterministic (OpenMP in sver.cpp) and excluded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.replay

_ASSETS = Path(__file__).resolve().parent / "assets"
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_fixture import build_database, load_batch  # noqa: E402
from hotspotter.config import HotSpotterConfig, IdentificationConfig  # noqa: E402
from hotspotter.pipeline import identify  # noqa: E402

STAGES = (
    "nearest_neighbors",
    "baseline_neighbor_filter",
    "neighbor_weights",
    "chipmatches_pre_sv",
)
GOLDEN = _ASSETS / "golden_traces"
BATCH = _ASSETS / "batch" / "reference_batch.json"
IMAGE_DIR = _ASSETS / "batch" / "images"

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
    "all_filters": {"fg_on": True, "bar_l2_on": True, "ratio_thresh": 0.8},
    "const_on": {"const_on": True},
    "lograt_on": {"lograt_on": True},
    "cos_on": {"cos_on": True},
    "lnbnn_normer": {"lnbnn_normer": "dummy", "lnbnn_norm_thresh": 0.01},
    "kpad_dynamic": {"kpad_policy": "dynamic"},
}


def _skip_if_missing() -> None:
    if not GOLDEN.is_dir():
        pytest.skip(f"golden traces missing: {GOLDEN}")
    if not BATCH.is_file():
        pytest.skip(f"batch missing: {BATCH}")


def _arrays(
    trace_dir: Path, stage: str, config_label: str, query_index: int
) -> dict[str, np.ndarray]:
    name = f"{config_label}_{query_index:06d}.parquet"
    parquet_path = trace_dir / stage / name
    if not parquet_path.is_file():
        return {}
    row = pd.read_parquet(parquet_path).iloc[0]
    out: dict[str, np.ndarray] = {}
    for key, value in row.items():
        if not isinstance(value, str):
            continue
        try:
            meta = json.loads(value)
        except (ValueError, TypeError):
            continue
        if "npy_path" not in meta:
            continue
        npy = trace_dir / stage / "arrays" / Path(meta["npy_path"]).name
        if npy.is_file():
            arr = np.load(npy, allow_pickle=True)
            if hasattr(arr, "keys"):
                arr = arr[list(arr.keys())[0]]
            out[key] = arr
    return out


_db_cache: tuple[list, list] | None = None


def _db_and_qidx():
    global _db_cache
    if _db_cache is None:
        _skip_if_missing()
        db, qidx, _ = build_database(load_batch(BATCH), IMAGE_DIR)
        _db_cache = (db, qidx)
    return _db_cache


@pytest.mark.slow
@pytest.mark.parametrize("config_name", list(CONFIGS.keys()))
def test_golden_replay(config_name: str, tmp_path: Path):
    """Re-run query 0 with each config and compare pre-SV stages bit-exact."""
    golden_dir = GOLDEN / config_name
    if not golden_dir.is_dir():
        pytest.skip(f"golden trace missing for config: {config_name}")

    db, qidx = _db_and_qidx()

    run_dir = tmp_path / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOTSPOTTER_TRACE_DIR"] = str(run_dir)
    os.environ["HOTSPOTTER_TRACE_RUN_ID"] = "replay"
    os.environ["HOTSPOTTER_TRACE_CONFIG_LABEL"] = config_name

    overrides = {"fg_on": False, "knn_backend": "exact"}
    overrides.update(CONFIGS[config_name])
    config = IdentificationConfig(hotspotter=HotSpotterConfig(**overrides))
    identify(qidx[0], db, config, trace_query_index=0)

    compared = 0
    for stage in STAGES:
        expected = _arrays(golden_dir, stage, config_name, 0)
        got = _arrays(run_dir, stage, config_name, 0)
        if not expected and not got:
            continue
        assert set(got) == set(
            expected
        ), f"{config_name}/{stage}: columns {set(got)} vs {set(expected)}"
        for key in expected:
            a, b = expected[key], got[key]
            assert (
                a.shape == b.shape
            ), f"{config_name}/{stage}/{key}: shape {a.shape} vs {b.shape}"
            assert np.array_equal(a, b), (
                f"{config_name}/{stage}/{key}: not bit-identical "
                f"maxdiff={np.abs(a.astype(float)-b.astype(float)).max()}"
            )
            compared += 1
    assert compared > 0, f"{config_name}: no stage arrays compared"
