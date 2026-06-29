"""Bit-exact replay of the committed Hotspotter golden trace.

Hotspotter's ``exact`` KNN backend and deterministic Hessian-affine SIFT
extraction make every pre-SV pipeline stage reproducible run-to-run. This test
re-runs :func:`identify` on the committed reference batch and asserts each
recorded stage matches the committed golden trace bit-for-bit — a regression
net that screams if a refactor drifts any number.

The golden trace was generated from ``run_fixture.py`` on the 19-annot COCO
zebra reference batch (3 queries). The trace is committed at
``tests/assets/hs_golden_trace/``; the batch + images at
``tests/assets/batch/``. No external mounts are needed.
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
TRACE = _ASSETS / "hs_golden_trace"
BATCH = _ASSETS / "batch" / "reference_batch.json"
IMAGE_DIR = _ASSETS / "batch" / "images"


def _skip_if_missing() -> None:
    if not TRACE.is_dir():
        pytest.skip(f"golden trace missing: {TRACE}")
    if not BATCH.is_file():
        pytest.skip(f"batch missing: {BATCH}")
    if not IMAGE_DIR.is_dir():
        pytest.skip(f"images missing: {IMAGE_DIR}")


def _parquet(trace: Path, stage: str, query_index: int) -> pd.DataFrame:
    name = f"default_{query_index:06d}.parquet"
    return pd.read_parquet(trace / stage / name)


def _arrays(trace: Path, stage: str, query_index: int) -> dict[str, np.ndarray]:
    row = _parquet(trace, stage, query_index).iloc[0]
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
        npy = trace / stage / "arrays" / Path(meta["npy_path"]).name
        out[key] = np.load(npy, allow_pickle=True)
    return out


def _run_with_trace(database, qidx, trace_dir: Path) -> None:
    os.environ["HOTSPOTTER_TRACE_DIR"] = str(trace_dir)
    os.environ["HOTSPOTTER_TRACE_RUN_ID"] = "replay"
    os.environ["HOTSPOTTER_TRACE_CONFIG_LABEL"] = "default"
    config = IdentificationConfig(
        hotspotter=HotSpotterConfig(fg_on=False, knn_backend="exact")
    )
    identify(qidx, database, config, trace_query_index=0)


def _db_and_qidx():
    _skip_if_missing()
    db, qidx, _ = build_database(load_batch(BATCH), IMAGE_DIR)
    return db, qidx


@pytest.mark.slow
class TestReplaysGoldenTrace:
    """Re-run query 0 and diff every deterministic stage against the
    committed golden trace from tests/assets/hs_golden_trace/."""

    def test_query0_stages_match(self, tmp_path):
        db, qidx = _db_and_qidx()
        _run_with_trace(db, qidx[0], tmp_path)

        golden_stages = [
            s for s in STAGES if (TRACE / s / "default_000000.parquet").is_file()
        ]
        if not golden_stages:
            pytest.skip(
                f"golden trace at {TRACE} has no pre-SV stage parquets "
                "(pruned to final_scores only) — cannot run bit-exact replay"
            )

        compared = 0
        for stage in STAGES:
            try:
                got = _arrays(tmp_path, stage, 0)
                expected = _arrays(TRACE, stage, 0)
            except FileNotFoundError:
                continue
            assert set(got) == set(
                expected
            ), f"{stage}: column mismatch {set(got)} vs {set(expected)}"
            for key in expected:
                a, b = expected[key], got[key]
                assert (
                    a.shape == b.shape
                ), f"{stage}/{key}: shape {a.shape} vs {b.shape}"
                assert np.array_equal(a, b), (
                    f"{stage}/{key}: not bit-identical "
                    f"maxdiff={np.abs(a.astype(float)-b.astype(float)).max()}"
                )
                compared += 1
        assert compared > 0, "no stage arrays were compared"
