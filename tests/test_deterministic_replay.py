"""Bit-exact replay of the recorded Hotspotter pipeline trace.

Hotspotter's ``exact`` KNN backend and deterministic Hessian-affine SIFT
extraction make every pipeline stage reproducible run-to-run. These tests
re-run :func:`identify` on the same batch that produced a golden trace and
assert each recorded stage matches bit-for-bit, giving a regression net that
screams if a refactor drifts any number.

Fixtures are located via environment variables and skipped when absent, so the
file is inert in the fast unit suite::

    HS_TRACE_DIR   golden trace root (.../hs-atrw200-<ts>)
    HS_BATCH_PATH  batch.json used to build the trace
    HS_IMAGE_DIR   chip directory referenced by the batch
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


def _trace_dir() -> Path:
    return Path(os.environ.get("HS_TRACE_DIR", "/artifacts/hs-atrw200-20260627-165148"))


def _batch_path() -> Path:
    return Path(os.environ.get("HS_BATCH_PATH", "/batches/atrw200.json"))


def _image_dir() -> Path:
    return Path(os.environ.get("HS_IMAGE_DIR", "/batches/atrw200_images"))


def _skip_if_missing() -> None:
    if not _trace_dir().is_dir():
        pytest.skip(f"golden trace not found: {_trace_dir()}", allow_module_level=False)
    if not _batch_path().is_file():
        pytest.skip(f"batch not found: {_batch_path()}")
    if not _image_dir().is_dir():
        pytest.skip(f"images not found: {_image_dir()}")


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
    db, qidx, _ = build_database(load_batch(_batch_path()), _image_dir())
    return db, qidx


@pytest.mark.slow
class TestReplaysGoldenTrace:
    """Re-run query 0 and diff every stage against the golden ATRW200 trace."""

    def test_query0_stages_match(self, tmp_path):
        db, qidx = _db_and_qidx()
        _run_with_trace(db, qidx[0], tmp_path)

        golden = _trace_dir()
        compared = 0
        for stage in STAGES:
            try:
                got = _arrays(tmp_path, stage, 0)
                expected = _arrays(golden, stage, 0)
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
