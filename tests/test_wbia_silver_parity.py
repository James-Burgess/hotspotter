"""WBIA silver-standard parity — HS final rankings vs WBIA final rankings.

HS-vs-HS bit-exact replay lives in ``test_deterministic_replay`` (the golden
net). This is the silver layer: does Hotspotter reach the same identification
*decisions* as WBIA on the same batch?

Both the HS golden trace and the WBIA silver trace are committed in
``tests/assets/`` (no external mounts needed). The WBIA trace only recorded
``final_scores`` (post-SV + FLANN, non-deterministic), so this is a
statistical decision-parity check by ``daid`` — not bit-exact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.parity

_ASSETS = Path(__file__).resolve().parent / "assets"
_HS_TRACE = _ASSETS / "hs_golden_trace"
_WBIA_TRACE = _ASSETS / "wbia_silver_trace"

HS_CONFIG = os.environ.get("HS_SILVER_CONFIG", "default")
WBIA_CONFIG = os.environ.get("WBIA_SILVER_CONFIG", "sv_on_true")


def _skip_if_missing() -> None:
    if not (_HS_TRACE / "final_scores").is_dir():
        pytest.skip(f"HS final_scores missing: {_HS_TRACE}")
    if not (_WBIA_TRACE / "final_scores").is_dir():
        pytest.skip(f"WBIA final_scores missing: {_WBIA_TRACE}")


def _load_array(trace: Path, stage: str, row, key) -> np.ndarray:
    meta = json.loads(row[key])
    return np.load(
        trace / stage / "arrays" / Path(meta["npy_path"]).name, allow_pickle=True
    )


def _ranking(trace: Path, stage: str, config_label: str, query_index: int) -> list[int]:
    fname = f"{config_label}_{query_index:06d}.parquet"
    row = pd.read_parquet(trace / stage / fname).iloc[0]
    daids = _load_array(trace, stage, row, "daid_list_array").astype(int)
    scores = np.atleast_1d(
        _load_array(trace, stage, row, "score_list_array").astype(float)
    )
    order = np.argsort(-scores, kind="stable")
    return [int(d) for d, s in zip(daids[order], scores[order]) if np.isfinite(s)]


def _discover_queries() -> list[int]:
    hs_dir = _HS_TRACE / "final_scores"
    wb_dir = _WBIA_TRACE / "final_scores"
    out = []
    for q in range(1000):
        if (hs_dir / f"{HS_CONFIG}_{q:06d}.parquet").is_file() and (
            wb_dir / f"{WBIA_CONFIG}_{q:06d}.parquet"
        ).is_file():
            out.append(q)
    return out


@pytest.mark.slow
class TestWbiaSilverParity:
    """HS vs WBIA final-ranking decision parity across the committed batch."""

    def test_top1_daid_agreement(self):
        _skip_if_missing()
        queries = _discover_queries()
        assert queries, "no query parquets common to both traces"
        agree = 0
        for q in queries:
            hs = _ranking(_HS_TRACE, "final_scores", HS_CONFIG, q)
            wb = _ranking(_WBIA_TRACE, "final_scores", WBIA_CONFIG, q)
            if hs and wb and hs[0] == wb[0]:
                agree += 1
        rate = agree / len(queries)
        print(f"\nTop-1 daid agreement: {agree}/{len(queries)} = {rate:.1%}")
        assert rate >= 0.70, f"HS/WBIA Top-1 daid agreement {rate:.1%} < 70%"

    def test_top5_overlap(self):
        _skip_if_missing()
        queries = _discover_queries()
        overlaps = []
        for q in queries:
            hs_rank = _ranking(_HS_TRACE, "final_scores", HS_CONFIG, q)
            wb_rank = _ranking(_WBIA_TRACE, "final_scores", WBIA_CONFIG, q)
            hs_set = set(hs_rank[:5])
            wb_set = set(wb_rank[:5])
            if hs_set or wb_set:
                overlaps.append(len(hs_set & wb_set) / len(hs_set | wb_set))
        mean = float(np.mean(overlaps)) if overlaps else 0.0
        print(f"\nTop-5 daid Jaccard overlap: mean={mean:.3f} (n={len(overlaps)})")
        assert mean >= 0.50, f"Top-5 overlap {mean:.3f} < 0.50"
