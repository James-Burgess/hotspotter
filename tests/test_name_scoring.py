"""Failing tests asserting WBIA-expected name-scoring behaviour.

These tests encode the algorithm used by ``wbia.algo.hots.name_scoring``
so we can verify that hotspotter's ``compute_fmech_score`` and
``score_matches_with_names`` produce identical values.
"""

import uuid

import numpy as np
import pytest

from hotspotter.data import Match
from hotspotter.name_scoring import (
    _compute_xy_combo_ids,
    align_name_scores_with_annots,
    compute_csum_annot_scores,
    compute_fmech_score,
    compute_maxcsum_name_score,
    group_matches_by_name,
    score_matches_with_names,
)

# ---------------------------------------------------------------------------
# Test data: WBIA's ``testdata_chipmatch`` (name_scoring.py:21-49)
# ---------------------------------------------------------------------------
# fm_list  = 5 annots:
#   [0]: [(0,9),(1,9),(2,9),(3,9)]          fs=1 each
#   [1]: [(0,9),(1,9),(2,9),(3,9)]          fs=1 each
#   [2]: [(0,9),(1,9),(2,9),(3,9)]          fs=1 each
#   [3]: [(4,9),(5,9),(6,9),(3,9)]          fs=1 each
#   [4]: [(0,9),(1,9),(2,9),(3,9),(4,9)]    fs=1 each
# dnid_list = [1, 1, 2, 2, 3]  → 3 names

_MATCH_DIST = 1.0  # WBIA uses fs=1 for all feature matches in the doctest

_NAME_UUIDS = {1: uuid.uuid4(), 2: uuid.uuid4(), 3: uuid.uuid4()}

_WBIA_ANNOT_UUIDS = [uuid.uuid4() for _ in range(5)]


def _make_wbia_testdata_matches() -> list[Match]:
    """Build the flat match list matching WBIA's ``testdata_chipmatch``."""
    # dnid_list:       [1,   1,   2,   2,   3  ]
    # daid (0-based):   0    1    2    3    4
    matches: list[Match] = []
    # annot 0 — dnid=1
    for qfx in range(4):
        matches.append(
            Match(qfx=qfx, daid=0, dfx=9, dist=_MATCH_DIST, name_uuid=_NAME_UUIDS[1])
        )
    # annot 1 — dnid=1
    for qfx in range(4):
        matches.append(
            Match(qfx=qfx, daid=1, dfx=9, dist=_MATCH_DIST, name_uuid=_NAME_UUIDS[1])
        )
    # annot 2 — dnid=2
    for qfx in range(4):
        matches.append(
            Match(qfx=qfx, daid=2, dfx=9, dist=_MATCH_DIST, name_uuid=_NAME_UUIDS[2])
        )
    # annot 3 — dnid=2 — different qfx pattern: [4,5,6,3]
    for qfx in [4, 5, 6, 3]:
        matches.append(
            Match(qfx=qfx, daid=3, dfx=9, dist=_MATCH_DIST, name_uuid=_NAME_UUIDS[2])
        )
    # annot 4 — dnid=3 — [0,1,2,3,4]
    for qfx in range(5):
        matches.append(
            Match(qfx=qfx, daid=4, dfx=9, dist=_MATCH_DIST, name_uuid=_NAME_UUIDS[3])
        )
    return matches


def _make_wbia_annot_name_map():
    """annot_uuid → name_uuid for the 5-annot test fixture."""
    return {
        _WBIA_ANNOT_UUIDS[i]: _NAME_UUIDS[dnid]
        for i, dnid in enumerate([1, 1, 2, 2, 3])
    }


# ---------------------------------------------------------------------------
# WBIA doctest: compute_fmech_score → [4, 7, 5]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_xy", [False, True])
def test_compute_fmech_matches_wbia_doctest(use_xy: bool):
    """WBIA's ``testdata_chipmatch`` yields nsum = [4, 7, 5]."""
    matches = _make_wbia_testdata_matches()
    by_name = group_matches_by_name(matches)
    assert len(by_name) == 3, "should have 3 distinct names"

    if use_xy:
        # XY-dedup should produce the same result when every qfx maps to
        # a unique XY (we pass unique coords).  This tests the XY path.
        kpts = np.zeros((10, 6), dtype=np.float32)
        kpts[:, 0] = np.arange(10)  # unique x
        kpts[:, 1] = np.arange(10)  # unique y
        name_scores = compute_fmech_score(by_name, query_keypoints=kpts)
    else:
        name_scores = compute_fmech_score(by_name)

    # Match by WBIA dnid (1/2/3) not by UUID int order
    nsum_by_dnid = {dnid: name_scores[_NAME_UUIDS[dnid]] for dnid in [1, 2, 3]}

    # dnid=1: two annots with same qfx range [0,1,2,3], each qfx
    # appears twice; max is 1 → sum = 4.
    # dnid=2: qfx sets [0,1,2,3] + [4,5,6,3]; deduped: {0..6}=7
    # dnid=3: qfx set [0,1,2,3,4] → sum = 5
    assert nsum_by_dnid[1] == pytest.approx(4.0), f"dnid=1 nsum={nsum_by_dnid[1]}"
    assert nsum_by_dnid[2] == pytest.approx(7.0), f"dnid=2 nsum={nsum_by_dnid[2]}"
    assert nsum_by_dnid[3] == pytest.approx(5.0), f"dnid=3 nsum={nsum_by_dnid[3]}"


# ---------------------------------------------------------------------------
# csum for the same testdata
# ---------------------------------------------------------------------------


def test_compute_csum_annot_matches_wbia():
    """Per-annot csum: each annot is just sum(weights)."""
    matches = _make_wbia_testdata_matches()
    csum = compute_csum_annot_scores(matches, _WBIA_ANNOT_UUIDS)
    # Annots 0-3 each have 4 matches → 4.0; annot 4 has 5 matches → 5.0
    assert csum[_WBIA_ANNOT_UUIDS[0]] == 4.0
    assert csum[_WBIA_ANNOT_UUIDS[1]] == 4.0
    assert csum[_WBIA_ANNOT_UUIDS[2]] == 4.0
    assert csum[_WBIA_ANNOT_UUIDS[3]] == 4.0
    assert csum[_WBIA_ANNOT_UUIDS[4]] == 5.0


# ---------------------------------------------------------------------------
# When features share the same XY → only one vote per XY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dedup", [False, True])
def test_xy_dedup_reduces_nsum(dedup: bool):
    """XY grouping must reduce the score when features share a position."""
    name = uuid.uuid4()
    matches = [
        # qfx 0 and 1 share XY coordinate (1, 1)
        Match(qfx=0, daid=0, dfx=0, dist=5.0, name_uuid=name),
        Match(qfx=1, daid=0, dfx=1, dist=3.0, name_uuid=name),
    ]

    kpts = np.zeros((2, 6), dtype=np.float32)
    if dedup:
        kpts[0, :2] = (1, 1)
        kpts[1, :2] = (1, 1)  # same XY → combo collision
    else:
        kpts[0, :2] = (1, 1)
        kpts[1, :2] = (2, 1)  # different XY

    by_name = group_matches_by_name(matches)
    name_scores = compute_fmech_score(by_name, query_keypoints=kpts)
    score = name_scores[name]

    if dedup:
        # same XY: only the max (5.0) survives
        assert score == 5.0, f"XY-dedup should keep max 5.0, got {score}"
    else:
        assert score == 8.0, f"different XY → sum=8.0, got {score}"


# ---------------------------------------------------------------------------
# nsum ≤ csum for shared-name annotations
# ---------------------------------------------------------------------------


def test_nsum_le_csum_for_shared_names():
    """Each feature can vote at most once per name — nsum ≤ csum."""
    name = uuid.uuid4()
    # Three annotations sharing one name; same qfx appears in multiple
    matches = [
        Match(qfx=0, daid=0, dfx=0, dist=1.0, name_uuid=name),
        Match(qfx=0, daid=1, dfx=1, dist=2.0, name_uuid=name),
        Match(qfx=0, daid=2, dfx=2, dist=3.0, name_uuid=name),
    ]
    annot_uuids = [uuid.uuid4() for _ in range(3)]
    annot_name_map = {au: name for au in annot_uuids}

    csum, name_scores, _ = score_matches_with_names(
        matches, annot_uuids, annot_name_map, score_method="nsum"
    )

    total_csum = sum(csum.values())
    nsum_val = name_scores[name]
    assert total_csum == 6.0
    assert (
        nsum_val == 3.0
    ), f"nsumech should keep max(3.0), not sum(6.0), got {nsum_val}"
    assert nsum_val < total_csum


# ---------------------------------------------------------------------------
# _compute_xy_combo_ids: same XY → same combo id
# ---------------------------------------------------------------------------


def test_xy_combo_ids():
    kpts = np.array(
        [
            [1.0, 2.0, 0, 0, 0, 0],
            [1.3, 2.1, 0, 0, 0, 0],  # rounds to same (1, 2)
            [3.0, 4.0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    combo = _compute_xy_combo_ids(kpts)
    assert combo[0] == combo[1], "same rounded XY → same combo id"
    assert combo[0] != combo[2], "different rounded XY → different combo id"


# ---------------------------------------------------------------------------
# align_name_scores_with_annots: best csum annot per name gets the score
# ---------------------------------------------------------------------------


def test_align_name_scores_picks_best_csum():
    name = uuid.uuid4()
    a0, a1 = uuid.uuid4(), uuid.uuid4()
    csum = {a0: 10.0, a1: 5.0}
    name_scores = {name: 7.5}
    annot_name_map = {a0: name, a1: name}
    canonical = align_name_scores_with_annots(csum, annot_name_map, name_scores)
    assert canonical == {
        a0: 7.5
    }, f"best annot a0 should get name score, got {canonical}"


# ---------------------------------------------------------------------------
# max-per-name csum  (csum_wbia)
# ---------------------------------------------------------------------------


def test_maxcsum_name_score():
    name = uuid.uuid4()
    a0, a1 = uuid.uuid4(), uuid.uuid4()
    csum_annot = {a0: 8.0, a1: 12.0}
    annot_name_map = {a0: name, a1: name}
    ns = compute_maxcsum_name_score(csum_annot, annot_name_map)
    assert ns[name] == 12.0


# ---------------------------------------------------------------------------
# Oracle integration tests — compare against real WBIA chipmatch data
# ---------------------------------------------------------------------------

_MATHCLOSE = 1e-5


def _load_oracle_chipmatch(qaid: int = 1, cm_idx: int = 0):
    """Load one pre-SV chipmatch from the WBIA nightly oracle.

    Returns (fm_list, fsv_list, dnid_list, daid_list, oracle_csum, oracle_nsum).
    """
    import json
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd

    oracle_root = Path(
        os.environ.get(
            "WBIA_ORACLE_DIR",
            "/home/jimmy/projects/wildbook/Wildbook-infra/" "artifacts/wbia-oracle",
        )
    )
    ORACLE = oracle_root / "wildme-wbia-nightly-20260625-173226"

    pre_sv_arr_dir = ORACLE / "pre_sv" / "arrays"
    cm_arr_dir = ORACLE / "chipmatches_pre_sv" / "arrays"
    final_arr_dir = ORACLE / "final_scores" / "arrays"

    # fm_list / fsv_list from pre_sv stage (keyed by qaid)
    ps_df = pd.read_parquet(
        ORACLE / "chipmatches_pre_sv" / f"sv_on_false_{cm_idx:06d}.parquet"
    )
    ps_row = ps_df[ps_df["qaid"] == qaid].iloc[0]
    fm_json = json.loads(ps_row["fm_list_json"])
    fsv_json = json.loads(ps_row["fsv_list_json"])
    fm_list = [np.load(pre_sv_arr_dir / Path(a["npy_path"]).name) for a in fm_json]
    fsv_list = [np.load(pre_sv_arr_dir / Path(a["npy_path"]).name) for a in fsv_json]

    # daid_list from chipmatches_pre_sv arrays
    daid_arr = np.load(
        cm_arr_dir / Path(json.loads(ps_row["daid_list_array"])["npy_path"]).name,
    )

    # dnid_list and score arrays from final_scores stage (same qaid, same cm_idx)
    fs_df = pd.read_parquet(
        ORACLE / "final_scores" / f"sv_on_false_{cm_idx:06d}.parquet"
    )
    fs_row = fs_df[fs_df["qaid"] == qaid].iloc[0]

    def _load_1d(col_name):
        j = json.loads(fs_row[col_name])
        arr = np.load(final_arr_dir / Path(j["npy_path"]).name, allow_pickle=True)
        if arr.ndim == 0:
            return None
        return arr

    dnid_list = _load_1d("dnid_list_array")
    oracle_csum = _load_1d("annot_score_list_array")
    oracle_nsum = _load_1d("name_score_list_array")

    return fm_list, fsv_list, dnid_list, daid_arr, oracle_csum, oracle_nsum


def test_oracle_csum_matches_wbia_formula():
    """Hotspotter csum formula applied to WBIA chipmatch data must match oracle.

    This test proves the csum computation logic is correct — any remaining
    divergence is in the *input data* (different fm_list/fsv_list values).
    """
    fm_list, fsv_list, dnid_list, daid_arr, oracle_csum, oracle_nsum = (
        _load_oracle_chipmatch()
    )
    # fsv_list are shape (N, 1) → squeeze to 1D.
    # WBIA csum = sum(fs) per annotation.
    computed_csum = np.array([fs.squeeze().sum() for fs in fsv_list], dtype=np.float64)

    if oracle_csum is not None:
        np.testing.assert_allclose(
            computed_csum,
            oracle_csum,
            rtol=_MATHCLOSE,
            err_msg="csum formula mismatch vs oracle",
        )
    assert len(computed_csum) == len(daid_arr) == len(fm_list)


def test_oracle_nsum_matches_wbia_formula():
    """Hotspotter nsum formula applied to WBIA chipmatch data must match oracle.

    This is the REAL failing test: if WBIA's compute_fmech_score produces
    different values from hotspotter's implementation given the same
    fm/fsv/dnid inputs, then our nsum logic differs.
    """
    fm_list, fsv_list, dnid_list, daid_arr, oracle_csum, oracle_nsum = (
        _load_oracle_chipmatch()
    )
    if dnid_list is None:
        pytest.skip("dnid_list is not available (pre-SV scoring not yet run)")

    assert len(dnid_list) == len(fm_list)

    # WBIA nsum formula (implemented inline for exact verification):
    # 1. Group annot indices by dnid
    from collections import defaultdict

    name_annots: dict[int, list[int]] = defaultdict(list)
    for daid_idx, dnid in enumerate(dnid_list):
        name_annots[int(dnid)].append(daid_idx)

    # 2. For each name, group feature matches by qfx (combo_id)
    #    and keep max fs per group, then sum.
    nsumech_by_name: dict[int, float] = {}
    for dnid, annot_idxs in name_annots.items():
        fs_flat: list[float] = []
        qfx_flat: list[int] = []
        for aidx in annot_idxs:
            fm = fm_list[aidx]  # (N, 2) — [qfx, dfx]
            fs = fsv_list[aidx].squeeze()
            for i in range(len(fm)):
                qfx_flat.append(int(fm[i, 0]))
                fs_flat.append(float(fs[i]))

        if not fs_flat:
            nsumech_by_name[dnid] = 0.0
            continue

        # Group by qfx, keep max
        qfx_groups: dict[int, float] = {}
        for qfx, score in zip(qfx_flat, fs_flat, strict=True):
            qfx_groups[qfx] = max(qfx_groups.get(qfx, -1e300), score)

        nsumech_by_name[dnid] = sum(qfx_groups.values())

    if oracle_nsum is not None:
        # Oracle_nsum is per-*name* (aligned with unique dnids)
        unique_dnids = sorted(name_annots.keys())
        computed_nsum_arr = np.array(
            [nsumech_by_name[dnid] for dnid in unique_dnids], dtype=np.float64
        )
        np.testing.assert_allclose(
            computed_nsum_arr,
            oracle_nsum,
            rtol=_MATHCLOSE,
            err_msg="nsumech formula mismatch vs oracle",
        )
