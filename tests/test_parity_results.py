"""Compare hotspotter live identification results against WBIA oracle.

Runs the full pipeline for all queries and compares final rankings
against committed oracle data in ``tests/assets/oracle/final_scores/``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hotspotter.chip import extract_chip
from hotspotter.config import HotSpotterConfig, IdentificationConfig, SiftConfig
from hotspotter.data import AnnotatedImage
from hotspotter.features import extract_features
from hotspotter.pipeline import identify

pytestmark = pytest.mark.parity

_QUERIES = 3
_QUERY_INDICES = [0, 5, 16]
_ASSETS = Path(__file__).resolve().parent / "assets"
_DATASET = Path(__file__).resolve().parent / "test-dataset"

_hs_cache: dict[int, list] = {}
_oracle_ranking_cache: dict[int, list[tuple[str, str, float]]] = {}
_oracle_raw_cache: dict[int, dict] = {}


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float) + 1.0
    ry = np.argsort(np.argsort(y)).astype(float) + 1.0
    d2 = (rx - ry) ** 2
    return float(1.0 - (6.0 * d2.sum()) / (n * (n**2 - 1)))


def _oracle_dir() -> Path:
    return _ASSETS / "oracle"


def _batch_path() -> Path:
    return _DATASET / "reference_batch.json"


def _image_dir() -> Path:
    return _DATASET / "images"


def _build_database() -> list[AnnotatedImage]:
    import cv2

    batch_path = _batch_path()
    image_dir = _image_dir()
    with open(batch_path) as f:
        batch = json.load(f)

    annots = batch["annotations"]
    name_to_uuid: dict[str, uuid.UUID] = {}
    seen_ids: set[str] = set()
    database: list[AnnotatedImage] = []

    for ann in annots:
        if ann["annot_id"] in seen_ids:
            continue
        img_path = image_dir / ann["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        bbox = ann["bbox"]
        chip = extract_chip(img, bbox)
        features = extract_features(chip, SiftConfig())
        individual_ids = ann.get("individual_ids", [])
        nid = str(individual_ids[0]) if individual_ids else f"name_{ann['annot_id']}"
        if nid not in name_to_uuid:
            name_to_uuid[nid] = uuid.uuid5(uuid.NAMESPACE_DNS, nid)
        database.append(
            AnnotatedImage(
                annot_uuid=uuid.uuid5(uuid.NAMESPACE_DNS, f"annot_{ann['annot_id']}"),
                name_uuid=name_to_uuid[nid],
                image=chip,
                features=features,
                bbox=tuple(bbox),
            )
        )
        seen_ids.add(ann["annot_id"])

    return database


def _load_oracle_arr(row, key, oracle, stage="final_scores"):
    meta = json.loads(row[key])
    fname = Path(meta["npy_path"]).name
    arrays_dir = oracle / stage / "arrays"
    path = arrays_dir / fname
    if not path.exists():
        stem = fname.rsplit(".", 1)[0]
        path = arrays_dir / f"{stem}.npz"
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        data = data[data.files[0]]
    return data


def _get_hs_results(database, qidx):
    if qidx not in _hs_cache:
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(
                sv_on=True, fg_on=False, num_return=50, knn_backend="linear"
            )
        )
        db_index = _QUERY_INDICES[qidx]
        _hs_cache[qidx] = identify(db_index, database, config)
    return _hs_cache[qidx]


def _get_oracle_ranking(qidx, annot_uuids, name_uuids):
    if qidx not in _oracle_ranking_cache:
        oracle = _oracle_dir()
        fname = f"sv_on_true_{qidx:06d}.parquet"
        df = pd.read_parquet(oracle / "final_scores" / fname)
        row = df.iloc[0]
        daids = _load_oracle_arr(row, "daid_list_array", oracle).astype(int)
        scores = np.atleast_1d(
            np.asarray(_load_oracle_arr(row, "score_list_array", oracle), dtype=float)
        )
        ranking: list[tuple[str, str, float]] = []
        for daid, score in zip(daids, scores):
            if np.isfinite(score) and score > -1e300:
                db_idx = int(daid) - 1
                auuid = (
                    annot_uuids[db_idx] if 0 <= db_idx < len(annot_uuids) else str(daid)
                )
                nuuid = (
                    name_uuids[db_idx] if 0 <= db_idx < len(name_uuids) else str(-daid)
                )
                ranking.append((auuid, nuuid, float(score)))
        ranking.sort(key=lambda x: x[2], reverse=True)
        _oracle_ranking_cache[qidx] = ranking
    return _oracle_ranking_cache[qidx]


def _get_oracle_final_row(qidx):
    if qidx not in _oracle_raw_cache:
        oracle = _oracle_dir()
        fname = f"sv_on_true_{qidx:06d}.parquet"
        df = pd.read_parquet(oracle / "final_scores" / fname)
        _oracle_raw_cache[qidx] = {
            "row": df.iloc[0],
            "oracle": oracle,
        }
    return _oracle_raw_cache[qidx]


class TestParityResults:
    """Compare hotspotter live results against WBIA oracle for all queries."""

    @classmethod
    @pytest.fixture(scope="class")
    def database(cls) -> list[AnnotatedImage]:
        oracle = _oracle_dir()
        if not oracle.exists():
            pytest.skip(f"Oracle not found: {oracle}")
        batch_path = _batch_path()
        if not batch_path.exists():
            pytest.skip(f"Reference batch not found: {batch_path}")
        return _build_database()

    @classmethod
    @pytest.fixture(scope="class")
    def annot_uuids(cls, database) -> list[str]:
        return [str(a.annot_uuid) for a in database]

    @classmethod
    @pytest.fixture(scope="class")
    def name_uuids(cls, database) -> list[str]:
        return [str(a.name_uuid) if a.name_uuid else "" for a in database]

    @classmethod
    @pytest.fixture(scope="class")
    def results_for_query(cls, database) -> dict[int, list]:
        return {q: _get_hs_results(database, q) for q in range(_QUERIES)}

    @classmethod
    @pytest.fixture(scope="class")
    def oracle_for_query(
        cls, annot_uuids, name_uuids
    ) -> dict[int, list[tuple[str, str, float]]]:
        return {
            q: _get_oracle_ranking(q, annot_uuids, name_uuids) for q in range(_QUERIES)
        }

    @pytest.mark.parametrize("qidx", range(_QUERIES))
    def test_top_5_annot_overlap(
        self, database, results_for_query, oracle_for_query, qidx
    ):
        hs = results_for_query[qidx]
        wb = oracle_for_query[qidx]
        wbia_top = {aid for aid, _, _ in wb[:5]}
        hs_top = {
            str(r.annot_uuid)
            for r in hs[:10]
            if np.isfinite(r.score) and r.score > -1e300
        }
        assert len(wbia_top & hs_top) > 0, f"Q{qidx}: no annot overlap"

    @pytest.mark.parametrize("qidx", range(_QUERIES))
    def test_top_3_name_overlap(
        self, database, results_for_query, oracle_for_query, qidx
    ):
        hs = results_for_query[qidx]
        wb = oracle_for_query[qidx]
        wbia_top_names = set()
        for _, nid, _ in wb:
            if nid:
                wbia_top_names.add(nid)
            if len(wbia_top_names) >= 3:
                break
        hs_top_names: set[str] = set()
        for r in hs:
            if r.name_uuid is not None and np.isfinite(r.score) and r.score > -1e300:
                hs_top_names.add(str(r.name_uuid))
            if len(hs_top_names) >= 5:
                break
        assert len(wbia_top_names & hs_top_names) > 0, f"Q{qidx}: no name overlap"

    @pytest.mark.parametrize("qidx", range(_QUERIES))
    def test_rank_reciprocal_overlap(
        self, database, results_for_query, oracle_for_query, qidx
    ):
        hs = results_for_query[qidx]
        wb = oracle_for_query[qidx]
        wbia_rank = {aid: i + 1 for i, (aid, _, _) in enumerate(wb)}
        rro = 0.0
        for r in hs[:5]:
            if not np.isfinite(r.score) or r.score <= -1e300:
                continue
            if str(r.annot_uuid) in wbia_rank:
                rro += 1.0 / wbia_rank[str(r.annot_uuid)]
        assert rro > 0.0, f"Q{qidx}: rank-reciprocal overlap = {rro:.4f}"

    @pytest.mark.parametrize("qidx", range(_QUERIES))
    def test_top_name_rank(self, database, results_for_query, oracle_for_query, qidx):
        hs = results_for_query[qidx]
        wb = oracle_for_query[qidx]
        hs_top_name = None
        for r in hs[:5]:
            if r.name_uuid is not None and np.isfinite(r.score) and r.score > -1e300:
                hs_top_name = str(r.name_uuid)
                break
        if hs_top_name is None:
            pytest.skip(f"Q{qidx}: no valid named result in HS top-5")
        wbia_top_3_names = set()
        for _, nid, _ in wb:
            if nid:
                wbia_top_3_names.add(nid)
            if len(wbia_top_3_names) >= 3:
                break
        assert (
            hs_top_name in wbia_top_3_names
        ), f"Q{qidx}: HS top name not in WBIA top-3"

    @pytest.mark.parametrize("qidx", range(_QUERIES))
    def test_result_count(self, results_for_query, qidx):
        hs = results_for_query[qidx]
        valid = [r for r in hs if np.isfinite(r.score) and r.score > -1e300]
        assert len(valid) >= 1, f"Q{qidx}: no valid results"

    @pytest.mark.parametrize("qidx", range(_QUERIES))
    def test_scores_monotonic(self, results_for_query, qidx):
        hs = results_for_query[qidx]
        prev = float("inf")
        for r in hs:
            if not np.isfinite(r.score) or r.score <= -1e300:
                continue
            if r.score > prev:
                pytest.fail(f"Q{qidx}: score increased: {prev} → {r.score}")
            prev = r.score

    @pytest.mark.parametrize("qidx", range(_QUERIES))
    def test_daid_aware_annot_csum_correlation(self, database, results_for_query, qidx):
        hs = results_for_query[qidx]
        raw = _get_oracle_final_row(qidx)
        oracle = raw["oracle"]
        row = raw["row"]
        wb_daids = _load_oracle_arr(row, "daid_list_array", oracle).astype(int)
        wb_csum = _load_oracle_arr(row, "annot_score_list_array", oracle)
        wb_by = {int(d): float(c) for d, c in zip(wb_daids, wb_csum) if np.isfinite(c)}

        annot_uuids = [str(a.annot_uuid) for a in database]
        uuid_to_daid = {au: i + 1 for i, au in enumerate(annot_uuids)}

        common = sorted(
            set(
                uuid_to_daid.get(str(r.annot_uuid), 0)
                for r in hs
                if np.isfinite(r.annot_csum) and r.annot_csum > 0
            )
            & set(wb_by.keys())
        )

        h_vals = []
        w_vals = []
        for d in common:
            for r in hs:
                if uuid_to_daid.get(str(r.annot_uuid)) == d:
                    h_vals.append(float(r.annot_csum))
                    w_vals.append(wb_by[d])
                    break

        if len(common) < 5:
            pytest.skip(f"Q{qidx}: only {len(common)} common daids (need >= 5)")

        h_arr = np.array(h_vals)
        w_arr = np.array(w_vals)
        pearson = float(np.corrcoef(h_arr, w_arr)[0, 1])
        spearman = _spearmanr(h_arr, w_arr)

        lines = [
            f"\nQ{qidx}: daid-aware annot csum — Pearson r = {pearson:.4f}  "
            f"Spearman ρ = {spearman:.4f}  (n = {len(common)})",
            f"{'daid':>5}  {'HS csum':>10}  {'WBIA csum':>10}  {'Δ':>10}  {'Δ%':>8}",
            "-" * 55,
        ]
        for d in common:
            hv = h_arr[common.index(d)]
            wv = w_arr[common.index(d)]
            delta = hv - wv
            pct = (delta / wv * 100) if abs(wv) > 0.0001 else 0.0
            lines.append(
                f"{d:5d}  {hv:10.4f}  {wv:10.4f}  {delta:+10.4f}  {pct:+7.1f}%"
            )
        print("\n".join(lines))

        assert (
            spearman >= 0.10
        ), f"Q{qidx}: annot csum Spearman ρ = {spearman:.4f} < 0.10"

    @pytest.mark.parametrize("qidx", range(_QUERIES))
    def test_daid_aware_vs_positional(self, database, results_for_query, qidx):
        hs = results_for_query[qidx]
        raw = _get_oracle_final_row(qidx)
        oracle = raw["oracle"]
        row = raw["row"]
        wb_daids = _load_oracle_arr(row, "daid_list_array", oracle).astype(int)
        wb_csum = _load_oracle_arr(row, "annot_score_list_array", oracle)

        annot_uuids = [str(a.annot_uuid) for a in database]
        uuid_to_daid = {au: i + 1 for i, au in enumerate(annot_uuids)}

        ml = min(len(hs), len(wb_csum))
        hs_pos = np.array(
            [float(r.annot_csum) for r in hs[:ml] if np.isfinite(r.annot_csum)]
        )
        wb_pos = np.array(
            [float(wb_csum[i]) for i in range(ml) if np.isfinite(wb_csum[i])]
        )
        m = min(len(hs_pos), len(wb_pos))
        pos_rho = _spearmanr(hs_pos[:m], wb_pos[:m]) if m >= 3 else np.nan

        wb_by = {int(d): float(c) for d, c in zip(wb_daids, wb_csum) if np.isfinite(c)}
        common = sorted(
            set(
                uuid_to_daid.get(str(r.annot_uuid), 0)
                for r in hs
                if np.isfinite(r.annot_csum) and r.annot_csum > 0
            )
            & set(wb_by.keys())
        )

        h_vals = []
        w_vals = []
        for d in common:
            for r in hs:
                if uuid_to_daid.get(str(r.annot_uuid)) == d:
                    h_vals.append(float(r.annot_csum))
                    w_vals.append(wb_by[d])
                    break

        daid_h = np.array(h_vals)
        daid_w = np.array(w_vals)
        daid_rho = _spearmanr(daid_h, daid_w) if len(common) >= 3 else np.nan

        print(f"\nQ{qidx} comparison method       Spearman ρ")
        print(f"─────────────────────────  ──────────")
        print(f"Positional (HS script)     {pos_rho:.4f}")
        print(f"Daid-aware (correct)       {daid_rho:.4f}")
        if np.isfinite(pos_rho) and np.isfinite(daid_rho):
            print(f"Lost to sort-order          {pos_rho - daid_rho:+.4f}")

        assert daid_rho >= 0.10, f"Q{qidx}: daid-aware ρ = {daid_rho:.4f} < 0.10"
