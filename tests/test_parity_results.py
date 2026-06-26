"""Compare final identification results against WBIA oracle.

Environment:
    WBIA_ORACLE_DIR — path to the WBIA oracle trace directory
    WBIA_BATCH_PATH — path to reference_batch.json
"""

from __future__ import annotations

import json
import os
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
    raw = os.environ.get("WBIA_ORACLE_DIR", "")
    if raw:
        p = Path(raw)
        nightly = p / "wildme-wbia-nightly-20260625-173226"
        if nightly.is_dir():
            return nightly
        if p.is_dir():
            return p
    return Path("/artifacts/wbia-oracle/wildme-wbia-nightly-20260625-173226")


def _batch_path() -> Path:
    raw = os.environ.get("WBIA_BATCH_PATH", "")
    if raw:
        return Path(raw)
    return Path("/app/pipeline/tests/reference_batch.json")


def _image_dir() -> Path:
    return _batch_path().parent / "assets" / "images"


def _build_database() -> list[AnnotatedImage]:
    batch_path = _batch_path()
    import cv2

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
    return np.load(
        oracle / stage / "arrays" / Path(meta["npy_path"]).name, allow_pickle=True
    )


# ---------------------------------------------------------------------------
# Live pipeline comparison tests
# ---------------------------------------------------------------------------


class TestParityResults:
    """Compare hotspotter live results against WBIA oracle.

    Runs the full pipeline and compares final rankings against the
    oracle trace, using both positional and daid-aware alignment.
    """

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
    def hotspotter_results(cls, database) -> list:
        config = IdentificationConfig(
            hotspotter=HotSpotterConfig(sv_on=True, fg_on=False, num_return=50)
        )
        return identify(0, database, config)

    @classmethod
    @pytest.fixture(scope="class")
    def oracle_ranking(cls, annot_uuids, name_uuids) -> list[tuple[str, str, float]]:
        oracle = _oracle_dir()
        df = pd.read_parquet(oracle / "final_scores" / "sv_on_true_000000.parquet")
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
        return ranking

    # -- Basic overlap tests --

    def test_top_5_annot_overlap(self, database, hotspotter_results, oracle_ranking):
        wbia_top = {aid for aid, _, _ in oracle_ranking[:5]}
        hs_top = {
            str(r.annot_uuid)
            for r in hotspotter_results[:10]
            if np.isfinite(r.score) and r.score > -1e300
        }
        assert len(wbia_top & hs_top) > 0, "No annot overlap"

    def test_top_3_name_overlap(self, database, hotspotter_results, oracle_ranking):
        wbia_top_names = set()
        for _, nid, _ in oracle_ranking:
            if nid:
                wbia_top_names.add(nid)
            if len(wbia_top_names) >= 3:
                break
        hs_top_names: set[str] = set()
        for r in hotspotter_results:
            if r.name_uuid is not None and np.isfinite(r.score) and r.score > -1e300:
                hs_top_names.add(str(r.name_uuid))
            if len(hs_top_names) >= 5:
                break
        assert len(wbia_top_names & hs_top_names) > 0, "No name overlap"

    def test_rank_reciprocal_overlap(self, database, hotspotter_results, oracle_ranking):
        wbia_rank = {aid: i + 1 for i, (aid, _, _) in enumerate(oracle_ranking)}
        rro = 0.0
        for r in hotspotter_results[:5]:
            if not np.isfinite(r.score) or r.score <= -1e300:
                continue
            if str(r.annot_uuid) in wbia_rank:
                rro += 1.0 / wbia_rank[str(r.annot_uuid)]
        assert rro > 0.0, f"Rank-reciprocal overlap = {rro:.4f}"

    def test_top_name_rank(self, database, hotspotter_results, oracle_ranking):
        hs_top_name = None
        for r in hotspotter_results[:5]:
            if r.name_uuid is not None and np.isfinite(r.score) and r.score > -1e300:
                hs_top_name = str(r.name_uuid)
                break
        if hs_top_name is None:
            pytest.skip("No valid named result in HS top-5")
        wbia_top_3_names = set()
        for _, nid, _ in oracle_ranking:
            if nid:
                wbia_top_3_names.add(nid)
            if len(wbia_top_3_names) >= 3:
                break
        assert hs_top_name in wbia_top_3_names, f"HS top name not in WBIA top-3"

    def test_result_count(self, hotspotter_results):
        valid = [
            r for r in hotspotter_results if np.isfinite(r.score) and r.score > -1e300
        ]
        assert len(valid) >= 1, "No valid results"

    def test_scores_monotonic(self, hotspotter_results):
        prev = float("inf")
        for r in hotspotter_results:
            if not np.isfinite(r.score) or r.score <= -1e300:
                continue
            if r.score > prev:
                pytest.fail(f"Score increased: {prev} → {r.score}")
            prev = r.score

    # -- Daid-aware correlation tests --

    def test_daid_aware_annot_csum_correlation(
        self, database, hotspotter_results, oracle_ranking
    ):
        """Per-annot csum aligned by daid: Spearman ρ >= 0.60."""
        oracle = _oracle_dir()
        df = pd.read_parquet(oracle / "final_scores" / "sv_on_true_000000.parquet")
        row = df.iloc[0]
        wb_daids = _load_oracle_arr(row, "daid_list_array", oracle).astype(int)
        wb_csum = _load_oracle_arr(row, "annot_score_list_array", oracle)
        wb_by = {int(d): float(c) for d, c in zip(wb_daids, wb_csum) if np.isfinite(c)}

        annot_uuids = [str(a.annot_uuid) for a in database]
        uuid_to_daid = {au: i + 1 for i, au in enumerate(annot_uuids)}

        common = sorted(
            set(
                uuid_to_daid.get(str(r.annot_uuid), 0)
                for r in hotspotter_results
                if np.isfinite(r.annot_csum) and r.annot_csum > 0
            )
            & set(wb_by.keys())
        )

        h_vals = []
        w_vals = []
        for d in common:
            for r in hotspotter_results:
                if uuid_to_daid.get(str(r.annot_uuid)) == d:
                    h_vals.append(float(r.annot_csum))
                    w_vals.append(wb_by[d])
                    break

        if len(common) < 5:
            pytest.skip(f"Only {len(common)} common daids (need >= 5)")

        h_arr = np.array(h_vals)
        w_arr = np.array(w_vals)
        pearson = float(np.corrcoef(h_arr, w_arr)[0, 1])
        spearman = _spearmanr(h_arr, w_arr)

        lines = [
            f"\nDaid-aware annot csum: Pearson r = {pearson:.4f}  "
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

        assert spearman >= 0.60, f"Annot csum Spearman ρ = {spearman:.4f} < 0.60"

    def test_daid_aware_vs_positional(self, database, hotspotter_results, oracle_ranking):
        """Show positional vs daid-aware Spearman ρ comparison."""
        oracle = _oracle_dir()
        df = pd.read_parquet(oracle / "final_scores" / "sv_on_true_000000.parquet")
        row = df.iloc[0]
        wb_daids = _load_oracle_arr(row, "daid_list_array", oracle).astype(int)
        wb_csum = _load_oracle_arr(row, "annot_score_list_array", oracle)

        annot_uuids = [str(a.annot_uuid) for a in database]
        uuid_to_daid = {au: i + 1 for i, au in enumerate(annot_uuids)}

        # Positional: compare HS results in order vs WBIA results in order
        ml = min(len(hotspotter_results), len(wb_csum))
        hs_pos = np.array(
            [
                float(r.annot_csum)
                for r in hotspotter_results[:ml]
                if np.isfinite(r.annot_csum)
            ]
        )
        wb_pos = np.array(
            [float(wb_csum[i]) for i in range(ml) if np.isfinite(wb_csum[i])]
        )
        m = min(len(hs_pos), len(wb_pos))
        pos_rho = _spearmanr(hs_pos[:m], wb_pos[:m]) if m >= 3 else np.nan

        # Daid-aware: compare same annotations
        wb_by = {int(d): float(c) for d, c in zip(wb_daids, wb_csum) if np.isfinite(c)}
        common = sorted(
            set(
                uuid_to_daid.get(str(r.annot_uuid), 0)
                for r in hotspotter_results
                if np.isfinite(r.annot_csum) and r.annot_csum > 0
            )
            & set(wb_by.keys())
        )

        h_vals = []
        w_vals = []
        for d in common:
            for r in hotspotter_results:
                if uuid_to_daid.get(str(r.annot_uuid)) == d:
                    h_vals.append(float(r.annot_csum))
                    w_vals.append(wb_by[d])
                    break

        daid_h = np.array(h_vals)
        daid_w = np.array(w_vals)
        daid_rho = _spearmanr(daid_h, daid_w) if len(common) >= 3 else np.nan

        print(f"\nComparison method          Spearman ρ")
        print(f"─────────────────────────  ──────────")
        print(f"Positional (HS script)     {pos_rho:.4f}")
        print(f"Daid-aware (correct)       {daid_rho:.4f}")
        if np.isfinite(pos_rho) and np.isfinite(daid_rho):
            print(f"Lost to sort-order          {pos_rho - daid_rho:+.4f}")

        assert daid_rho >= 0.60, f"Daid-aware ρ = {daid_rho:.4f} < 0.60"
