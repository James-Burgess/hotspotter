"""Smoke tests for the benchmark runner — no Docker containers started."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.benchmark.compare import compare_results
from tests.benchmark.runner import run_benchmark, _strip_images
from tests.benchmark.targets.base import QueryResult, TargetConfig, TargetRunner

# ---------------------------------------------------------------------------
# Mock target runner
# ---------------------------------------------------------------------------


class MockTargetRunner(TargetRunner):
    def __init__(self, config: TargetConfig, fail_start: bool = False):
        super().__init__(config)
        self.fail_start = fail_start
        self.started = False
        self.stopped = False
        self.queries_run: list[int] = []

    def start(self) -> dict:
        if self.fail_start:
            raise RuntimeError("Mock start failure")
        self.started = True
        return {
            "target": self.config.name,
            "image": self.config.image,
            "container_id": "mock-123",
            "started_at": "2026-01-01T00:00:00Z",
        }

    def run_query(self, query_index: int, request_body: dict) -> QueryResult:
        self.queries_run.append(query_index)
        import random

        scores = [
            {
                "aid": f"coco-annot-{i}",
                "score": random.uniform(0, 10),
                "num_matches": random.randint(1, 50),
            }
            for i in range(5)
        ]
        scores.sort(key=lambda x: x["score"], reverse=True)
        return QueryResult(
            query_index=query_index,
            annot_scores=scores,
            timing_ms=random.uniform(50, 200),
            raw_response={"status": "completed"},
        )

    def stop(self) -> None:
        self.stopped = True


class MockTargetRunnerErrorQuery(TargetRunner):
    def __init__(self, config: TargetConfig):
        super().__init__(config)

    def start(self) -> dict:
        return {
            "target": self.config.name,
            "image": self.config.image,
            "container_id": "mock-err",
            "started_at": "2026-01-01T00:00:00Z",
        }

    def run_query(self, query_index: int, request_body: dict) -> QueryResult:
        return QueryResult(query_index=query_index, error="Query failed")

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# _strip_images tests
# ---------------------------------------------------------------------------


class TestStripImages:
    def test_strips_query_image(self):
        body = {"query_image_b64": "abc123", "config": {"K": 4}}
        stripped = _strip_images(body)
        assert "abc123" not in str(stripped["query_image_b64"])
        assert stripped["config"]["K"] == 4

    def test_strips_database_images(self):
        body = {
            "database": [
                {"aid": "a", "image_b64": "img1", "bbox": [0, 0, 1, 1]},
                {"aid": "b", "image_b64": "img2", "bbox": [2, 2, 3, 3]},
            ],
        }
        stripped = _strip_images(body)
        for entry in stripped["database"]:
            assert "<base64" in entry["image_b64"]
            assert "bbox" in entry

    def test_preserves_other_fields(self):
        body = {
            "query_image_b64": "data",
            "query_bbox": [10, 20, 100, 200],
            "extra": "keep",
        }
        stripped = _strip_images(body)
        assert stripped["query_bbox"] == [10, 20, 100, 200]
        assert stripped["extra"] == "keep"


# ---------------------------------------------------------------------------
# run_benchmark tests
# ---------------------------------------------------------------------------


def _make_dummy_subset(n_annots=10, n_queries=3):
    """Create a minimal CocoSubset with synthetic data."""
    from tests.benchmark.coco.loader import CocoAnnotation, CocoSubset

    annots = []
    for i in range(n_annots):
        annots.append(
            CocoAnnotation(
                annot_id=1000 + i,
                image_id=2000 + i,
                bbox=(10, 20, 200, 300),
                species="zebra_plains",
                individual_ids=[f"indiv_{i}"],
                image=b"fake-image-bytes-" + str(i).encode(),
                width=640,
                height=480,
            )
        )
    return CocoSubset(
        annotations=annots,
        query_indices=list(range(n_queries)),
        config={
            "n_annots": n_annots,
            "n_queries": n_queries,
            "species": "zebra_plains",
            "seed": 42,
        },
    )


class TestRunBenchmark:
    def test_basic_run(self):
        subset = _make_dummy_subset(n_annots=10, n_queries=3)
        target = MockTargetRunner(
            TargetConfig(name="mock", image="mock:latest", port=5000)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_benchmark(
                subset, [target], tmpdir, {"pipeline_root": "vsmany"}
            )

        assert "targets" in result
        assert "mock" in result["targets"]
        assert target.started is True
        assert target.stopped is True
        assert len(target.queries_run) == 3

    def test_directories_created(self):
        subset = _make_dummy_subset(n_annots=10, n_queries=2)
        target = MockTargetRunner(
            TargetConfig(name="mock", image="mock:latest", port=5000)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_benchmark(subset, [target], tmpdir, {})

            results_path = Path(tmpdir)
            assert (results_path / "config.json").exists()
            assert (results_path / "target-mock").exists()
            assert (results_path / "target-mock/manifest.json").exists()
            assert (results_path / "target-mock/query_000").exists()
            assert (results_path / "target-mock/query_001").exists()
            assert (results_path / "target-mock/query_000/request.json").exists()
            assert (results_path / "target-mock/query_000/response.json").exists()

    def test_request_json_has_no_base64(self):
        subset = _make_dummy_subset(n_annots=5, n_queries=1)
        target = MockTargetRunner(
            TargetConfig(name="mock", image="mock:latest", port=5000)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_benchmark(subset, [target], tmpdir, {})
            req = json.loads(
                (Path(tmpdir) / "target-mock/query_000/request.json").read_text()
            )
            assert "<base64" in req["query_image_b64"]
            for entry in req["database"]:
                assert "<base64" in entry["image_b64"]

    def test_target_start_failure_skipped(self):
        subset = _make_dummy_subset(n_annots=5, n_queries=1)
        target = MockTargetRunner(
            TargetConfig(name="failer", image="fail:latest", port=5001),
            fail_start=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_benchmark(subset, [target], tmpdir, {})

        manifest = result["targets"]["failer"]
        assert "Failed to start" in manifest["errors"][0]

    def test_query_error_recorded(self):
        subset = _make_dummy_subset(n_annots=5, n_queries=1)
        target = MockTargetRunnerErrorQuery(
            TargetConfig(name="err", image="err:latest", port=5002)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_benchmark(subset, [target], tmpdir, {})

        manifest = result["targets"]["err"]
        assert len(manifest["errors"]) == 1
        assert "Query failed" in manifest["errors"][0]


# ---------------------------------------------------------------------------
# compare_results tests
# ---------------------------------------------------------------------------


def _make_result_dir(
    tmpdir: str,
    target_name: str,
    query_scores: dict[int, list[dict]],
) -> Path:
    """Helper to create a result directory with hand-crafted response data."""
    results_path = Path(tmpdir)
    target_dir = results_path / f"target-{target_name}"
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "target": target_name,
        "image": "img:latest",
        "container_id": "abc",
        "n_queries": len(query_scores),
        "total_timing_ms": 100,
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest))

    for qnum, scores in query_scores.items():
        qdir = target_dir / f"query_{qnum:03d}"
        qdir.mkdir(exist_ok=True)
        response = {
            "query_index": qnum,
            "error": None,
            "response": {
                "annot_scores": scores,
                "timing_ms": 50,
            },
            "raw_response": {"status": "completed"},
        }
        (qdir / "response.json").write_text(json.dumps(response))

    return results_path


class TestCompareResults:
    def test_identical_results(self):
        scores = [
            {"aid": "a1", "score": 10.0, "num_matches": 5},
            {"aid": "a2", "score": 8.0, "num_matches": 4},
            {"aid": "a3", "score": 6.0, "num_matches": 3},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_result_dir(tmpdir, "A", {0: scores})
            _make_result_dir(tmpdir, "B", {0: scores})
            (Path(tmpdir) / "config.json").write_text(
                json.dumps({"n_annots": 10, "n_queries": 1})
            )

            summary = compare_results(tmpdir)

        assert summary["agreement"]["top1_identical"] is True
        assert summary["agreement"]["all_rankings_match"] is True
        assert summary["targets"] == ["A", "B"]

    def test_different_top1(self):
        scores_a = [{"aid": "a1", "score": 10.0, "num_matches": 5}]
        scores_b = [{"aid": "b1", "score": 10.0, "num_matches": 5}]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_result_dir(tmpdir, "A", {0: scores_a})
            _make_result_dir(tmpdir, "B", {0: scores_b})
            (Path(tmpdir) / "config.json").write_text(
                json.dumps({"n_annots": 10, "n_queries": 1})
            )

            summary = compare_results(tmpdir)

        assert summary["agreement"]["top1_identical"] is False
        assert summary["errors"] == []

    def test_max_score_delta(self):
        scores_a = [{"aid": "a1", "score": 10.0, "num_matches": 5}]
        scores_b = [{"aid": "a1", "score": 7.0, "num_matches": 5}]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_result_dir(tmpdir, "A", {0: scores_a})
            _make_result_dir(tmpdir, "B", {0: scores_b})
            (Path(tmpdir) / "config.json").write_text(
                json.dumps({"n_annots": 10, "n_queries": 1})
            )

            summary = compare_results(tmpdir)

        assert summary["agreement"]["max_score_delta"] == pytest.approx(3.0)
        assert summary["agreement"]["top1_identical"] is True

    def test_errors_are_collected(self):
        scores_a = [{"aid": "a1", "score": 10.0, "num_matches": 5}]
        with tempfile.TemporaryDirectory() as tmpdir_a:
            db_path = _make_result_dir(tmpdir_a, "A", {0: scores_a})
            (db_path / "target-A/query_000/response.json").write_text(
                json.dumps(
                    {
                        "query_index": 0,
                        "error": "timeout",
                        "response": {"annot_scores": [], "timing_ms": 0},
                    }
                )
            )

            with tempfile.TemporaryDirectory() as tmpdir_b:
                results_path = _make_result_dir(tmpdir_b, "B", {0: scores_a})
                (results_path / "config.json").write_text(
                    json.dumps({"n_annots": 10, "n_queries": 1})
                )
                import shutil

                shutil.copytree(
                    db_path / "target-A",
                    results_path / "target-A",
                    dirs_exist_ok=True,
                )

                summary = compare_results(results_path)
                assert len(summary["errors"]) >= 1

    def test_single_target(self):
        scores = [{"aid": "a1", "score": 10.0, "num_matches": 5}]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_result_dir(tmpdir, "A", {0: scores})
            (Path(tmpdir) / "config.json").write_text(
                json.dumps({"n_annots": 10, "n_queries": 1})
            )

            summary = compare_results(tmpdir)

        assert summary["targets"] == ["A"]
        assert summary["agreement"]["top1_identical"] is True


class TestCliArgParsing:
    def test_run_benchmark_module_importable(self):
        from tests.benchmark import run_benchmark as rb

        assert rb.TARGET_MAP is not None
        assert "wbia-core" in rb.TARGET_MAP
        assert "wbia-latest" in rb.TARGET_MAP

    def test_build_targets(self):
        from tests.benchmark.run_benchmark import _build_targets

        targets = _build_targets(["wbia-core"], base_port=5000)
        assert len(targets) == 1
        assert targets[0].config.name == "wbia-core"
        assert targets[0].config.port == 5000

    def test_build_targets_multiple(self):
        from tests.benchmark.run_benchmark import _build_targets

        targets = _build_targets(["wbia-core", "wbia-latest"], base_port=5000)
        assert len(targets) == 2
        assert targets[0].config.port == 5000
        assert targets[1].config.port == 5001
        assert targets[1].config.image == "wildme/wbia:latest"

    def test_build_targets_unknown_skipped(self):
        from tests.benchmark.run_benchmark import _build_targets

        targets = _build_targets(["wbia-core", "unknown-target"], base_port=5000)
        assert len(targets) == 1

    def test_default_config_exported(self):
        from tests.benchmark import run_benchmark as rb

        assert rb.DEFAULT_CONFIG["pipeline_root"] == "vsmany"
        assert rb.DEFAULT_CONFIG["K"] == 4
        assert rb.DEFAULT_CONFIG["sv_on"] is False
