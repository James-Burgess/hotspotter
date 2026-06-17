"""Tests for WBIA result normalisation and query result creation."""

from __future__ import annotations

import pytest

from tests.benchmark.targets.base import QueryResult
from tests.benchmark.targets.wbia import normalise_wbia_result


class TestNormaliseWbiaResult:
    def _make_raw(
        self,
        score_list: list | None = None,
        num_matches_list: list | None = None,
        status: str = "completed",
    ) -> dict:
        return {
            "status": status,
            "json_result": {
                "cm_dict": {
                    "q-0000-0000": {
                        "dannot_uuid_list": [
                            {"__UUID__": "db-1"},
                            {"__UUID__": "db-2"},
                            {"__UUID__": "db-3"},
                        ],
                        "annot_score_list": score_list or [],
                        "num_matches_list": num_matches_list or [],
                    }
                }
            },
        }

    def test_basic_normalisation(self):
        raw = self._make_raw(
            score_list=[12.34, 8.56, 5.0],
            num_matches_list=[42, 31, 15],
        )
        result = normalise_wbia_result(raw, ["uuid-1", "uuid-2", "uuid-3"])
        assert len(result) == 3
        assert result[0] == {"aid": "uuid-1", "score": 12.34, "num_matches": 42}
        assert result[1] == {"aid": "uuid-2", "score": 8.56, "num_matches": 31}
        assert result[2] == {"aid": "uuid-3", "score": 5.0, "num_matches": 15}

    def test_empty_lists(self):
        raw = self._make_raw()
        result = normalise_wbia_result(raw, [])
        assert result == []

    def test_score_types_are_coerced(self):
        raw = self._make_raw(
            score_list=["12.34", 8.56, 5],
            num_matches_list=["42", 31, 15],
        )
        result = normalise_wbia_result(raw, ["a", "b", "c"])
        assert result[0]["score"] == 12.34
        assert isinstance(result[0]["score"], float)
        assert result[0]["num_matches"] == 42
        assert isinstance(result[0]["num_matches"], int)

    def test_mismatched_lengths(self):
        raw = self._make_raw(
            score_list=[1.0, 2.0],
            num_matches_list=[10],
        )
        result = normalise_wbia_result(raw, ["a", "b"])
        assert len(result) == 2

    def test_raises_on_non_completed(self):
        raw = {"status": "error", "message": "something broke"}
        with pytest.raises(AssertionError):
            normalise_wbia_result(raw, [])

    def test_raises_on_missing_cm_dict(self):
        raw = {"status": "completed", "json_result": {}}
        result = normalise_wbia_result(raw, ["a", "b"])
        assert result == []


class TestQueryResult:
    def test_default_creation(self):
        r = QueryResult(query_index=0)
        assert r.query_index == 0
        assert r.annot_scores == []
        assert r.timing_ms == 0.0
        assert r.raw_response is None
        assert r.error is None

    def test_full_creation(self):
        r = QueryResult(
            query_index=1,
            annot_scores=[{"aid": "a", "score": 1.0, "num_matches": 5}],
            timing_ms=123.4,
            raw_response={"status": "completed", "response": {}},
        )
        assert r.query_index == 1
        assert len(r.annot_scores) == 1
        assert r.timing_ms == 123.4
        assert r.raw_response is not None
        assert r.raw_response["status"] == "completed"

    def test_error_creation(self):
        r = QueryResult(query_index=2, error="Container crashed")
        assert r.error == "Container crashed"
        assert r.annot_scores == []
