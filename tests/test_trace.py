"""Unit tests for hotspotter.trace — TraceContext, serialization, env-var gating."""

import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hotspotter.data import AnnotatedImage, FeatureSet, Match
from hotspotter.trace import (
    _is_trace_enabled,
    _trace_config_label,
    _trace_dir,
    _trace_run_id,
    _zstd_compress,
    _zstd_decompress,
    get_trace_context,
    TraceContext,
)


def _dummy_annot(index: int, name_uuid: uuid.UUID | None = None) -> AnnotatedImage:
    return AnnotatedImage(
        annot_uuid=uuid.uuid4(),
        name_uuid=name_uuid or uuid.uuid4(),
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        features=FeatureSet(
            keypoints=np.zeros((3, 6), dtype=np.float32),
            descriptors=np.zeros((3, 128), dtype=np.uint8),
        ),
        bbox=(0, 0, 64, 64),
    )


class TestZstdRoundtrip:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.bin"
            dst = Path(td) / "dst.zst"
            data = b"hello world " * 100
            src.write_bytes(data)
            _zstd_compress(src, dst)
            assert dst.exists()
            assert dst.stat().st_size < len(data)
            decoded = _zstd_decompress(dst)
            assert decoded == data

    def test_empty_data(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.bin"
            dst = Path(td) / "dst.zst"
            src.write_bytes(b"")
            _zstd_compress(src, dst)
            assert dst.exists()
            decoded = _zstd_decompress(dst)
            assert decoded == b""

    def test_binary_not_text(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.bin"
            dst = Path(td) / "dst.zst"
            data = bytes(range(256)) * 10
            src.write_bytes(data)
            _zstd_compress(src, dst)
            decoded = _zstd_decompress(dst)
            assert decoded == data


class TestArraySummary:
    def test_finite_array(self):
        ctx = TraceContext("test", Path("/tmp"), "label", 0)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        summary = ctx._array_summary(arr)
        assert summary["shape"] == [3]
        assert summary["dtype"] == "float64"
        assert summary["size"] == 3
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0
        assert summary["mean"] == 2.0

    def test_empty_array(self):
        ctx = TraceContext("test", Path("/tmp"), "label", 0)
        arr = np.array([], dtype=np.float32)
        summary = ctx._array_summary(arr)
        assert summary["shape"] == [0]
        assert summary["size"] == 0
        assert "min" not in summary

    def test_with_inf_nan(self):
        ctx = TraceContext("test", Path("/tmp"), "label", 0)
        arr = np.array([1.0, np.inf, np.nan, 3.0], dtype=np.float64)
        summary = ctx._array_summary(arr)
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0

    def test_integer_array(self):
        ctx = TraceContext("test", Path("/tmp"), "label", 0)
        arr = np.array([10, 20, 30], dtype=np.int32)
        summary = ctx._array_summary(arr)
        assert summary["dtype"] == "int32"
        assert summary["min"] == 10
        assert summary["max"] == 30


class TestSaveArray:
    def test_creates_npz_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 0)
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            info = ctx._save_array("test_stage", 1, "my_array", arr)
            assert "npy_path" in info
            path = Path(info["npy_path"])
            assert path.suffix == ".npz"
            assert path.exists()
            loaded = np.load(path)
            np.testing.assert_array_equal(loaded[loaded.files[0]], arr)

    def test_inlines_small_array(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 0)
            arr = np.array([1.0, 2.0], dtype=np.float64)
            info = ctx._save_array("test_stage", 1, "small_array", arr)
            assert "values" in info
            assert info["values"] == [1.0, 2.0]

    def test_does_not_inline_large_array(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 0)
            arr = np.arange(100, dtype=np.float64)
            info = ctx._save_array("test_stage", 1, "large_array", arr)
            assert "values" not in info


class TestToRows:
    def test_dict_payload(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 0)
            rows = ctx._to_rows("mystage", 1, {"key": "value", "num": 42})
            assert len(rows) == 1
            assert rows[0]["trace_run_id"] == "test"
            assert rows[0]["stage"] == "mystage"
            assert rows[0]["stage_counter"] == 1
            assert rows[0]["row_index"] == 0
            assert rows[0]["key"] == "value"
            assert rows[0]["num"] == 42

    def test_list_of_dicts_payload(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 0)
            rows = ctx._to_rows("mystage", 2, [{"a": 1}, {"a": 2}])
            assert len(rows) == 2
            assert rows[0]["row_index"] == 0
            assert rows[1]["row_index"] == 1

    def test_ndarray_value_saved_as_array(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 0)
            rows = ctx._to_rows(
                "mystage", 1, {"data": np.array([1.0, 2.0], dtype=np.float64)}
            )
            assert "data_array" in rows[0]
            assert "data" not in rows[0]

    def test_list_value_json_encoded(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 0)
            rows = ctx._to_rows("mystage", 1, {"items": [1, 2, 3]})
            import json

            parsed = json.loads(rows[0]["items"])
            assert parsed == [1, 2, 3]


class TestDumpStage:
    def test_writes_parquet_file(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "config_x", 1)
            ctx.dump_stage("my_stage", {"value": 42})
            pq = Path(td) / "my_stage" / "config_x_000001.parquet"
            assert pq.exists()
            df = pd.read_parquet(pq)
            assert df.iloc[0]["value"] == 42
            assert df.iloc[0]["stage"] == "my_stage"

    def test_correct_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "config_x", 5)
            ctx.dump_stage("my_stage", {"x": 1})
            pq = Path(td) / "my_stage" / "config_x_000005.parquet"
            assert pq.exists()


class TestTraceOrder:
    def test_query_first(self):
        database = [_dummy_annot(i) for i in range(5)]
        order = TraceContext._trace_order(database, 2)
        assert order[0] == 2
        assert sorted(order[1:]) == [0, 1, 3, 4]

    def test_query_is_only_element(self):
        database = [_dummy_annot(0)]
        order = TraceContext._trace_order(database, 0)
        assert order == [0]

    def test_query_is_last(self):
        database = [_dummy_annot(i) for i in range(3)]
        order = TraceContext._trace_order(database, 2)
        assert order == [2, 0, 1]


class TestTraceNeighbors:
    def test_column_structure(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 5)
            ctx.trace_neighbors(
                qaid=1,
                neighbor_idxs=np.array([[0, 1]], dtype=np.int32),
                neighbor_dists=np.array([[0.1, 0.2]], dtype=np.float32),
            )
            pq = Path(td) / "nearest_neighbors" / "label_000005.parquet"
            assert pq.exists()
            df = pd.read_parquet(pq)
            assert df.iloc[0]["qaid"] == 1


class TestTraceAnnotations:
    def test_trace_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            db = [_dummy_annot(i) for i in range(3)]
            ctx = TraceContext("test", Path(td), "label", 0)
            ctx.trace_annotations(db, 0)
            pq = Path(td) / "annotations" / "label_000000.parquet"
            assert pq.exists()
            df = pd.read_parquet(pq)
            assert len(df) == 3
            assert df.iloc[0]["is_query"] == True
            assert df.iloc[1]["is_query"] == False


class TestEnvVarGating:
    def test_trace_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("HOTSPOTTER_TRACE_DIR", raising=False)
        assert _is_trace_enabled() is False

    def test_trace_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("HOTSPOTTER_TRACE_DIR", "/tmp/test_trace")
        assert _is_trace_enabled() is True

    def test_get_trace_context_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.delenv("HOTSPOTTER_TRACE_DIR", raising=False)
        assert get_trace_context(0) is None

    def test_get_trace_context_returns_context_when_enabled(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            monkeypatch.setenv("HOTSPOTTER_TRACE_DIR", td)
            ctx = get_trace_context(0)
            assert ctx is not None
            assert isinstance(ctx, TraceContext)

    def test_trace_run_id_default(self, monkeypatch):
        monkeypatch.delenv("HOTSPOTTER_TRACE_RUN_ID", raising=False)
        assert _trace_run_id() == "hotspotter"

    def test_trace_run_id_custom(self, monkeypatch):
        monkeypatch.setenv("HOTSPOTTER_TRACE_RUN_ID", "myrun")
        assert _trace_run_id() == "myrun"

    def test_trace_config_label_default(self, monkeypatch):
        monkeypatch.delenv("HOTSPOTTER_TRACE_CONFIG_LABEL", raising=False)
        assert _trace_config_label() == "default"

    def test_trace_config_label_custom(self, monkeypatch):
        monkeypatch.setenv("HOTSPOTTER_TRACE_CONFIG_LABEL", "custom")
        assert _trace_config_label() == "custom"

    def test_trace_dir_returns_env_path(self, monkeypatch):
        monkeypatch.setenv("HOTSPOTTER_TRACE_DIR", "/some/path")
        assert _trace_dir() == Path("/some/path")


class TestGlobalCounters:
    def test_counters_increment_across_instances(self):
        import hotspotter.trace as tr

        tr._GLOBAL_COUNTERS.clear()
        with tempfile.TemporaryDirectory() as td:
            ctx1 = TraceContext("test", Path(td), "label", 0)
            ctx2 = TraceContext("test", Path(td), "label", 0)
            c1 = ctx1._next_counter("stage_a")
            c2 = ctx2._next_counter("stage_a")
            assert c1 == 1
            assert c2 == 2

    def test_counters_per_stage_independent(self):
        import hotspotter.trace as tr

        tr._GLOBAL_COUNTERS.clear()
        with tempfile.TemporaryDirectory() as td:
            ctx = TraceContext("test", Path(td), "label", 0)
            assert ctx._next_counter("stage_a") == 1
            assert ctx._next_counter("stage_b") == 1


class TestManifestWriting:
    def test_get_trace_context_writes_manifest(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            monkeypatch.setenv("HOTSPOTTER_TRACE_DIR", td)
            monkeypatch.setenv("HOTSPOTTER_TRACE_CONFIG_LABEL", "myconfig")
            get_trace_context(0)
            manifest = Path(td) / "trace_manifest.json"
            assert manifest.exists()
            import json

            entries = json.loads(manifest.read_text())
            assert len(entries) >= 1
            assert entries[0]["config_label"] == "myconfig"
            assert entries[0]["query_index"] == 0
