"""Parquet trace writer matching WBIA oracle format exactly.

Mirrors the conventions of ``patches/wbia_parquet_trace.py`` so that
``compare_wbia_oracles.py`` can diff hotspotter traces against WBIA oracles
with identical file structure, column schemas, and array naming.

File naming::

    {config_label}_{query_index:06d}.parquet

Array sidecar naming::

    {counter:06d}_{row_index}_{field}.npy

Metadata columns (always first 5, matching WBIA)::

    trace_run_id, stage, stage_counter, row_index, timestamp_unix
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

_ARRAY_INLINE_LIMIT = 64

# Module-level per-stage counters, matching WBIA's _COUNTERS dict.
# Persists across TraceContext instances so that successive queries
# in the same run get distinct array filenames.
_GLOBAL_COUNTERS: dict[str, int] = {}


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #


def _is_trace_enabled() -> bool:
    return bool(os.environ.get("HOTSPOTTER_TRACE_DIR"))


def _trace_dir() -> Path:
    return Path(os.environ["HOTSPOTTER_TRACE_DIR"])


def _trace_run_id() -> str:
    return os.environ.get("HOTSPOTTER_TRACE_RUN_ID", "hotspotter")


def _trace_config_label() -> str:
    return os.environ.get("HOTSPOTTER_TRACE_CONFIG_LABEL", "default")


# --------------------------------------------------------------------------- #
# TraceContext — one instance per identify() call
# --------------------------------------------------------------------------- #


class TraceContext:
    """Per-query trace context mirroring WBIA's ``wbia_parquet_trace.py``.

    All file naming, column ordering, and array conventions are identical
    to the WBIA trace hooks so that oracle comparison is structural.
    """

    def __init__(
        self, run_id: str, base_dir: Path, config_label: str, query_index: int
    ):
        self._run_id = run_id
        self._base = base_dir
        self._config_label = config_label
        self._query_index = query_index

    # ------------------------------------------------------------------ #
    # Core infrastructure (direct mirrors of WBIA helpers)
    # ------------------------------------------------------------------ #

    def _stage_dir(self, stage: str) -> Path:
        path = self._base / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _next_counter(self, stage: str) -> int:
        _GLOBAL_COUNTERS[stage] = _GLOBAL_COUNTERS.get(stage, 0) + 1
        return _GLOBAL_COUNTERS[stage]

    @staticmethod
    def _array_summary(arr: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size),
        }
        if arr.size:
            try:
                finite = (
                    arr[np.isfinite(arr)]
                    if np.issubdtype(arr.dtype, np.number)
                    else arr
                )
                if np.size(finite):
                    out["min"] = float(np.min(finite))
                    out["max"] = float(np.max(finite))
                    out["mean"] = float(np.mean(finite))
            except Exception:
                pass
        return out

    def _save_array(
        self, stage: str, counter: int, label: str, arr: np.ndarray
    ) -> dict[str, Any]:
        arrays_dir = self._stage_dir(stage) / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
        path = arrays_dir / f"{counter:06d}_{safe_label}.npy"
        np.save(str(path), arr)
        out = self._array_summary(arr)
        out["npy_path"] = str(path)
        if arr.size <= _ARRAY_INLINE_LIMIT:
            out["values"] = arr.tolist()
        return out

    def _json_default(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return self._array_summary(value)
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "__UUID__"):
            return str(value)
        return repr(value)

    def _json_dumps(self, value: Any) -> str:
        return json.dumps(value, default=self._json_default, sort_keys=True)

    def _scalar(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return self._json_dumps(value)

    def _to_rows(self, stage: str, counter: int, payload: Any) -> list[dict[str, Any]]:
        """Normalise *payload* into parquet rows with WBIA metadata at front."""
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            rows = payload
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            rows = [{"value_json": self._json_dumps(payload)}]

        timestamp = time.time()
        normalized: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            out: dict[str, Any] = {
                "trace_run_id": self._run_id,
                "stage": stage,
                "stage_counter": counter,
                "row_index": row_index,
                "timestamp_unix": timestamp,
            }
            for key, value in row.items():
                if isinstance(value, np.ndarray):
                    out[f"{key}_array"] = self._json_dumps(
                        self._save_array(stage, counter, f"{row_index}_{key}", value)
                    )
                elif isinstance(value, (list, tuple, dict)):
                    out[key] = self._json_dumps(value)
                else:
                    out[key] = self._scalar(value)
            normalized.append(out)
        return normalized

    def dump_stage(self, stage: str, payload: Any) -> None:
        """Write a parquet file for *stage* using WBIA naming conventions."""
        counter = self._next_counter(stage)
        rows = self._to_rows(stage, counter, payload)
        prefix = f"{self._config_label}_{self._query_index:06d}"
        path = self._stage_dir(stage) / f"{prefix}.parquet"
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)

    def _save_arrays_sequence(
        self,
        stage: str,
        counter: int,
        base_label: str,
        arrays: list[np.ndarray] | None,
    ) -> list[dict[str, Any]]:
        """Save a sequence of arrays and return their metadata dicts."""
        if not arrays:
            return []
        results: list[dict[str, Any]] = []
        for idx, arr in enumerate(arrays):
            if not isinstance(arr, np.ndarray):
                continue
            results.append(self._save_array(stage, counter, f"{base_label}_{idx}", arr))
        return results

    # ------------------------------------------------------------------ #
    # Stage-specific trace methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trace_order(database: list, query_annot_index: int) -> list[int]:
        """Query aid first, then rest — matches WBIA qaids+daids ordering."""
        return [query_annot_index] + [
            i for i in range(len(database)) if i != query_annot_index
        ]

    def trace_annotations(
        self,
        database: list,
        query_annot_index: int,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for i in self._trace_order(database, query_annot_index):
            ann = database[i]
            bbox = ann.bbox
            if bbox is not None and hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            elif bbox is not None:
                bbox = list(bbox)
            rows.append(
                {
                    "aid": i + 1,
                    "annot_uuid": str(ann.annot_uuid),
                    "bbox": bbox,
                    "theta": 0.0,
                    "species": "unknown",
                    "is_query": (i == query_annot_index),
                }
            )
        self.dump_stage("annotations", rows)

    def trace_chips_and_features(
        self,
        database: list,
        query_annot_index: int,
    ) -> None:
        order = self._trace_order(database, query_annot_index)
        chip_rows: list[dict[str, Any]] = []
        kp_rows: list[dict[str, Any]] = []
        desc_rows: list[dict[str, Any]] = []

        for i in order:
            ann = database[i]
            aid = i + 1
            if ann.image is not None:
                h, w = ann.image.shape[:2]
                chip_rows.append({"aid": aid, "chip_fpath": "", "chip_size": [w, h]})
            else:
                chip_rows.append({"aid": aid, "chip_fpath": "", "chip_size": [0, 0]})
            kp_rows.append(
                {
                    "aid": aid,
                    "num_keypoints": int(ann.features.keypoints.shape[0]),
                    "keypoints": np.asarray(ann.features.keypoints),
                }
            )
            desc_rows.append(
                {
                    "aid": aid,
                    "num_descriptors": int(ann.features.descriptors.shape[0]),
                    "descriptors": np.asarray(ann.features.descriptors),
                }
            )

        self.dump_stage("chips", chip_rows)
        self.dump_stage("features_keypoints", kp_rows)
        self.dump_stage("features_descriptors", desc_rows)

    def trace_neighbors(
        self,
        qaid: int,
        neighbor_idxs: np.ndarray,
        neighbor_dists: np.ndarray,
    ) -> None:
        self.dump_stage(
            "nearest_neighbors",
            [
                {
                    "qaid": qaid,
                    "neighbor_idxs": np.asarray(neighbor_idxs),
                    "neighbor_dists": np.asarray(neighbor_dists),
                }
            ],
        )

    def trace_baseline_filter(
        self,
        qaid: int,
        valid: np.ndarray,
    ) -> None:
        self.dump_stage(
            "baseline_neighbor_filter",
            [{"qaid": qaid, "valid": np.asarray(valid)}],
        )

    def trace_neighbor_weights(
        self,
        qaid: int,
        weights: np.ndarray | list[float],
    ) -> None:
        if isinstance(weights, list):
            weights = np.array(weights, dtype=np.float64)
        self.dump_stage(
            "neighbor_weights",
            [
                {
                    "qaid": qaid,
                    "filtkeys": ["baseline"],
                    "weight_lnbnn": np.asarray(weights),
                }
            ],
        )

    # ------------------------------------------------------------------ #
    # Chipmatch / final-score traces
    # ------------------------------------------------------------------ #

    def _chipmatch_payload(
        self,
        fm_stage: str,
        qaid: int,
        qnid: Any,
        daid_list: Any,
        dnid_list: Any,
        annot_scores: Any,
        name_scores: Any,
        score_list: Any,
        fm_list: Any,
        fsv_list: Any,
    ) -> dict[str, Any]:
        fm_counter = self._next_counter("__internal__")
        fm_sidecars = (
            self._save_arrays_sequence(fm_stage, fm_counter, "fm", fm_list)
            if fm_list
            else []
        )
        fsv_sidecars = (
            self._save_arrays_sequence(fm_stage, fm_counter, "fsv", fsv_list)
            if fsv_list
            else []
        )
        return {
            "qaid": qaid,
            "qnid": int(qnid) if qnid is not None else None,
            "daid_list": np.asarray(
                daid_list if daid_list is not None else [], dtype=np.int64
            ),
            "dnid_list": np.asarray(
                dnid_list if dnid_list is not None else [], dtype=np.int64
            ),
            "annot_score_list": (
                np.asarray(annot_scores)
                if annot_scores is not None
                else np.asarray(None)
            ),
            "name_score_list": (
                np.asarray(name_scores) if name_scores is not None else np.asarray(None)
            ),
            "score_list": (
                np.asarray(score_list) if score_list is not None else np.asarray(None)
            ),
            "fm_list_json": self._json_dumps(fm_sidecars),
            "fsv_list_json": self._json_dumps(fsv_sidecars),
            "stage_name": fm_stage,
        }

    def trace_chipmatches(
        self,
        stage_name: str,
        qaid: int,
        qnid: Any = None,
        daid_list: Any = None,
        dnid_list: Any = None,
        annot_scores: Any = None,
        name_scores: Any = None,
        score_list: Any = None,
        fm_list: Any = None,
        fsv_list: Any = None,
    ) -> None:
        fm_stage = stage_name.replace("chipmatches_", "")
        payload = self._chipmatch_payload(
            fm_stage,
            qaid,
            qnid,
            daid_list,
            dnid_list,
            annot_scores,
            name_scores,
            score_list,
            fm_list,
            fsv_list,
        )
        self.dump_stage(stage_name, [payload])

    def trace_final_scores(
        self,
        qaid: int,
        qnid: Any = None,
        score_method: str = "csum",
        daid_list: Any = None,
        dnid_list: Any = None,
        annot_score_list: Any = None,
        name_score_list: Any = None,
        score_list: Any = None,
        fm_list: Any = None,
        fsv_list: Any = None,
    ) -> None:
        fm_stage = "final"
        payload = self._chipmatch_payload(
            fm_stage,
            qaid,
            qnid,
            daid_list,
            dnid_list,
            annot_score_list,
            name_score_list,
            score_list,
            fm_list,
            fsv_list,
        )
        payload = {"score_method": score_method, **payload}
        self.dump_stage("final_scores", [payload])


# --------------------------------------------------------------------------- #
# Public factory
# --------------------------------------------------------------------------- #


def get_trace_context(query_index: int = 0) -> TraceContext | None:
    """Return a ``TraceContext`` if ``HOTSPOTTER_TRACE_DIR`` is set."""
    if not _is_trace_enabled():
        return None
    base_dir = _trace_dir()
    config_label = _trace_config_label()
    run_id = _trace_run_id()
    ctx = TraceContext(run_id, base_dir, config_label, query_index)

    manifest_path = base_dir / "trace_manifest.json"
    entries: list[dict] = []
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text())
        if not isinstance(entries, list):
            entries = []
    entries.append(
        {
            "config_label": config_label,
            "query_index": query_index,
            "trace_run_id": run_id,
            "timestamp_unix": time.time(),
        }
    )
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return ctx
