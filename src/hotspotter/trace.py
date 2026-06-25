"""Parquet trace writer for hotspotter pipeline stages.

When ``HOTSPOTTER_TRACE_DIR`` is set, each pipeline stage writes a parquet
row plus ``.npy`` sidecars for large arrays.  The schema matches WBIA oracle
dumps so that ``compare_wbia_oracles.py`` can compare hotspotter traces
directly against WBIA oracle outputs.

Naming convention::

    {config_label}_{query_index}.parquet

Where ``config_label`` is the canonical config name (e.g. ``sv_on_true``)
and ``query_index`` is the query position within that config.

Usage::

    HOTSPOTTER_TRACE_DIR=artifacts/hotspotter-trace/run-001 \
    HOTSPOTTER_TRACE_RUN_ID=hs-dev-1 \
    HOTSPOTTER_TRACE_CONFIG_LABEL=sv_on_true \
        python scripts/run_fixture.py ...
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

_ARRAY_THRESHOLD = 64


def _is_trace_enabled() -> bool:
    return bool(os.environ.get("HOTSPOTTER_TRACE_DIR"))


def _trace_dir() -> Path:
    return Path(os.environ["HOTSPOTTER_TRACE_DIR"])


def _trace_run_id() -> str:
    return os.environ.get("HOTSPOTTER_TRACE_RUN_ID", str(uuid.uuid4()))


def _trace_config_label() -> str:
    return os.environ.get("HOTSPOTTER_TRACE_CONFIG_LABEL", "default")


def _write_manifest(
    base_dir: Path, config_label: str, query_index: int, config: dict | None = None
) -> None:
    """Write or append a trace manifest entry for the current run."""
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
            "config": config or {},
            "trace_run_id": _trace_run_id(),
            "timestamp_unix": time.time(),
        }
    )
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True))


class TraceContext:
    """Per-query trace context — one instance per ``identify()`` call."""

    def __init__(
        self, run_id: str, base_dir: Path, config_label: str, query_index: int
    ):
        self._run_id = run_id
        self._base = base_dir
        self._config_label = config_label
        self._query_index = query_index
        self._stage_counter = 0

    def _prefix(self) -> str:
        return f"{self._config_label}_{self._query_index:06d}"

    def _stage_rows(self, stage_name: str, rows: list[dict]) -> None:
        import pandas as pd

        timestamp = time.time()
        stage_dir = self._base / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            row.setdefault("trace_run_id", self._run_id)
            row.setdefault("stage", stage_name)
            row.setdefault("stage_counter", self._stage_counter)
            row.setdefault("row_index", 0)
            row.setdefault("timestamp_unix", timestamp)
            row.setdefault("config_label", self._config_label)
            row.setdefault("query_index", self._query_index)

        df = pd.DataFrame(rows)

        fname = f"{self._prefix()}.parquet"
        df.to_parquet(stage_dir / fname, index=False)
        self._stage_counter += 1

    def _array_sidecar_meta(
        self, stage_name: str, arr: np.ndarray, label: str, include_values: bool = False
    ) -> dict:
        arrays_dir = self._base / stage_name / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self._prefix()}_{label}.npy"
        path = arrays_dir / fname
        np.save(str(path), arr)
        meta = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "size": int(arr.size),
            "npy_path": str(
                Path("/artifacts/wbia-oracle")
                / self._base.name
                / stage_name
                / "arrays"
                / fname
            ),
        }
        if arr.size:
            meta.update(
                {
                    "min": _safe_scalar(np.min(arr)),
                    "max": _safe_scalar(np.max(arr)),
                    "mean": _safe_scalar(np.mean(arr.astype(np.float64))),
                }
            )
        if include_values:
            meta["values"] = arr.tolist()
        return meta

    def _write_array_sidecar(self, stage_name: str, arr: np.ndarray, label: str) -> str:
        return json.dumps(self._array_sidecar_meta(stage_name, arr, label))

    def _write_array_sequence(
        self, stage_name: str, arrays: Any, base_label: str
    ) -> list[dict]:
        if not arrays:
            return []
        sidecars = []
        for idx, arr in enumerate(arrays):
            if not isinstance(arr, np.ndarray):
                continue
            sidecars.append(
                self._array_sidecar_meta(
                    stage_name, arr, f"{base_label}_{idx}", include_values=True
                )
            )
        return sidecars

    def _maybe_array(self, stage_name: str, arr: np.ndarray, label: str) -> str:
        if arr.size <= _ARRAY_THRESHOLD:
            return json.dumps({"values": arr.tolist()})
        return self._write_array_sidecar(stage_name, arr, label)

    def trace_annotations(
        self,
        database: list,
    ) -> None:
        rows = []
        for i, ann in enumerate(database):
            rows.append(
                {
                    "aid": i,
                    "annot_uuid": str(ann.annot_uuid),
                    "bbox": str(tuple(ann.bbox)),
                    "theta": 0.0,
                    "species": "unknown",
                    "is_query": (i == 0),
                }
            )
        self._stage_rows("annotations", rows)

    def trace_chip(
        self,
        aid: int,
        chip: np.ndarray,
        bbox: tuple,
    ) -> None:
        h, w = chip.shape[:2]
        meta = self._array_sidecar_meta("chips", chip, f"chip_{aid:03d}")
        self._stage_rows(
            "chips",
            [
                {
                    "aid": aid,
                    "chip_fpath": meta.get("npy_path", ""),
                    "chip_size": str([h, w]),
                }
            ],
        )

    def trace_features(
        self,
        aid: int,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
    ) -> None:
        self._stage_rows(
            "features_keypoints",
            [
                {
                    "aid": aid,
                    "num_keypoints": int(keypoints.shape[0]),
                    "keypoints_array": self._maybe_array(
                        "features_keypoints", keypoints, f"keypoints_{aid:03d}"
                    ),
                }
            ],
        )
        self._stage_rows(
            "features_descriptors",
            [
                {
                    "aid": aid,
                    "num_descriptors": int(descriptors.shape[0]),
                    "descriptors_array": self._maybe_array(
                        "features_descriptors", descriptors, f"descriptors_{aid:03d}"
                    ),
                }
            ],
        )

    def trace_chips_and_features(
        self,
        database: list,
    ) -> None:
        chip_rows = []
        kp_rows = []
        desc_rows = []

        for i, ann in enumerate(database):
            if ann.image is not None:
                h, w = ann.image.shape[:2]
                meta = self._array_sidecar_meta("chips", ann.image, f"chip_{i:03d}")
                chip_rows.append(
                    {
                        "aid": i,
                        "chip_fpath": meta.get("npy_path", ""),
                        "chip_size": str([h, w]),
                    }
                )
            else:
                chip_rows.append(
                    {
                        "aid": i,
                        "chip_fpath": "",
                        "chip_size": str([0, 0]),
                    }
                )
            kp_rows.append(
                {
                    "aid": i,
                    "num_keypoints": int(ann.features.keypoints.shape[0]),
                    "keypoints_array": self._maybe_array(
                        "features_keypoints",
                        ann.features.keypoints,
                        f"keypoints_{i:03d}",
                    ),
                }
            )
            desc_rows.append(
                {
                    "aid": i,
                    "num_descriptors": int(ann.features.descriptors.shape[0]),
                    "descriptors_array": self._maybe_array(
                        "features_descriptors",
                        ann.features.descriptors,
                        f"descriptors_{i:03d}",
                    ),
                }
            )

        self._stage_rows("chips", chip_rows)
        self._stage_rows("features_keypoints", kp_rows)
        self._stage_rows("features_descriptors", desc_rows)

    def trace_neighbors(
        self,
        qaid: int,
        neighbor_idxs: np.ndarray,
        neighbor_dists: np.ndarray,
    ) -> None:
        self._stage_rows(
            "nearest_neighbors",
            [
                {
                    "qaid": qaid,
                    "neighbor_idxs_array": self._maybe_array(
                        "nearest_neighbors", neighbor_idxs, "neighbor_idxs"
                    ),
                    "neighbor_dists_array": self._maybe_array(
                        "nearest_neighbors", neighbor_dists, "neighbor_dists"
                    ),
                }
            ],
        )

    def trace_baseline_filter(
        self,
        invalid_mask: np.ndarray,
    ) -> None:
        self._stage_rows(
            "baseline_neighbor_filter",
            [
                {
                    "invalid_mask_array": self._maybe_array(
                        "baseline_neighbor_filter", invalid_mask, "invalid_mask"
                    ),
                }
            ],
        )

    def trace_neighbor_weights(
        self,
        weights: np.ndarray | list[float],
    ) -> None:
        if isinstance(weights, list):
            weights = np.array(weights, dtype=np.float64)
        self._stage_rows(
            "neighbor_weights",
            [
                {
                    "weights_array": self._maybe_array(
                        "neighbor_weights", weights, "weights"
                    ),
                    "num_weights": int(weights.size),
                    "nonzero_count": int((weights > 0).sum()),
                }
            ],
        )

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
        fm_list_json = json.dumps(self._write_array_sequence(stage_name, fm_list, "fm"))
        fsv_json = json.dumps(fsv_list or [])
        qnid_val = str(qnid) if qnid is not None else None
        _daids = np.asarray(daid_list if daid_list is not None else [])
        _dnids = np.asarray(dnid_list if dnid_list is not None else [])
        _ascores = np.asarray(
            annot_scores if annot_scores is not None else [], dtype=np.float64
        )
        _nscores = np.asarray(
            name_scores if name_scores is not None else [], dtype=np.float64
        )
        _slist = np.asarray(
            score_list if score_list is not None else [], dtype=np.float64
        )

        self._stage_rows(
            stage_name,
            [
                {
                    "qaid": qaid,
                    "qnid": qnid_val,
                    "daid_list_array": self._maybe_array(
                        stage_name, _daids, "daid_list"
                    ),
                    "dnid_list_array": self._maybe_array(
                        stage_name, _dnids, "dnid_list"
                    ),
                    "annot_score_list_array": self._maybe_array(
                        stage_name, _ascores, "annot_score_list"
                    ),
                    "name_score_list_array": self._maybe_array(
                        stage_name, _nscores, "name_score_list"
                    ),
                    "score_list_array": self._maybe_array(
                        stage_name, _slist, "score_list"
                    ),
                    "fm_list_json": fm_list_json,
                    "fsv_list_json": fsv_json,
                    "stage_name": stage_name,
                }
            ],
        )

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
        fm_list_json = json.dumps(
            self._write_array_sequence("final_scores", fm_list, "fm")
        )
        fsv_json = json.dumps(fsv_list or [])
        qnid_val = str(qnid) if qnid is not None else None
        _daids = np.asarray(daid_list if daid_list is not None else [])
        _dnids = np.asarray(dnid_list if dnid_list is not None else [])
        _ascores = np.asarray(
            annot_score_list if annot_score_list is not None else [], dtype=np.float64
        )
        _nscores = np.asarray(
            name_score_list if name_score_list is not None else [], dtype=np.float64
        )
        _slist = np.asarray(
            score_list if score_list is not None else [], dtype=np.float64
        )

        self._stage_rows(
            "final_scores",
            [
                {
                    "score_method": score_method,
                    "qaid": qaid,
                    "qnid": qnid_val,
                    "daid_list_array": self._maybe_array(
                        "final_scores", _daids, "daid_list"
                    ),
                    "dnid_list_array": self._maybe_array(
                        "final_scores", _dnids, "dnid_list"
                    ),
                    "annot_score_list_array": self._maybe_array(
                        "final_scores", _ascores, "annot_score_list"
                    ),
                    "name_score_list_array": self._maybe_array(
                        "final_scores", _nscores, "name_score_list"
                    ),
                    "score_list_array": self._maybe_array(
                        "final_scores", _slist, "score_list"
                    ),
                    "fm_list_json": fm_list_json,
                    "fsv_list_json": fsv_json,
                    "stage_name": "final_scores",
                }
            ],
        )


def _safe_scalar(value: Any) -> Any:
    """Convert numpy scalar to Python scalar for JSON serialization."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def get_trace_context(query_index: int = 0) -> TraceContext | None:
    """Return a ``TraceContext`` if ``HOTSPOTTER_TRACE_DIR`` is set.

    Environment variables read:
      ``HOTSPOTTER_TRACE_DIR`` — base output directory
      ``HOTSPOTTER_TRACE_RUN_ID`` — run identifier (optional)
      ``HOTSPOTTER_TRACE_CONFIG_LABEL`` — canonical config name
    """
    if not _is_trace_enabled():
        return None
    base_dir = _trace_dir()
    config_label = _trace_config_label()
    ctx = TraceContext(_trace_run_id(), base_dir, config_label, query_index)
    _write_manifest(base_dir, config_label, query_index)
    return ctx
