import base64
import hashlib
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, request

from typing import Any, Literal, cast

from wbia_core.config import HotSpotterConfig, IdentificationConfig, SiftConfig
from wbia_core.data import AnnotatedImage, FeatureSet
from wbia_core.features import extract_features
from wbia_core.pipeline import identify

app = Flask(__name__)

_feature_cache: dict[str, FeatureSet] = {}


def _cache_key(aid: str, image_b64: str, bbox: tuple) -> str:
    img_hash = hashlib.sha256(image_b64.encode()).hexdigest()[:16]
    bbox_hash = hashlib.sha256(str(bbox).encode()).hexdigest()[:8]
    return f"{aid}:{img_hash}:{bbox_hash}"


@app.route("/api/health/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "0.1.0", "service": "wbia-core"})


@app.route("/api/v1/cache/clear/", methods=["POST"])
def cache_clear():
    n = len(_feature_cache)
    _feature_cache.clear()
    return jsonify({"status": "ok", "cleared": n})


def _compute_affine_matrix(
    bbox: tuple, new_size: tuple, theta: float = 0.0
) -> np.ndarray:
    """Build the image→chip affine transform matching WBIA's
    ``get_image_to_chip_transform``.

    Returns the 2×3 sub-matrix suitable for ``cv2.warpAffine``.
    """
    x, y, w, h = [float(v) for v in bbox]
    cw, ch = [float(v) for v in new_size]
    tx1, ty1 = -(x + w / 2.0), -(y + h / 2.0)
    sx, sy = cw / w, ch / h
    tx2, ty2 = cw / 2.0, ch / 2.0
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    T2 = np.array([[1, 0, tx2], [0, 1, ty2], [0, 0, 1]], dtype=np.float64)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    T1 = np.array([[1, 0, tx1], [0, 1, ty1], [0, 0, 1]], dtype=np.float64)
    C = T2 @ R @ S @ T1
    return C[:2]


def _extract_chip(
    img: np.ndarray, bbox: tuple, dim_size: int = 700, resize_dim: str = "maxwh"
) -> np.ndarray:
    """Crop *img* to *bbox* and resize using ``cv2.warpAffine``.

    Matches WBIA's ``extract_chip_from_img`` exactly — same affine
    transform, same Lanczos interpolation, same black border padding.
    """
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    if w <= 0 or h <= 0:
        return img
    if resize_dim == "width":
        scale = dim_size / w
        new_w, new_h = dim_size, max(1, int(round(h * scale)))
    elif resize_dim == "height":
        scale = dim_size / h
        new_h, new_w = dim_size, max(1, int(round(w * scale)))
    else:  # maxwh / area / diag — all use max dimension
        scale = dim_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
    M = _compute_affine_matrix((x, y, w, h), (new_w, new_h))
    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _decode_b64(b64_str: str) -> np.ndarray:
    buf = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img


@app.route("/api/v1/identify/", methods=["POST"])
def identify_view():
    try:
        body = request.get_json(force=True)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Invalid JSON: {exc}"}), 400

    try:
        result = _run_identify(body)
        return jsonify({"status": "completed", "response": result})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


def _run_identify(body: dict) -> dict:
    t0 = time.perf_counter()

    query_image_b64 = body["query_image_b64"]
    query_bbox = tuple(body["query_bbox"])
    database_entries = body["database"]
    cfg = body.get("config", {})

    query_img = _decode_b64(query_image_b64)
    query_chip = _extract_chip(query_img, query_bbox)
    query_features = extract_features(query_chip, SiftConfig())

    query_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, "query")

    aid_to_uuid: dict[str, uuid.UUID] = {}
    db_annots: list[AnnotatedImage] = [
        AnnotatedImage(
            annot_uuid=query_uuid,
            name_uuid=None,
            image=query_chip,
            features=query_features,
            bbox=query_bbox,
        )
    ]

    for entry in database_entries:
        aid = entry["aid"]
        image_b64 = entry["image_b64"]
        entry_bbox = tuple(entry["bbox"])

        ck = _cache_key(aid, image_b64, entry_bbox)
        if ck in _feature_cache:
            entry_features = _feature_cache[ck]
            entry_img = None
        else:
            entry_img = _decode_b64(image_b64)
            entry_chip = _extract_chip(entry_img, entry_bbox)
            entry_features = extract_features(entry_chip, SiftConfig())
            _feature_cache[ck] = entry_features

        annot_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, aid)
        aid_to_uuid[aid] = annot_uuid

        name_uuid = None
        raw = entry.get("name_uuid")
        if raw is not None:
            name_uuid = uuid.UUID(raw)
        else:
            name_uuid = annot_uuid

        db_annots.append(
            AnnotatedImage(
                annot_uuid=annot_uuid,
                name_uuid=name_uuid,
                image=entry_chip if entry_img is not None else None,
                features=entry_features,
                bbox=entry_bbox,
            )
        )

    _score_map: dict[str, Any] = {
        "nsum": "nsum_wbia",
        "csum": "csum_wbia",
        "sumamech": "sumamech",
    }
    score_method_raw: str = str(cfg.get("score_method", "nsum"))
    score_method = cast(
        Literal["csum", "nsum", "csum_wbia", "nsum_wbia", "sumamech"],
        _score_map.get(score_method_raw, "csum"),
    )

    hs_config = HotSpotterConfig(
        knn=cfg.get("K", 4),
        kpad=cfg.get("Kpad", 0),
        kpad_policy=cfg.get("kpad_policy", "fixed"),
        score_method=score_method,
        normalizer_rule=cast(
            Literal["last", "name"], cfg.get("normalizer_rule", "last")
        ),
        can_match_samename=cfg.get("can_match_samename", True),
        sqrd_dist_on=cfg.get("sqrd_dist_on", False),
        normonly_on=cfg.get("normonly_on", False),
        rotation_invariance=cfg.get("rotation_invariance", False),
        minscale_thresh=cfg.get("minscale_thresh"),
        maxscale_thresh=cfg.get("maxscale_thresh"),
        fgw_thresh=cfg.get("fgw_thresh"),
        sv_n_name_shortlist=cfg.get("sv_n_name_shortlist", 40),
        sv_n_annot_per_name=cfg.get("sv_n_annot_per_name", 3),
        sv_xy_thresh=cfg.get("sv_xy_thresh", 0.01),
        sv_scale_thresh=cfg.get("sv_scale_thresh", 2.0),
        sv_ori_thresh=cfg.get("sv_ori_thresh"),
        sv_use_chip_extent=cfg.get("sv_use_chip_extent", True),
        sv_weight_inliers=cfg.get("sv_weight_inliers", True),
        fg_on=cfg.get("fg_on", False),
        bar_l2_on=cfg.get("bar_l2_on", False),
        const_on=cfg.get("const_on", False),
        sv_on=cfg.get("sv_on", False),
        num_return=len(database_entries),
        flann_algorithm=cfg.get("flann_algorithm", "kdtree"),
        flann_random_seed=cfg.get("flann_random_seed", 42),
    )
    id_config = IdentificationConfig(hotspotter=hs_config)

    scored = identify(query_annot_index=0, database=db_annots, config=id_config)

    uuid_to_aid = {v: k for k, v in aid_to_uuid.items()}
    annot_scores = sorted(
        (
            {
                "aid": uuid_to_aid[match.annot_uuid],
                "score": round(float(match.score), 6),
                "num_matches": match.num_matches,
            }
            for match in scored
        ),
        key=lambda r: r["score"],
        reverse=True,
    )

    timing_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {"annot_scores": annot_scores, "timing_ms": timing_ms}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
