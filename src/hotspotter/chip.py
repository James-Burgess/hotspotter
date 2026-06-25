"""Chip extraction from full images.

Matches WBIA's ``extract_chip_from_img`` exactly — same affine
transform, same Lanczos interpolation, same black border padding.
"""

from __future__ import annotations

import cv2
import numpy as np


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


def extract_chip(
    img: np.ndarray, bbox: tuple, dim_size: int = 700, resize_dim: str = "maxwh"
) -> np.ndarray:
    """Crop *img* to *bbox* and resize using ``cv2.warpAffine``.

    Matches WBIA's ``extract_chip_from_img`` exactly — same affine
    transform, same Lanczos interpolation, same black border padding.

    Args:
        img: Full image [H, W] or [H, W, 3] uint8.
        bbox: (x, y, w, h) in image coordinates.
        dim_size: Output dimension in pixels (default 700).
        resize_dim: ``"width"``, ``"height"``, or ``"maxwh"`` (default).

    Returns:
        Resized chip [new_h, new_w] or [new_h, new_w, 3] uint8.
    """
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if resize_dim == "width":
        scale = dim_size / w
        new_w, new_h = dim_size, max(1, int(round(h * scale)))
    elif resize_dim == "height":
        scale = dim_size / h
        new_h, new_w = dim_size, max(1, int(round(w * scale)))
    else:
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
