"""Feature extraction via pyhesaff Hessian-affine SIFT."""

from __future__ import annotations

import pyhesaff as _h
import numpy as np

from wbia_core.config import SiftConfig
from wbia_core.data import FeatureSet


def _to_hesaff_kwargs(config: SiftConfig, default_params: dict | None = None) -> dict:
    """Merge SiftConfig overrides into pyhesaff default parameters."""
    kwargs = dict(default_params) if default_params else {}
    if config.scale is not None and len(config.scale) > 0:
        kwargs["numberOfScales"] = len(config.scale)
    kwargs["ori_maxima_thresh"] = config.ori_hist_threshold
    return kwargs


def extract_features(
    image: np.ndarray, config: SiftConfig = SiftConfig()
) -> FeatureSet:
    """Extract Hessian-affine SIFT features from *image*.

    Args:
        image: [H, W] or [H, W, 3] uint8.
        config: SIFT extraction parameters.

    Returns:
        FeatureSet with *N* keypoints and descriptors.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    hesaff_kwargs = _to_hesaff_kwargs(config, _h.get_hesaff_default_params())
    keypoints, descriptors = _h.detect_feats_in_image(image, **hesaff_kwargs)
    return FeatureSet(keypoints=keypoints, descriptors=descriptors)
