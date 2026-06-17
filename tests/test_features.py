"""Tests for wbia_core.features (pyhesaff SIFT)."""

import numpy as np
import pytest

from wbia_core.config import SiftConfig


def test_extract_features():
    from wbia_core.features import extract_features

    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    fs = extract_features(img, SiftConfig())
    assert fs.keypoints.shape[0] > 0
    assert fs.descriptors.shape[0] > 0
    assert fs.descriptors.shape[1] == 128


def test_extract_features_grayscale():
    from wbia_core.features import extract_features

    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    fs = extract_features(img, SiftConfig())
    assert fs.keypoints.shape[0] > 0
