"""Unit tests for hotspotter.chip — extract_chip and _compute_affine_matrix."""

import numpy as np
import pytest

from hotspotter.chip import _compute_affine_matrix, extract_chip


class TestComputeAffineMatrix:
    def test_top_left_corner_of_bbox_maps_to_origin(self):
        bbox = (10, 20, 100, 100)
        new_size = (100, 100)
        M = _compute_affine_matrix(bbox, new_size)
        assert M.shape == (2, 3)
        src = np.array([10.0, 20.0, 1.0])
        dst = M @ src
        np.testing.assert_allclose(dst, [0.0, 0.0], atol=1e-10)

    def test_bbox_center_maps_to_chip_center(self):
        bbox = (50, 30, 80, 60)
        new_size = (160, 120)
        M = _compute_affine_matrix(bbox, new_size)
        center = np.array([90.0, 60.0, 1.0])
        dst = M @ center
        np.testing.assert_allclose(dst, [80.0, 60.0], atol=1e-10)

    def test_bottom_right_corner_of_bbox_maps_to_chip_bottom_right(self):
        bbox = (0, 0, 200, 100)
        new_size = (200, 100)
        M = _compute_affine_matrix(bbox, new_size)
        corner = np.array([200.0, 100.0, 1.0])
        dst = M @ corner
        np.testing.assert_allclose(dst, [200.0, 100.0], atol=1e-10)

    def test_rectangular_bbox_to_square_chip_top_left(self):
        bbox = (0, 0, 200, 100)
        new_size = (100, 100)
        M = _compute_affine_matrix(bbox, new_size)
        src = np.array([0.0, 0.0, 1.0])
        dst = M @ src
        np.testing.assert_allclose(dst, [0.0, 0.0], atol=1e-10)

    def test_non_zero_theta_rotation(self):
        bbox = (0, 0, 100, 100)
        new_size = (100, 100)
        theta = np.pi / 4
        M = _compute_affine_matrix(bbox, new_size, theta=theta)
        center = np.array([50.0, 50.0, 1.0])
        dst = M @ center
        np.testing.assert_allclose(dst, [50.0, 50.0], atol=1e-10)
        assert not np.allclose(M, np.eye(2, 3), atol=1e-10)

    def test_theta_zero_preserves_cardinal_directions(self):
        bbox = (50, 50, 64, 64)
        new_size = (64, 64)
        M = _compute_affine_matrix(bbox, new_size, theta=0.0)
        src_tl = np.array([50.0, 50.0, 1.0])
        dst_tl = M @ src_tl
        np.testing.assert_allclose(dst_tl, [0.0, 0.0], atol=1e-10)
        src_br = np.array([114.0, 114.0, 1.0])
        dst_br = M @ src_br
        np.testing.assert_allclose(dst_br, [64.0, 64.0], atol=1e-10)


class TestExtractChip:
    def test_basic_crop_and_resize(self):
        img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        bbox = (50, 40, 100, 80)
        chip = extract_chip(img, bbox, dim_size=400)
        assert chip.shape[0] > 0
        assert chip.shape[1] > 0
        assert chip.dtype == np.uint8

    def test_output_dtype_matches_input(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        chip = extract_chip(img, (10, 10, 40, 40), dim_size=200)
        assert chip.dtype == np.uint8

    def test_grayscale_input(self):
        img = np.random.randint(0, 256, (150, 200), dtype=np.uint8)
        bbox = (30, 20, 60, 50)
        chip = extract_chip(img, bbox, dim_size=200)
        assert len(chip.shape) == 2
        assert chip.shape[0] > 0

    def test_resize_dim_maxwh(self):
        img = np.zeros((200, 300), dtype=np.uint8)
        bbox = (0, 0, 200, 100)
        chip = extract_chip(img, bbox, dim_size=200, resize_dim="maxwh")
        assert max(chip.shape[0], chip.shape[1]) == 200

    def test_resize_dim_width(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        bbox = (10, 10, 100, 50)
        chip = extract_chip(img, bbox, dim_size=300, resize_dim="width")
        assert chip.shape[1] == 300

    def test_resize_dim_height(self):
        img = np.zeros((300, 400), dtype=np.uint8)
        bbox = (10, 10, 100, 50)
        chip = extract_chip(img, bbox, dim_size=200, resize_dim="height")
        assert chip.shape[0] == 200

    def test_zero_width_bbox_returns_fallback(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        chip = extract_chip(img, (10, 10, 0, 50))
        assert chip.shape == (64, 64, 3)

    def test_zero_height_bbox_returns_fallback(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        chip = extract_chip(img, (10, 10, 50, 0))
        assert chip.shape == (64, 64, 3)

    def test_negative_width_returns_fallback(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        chip = extract_chip(img, (0, 0, -5, 30))
        assert chip.shape == (64, 64, 3)

    def test_chip_aspect_ratio_matches_bbox(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        bbox = (0, 0, 200, 100)
        chip = extract_chip(img, bbox, dim_size=700)
        bbox_aspect = bbox[2] / bbox[3]
        chip_aspect = chip.shape[1] / chip.shape[0]
        assert abs(bbox_aspect - chip_aspect) < 0.01

    def test_chip_is_not_all_black_for_nonzero_image(self):
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = (30, 30, 100, 80)
        chip = extract_chip(img, bbox, dim_size=300)
        assert chip.mean() > 0

    def test_dim_size_controls_output(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        small = extract_chip(img, bbox, dim_size=100, resize_dim="maxwh")
        large = extract_chip(img, bbox, dim_size=500, resize_dim="maxwh")
        assert large.shape[0] > small.shape[0]
        assert large.shape[1] > small.shape[1]
