"""Tests for wbia_core.config."""

import pytest
from pydantic import ValidationError

from wbia_core.config import HotSpotterConfig, IdentificationConfig, SiftConfig


class TestSiftConfig:
    def test_defaults(self):
        c = SiftConfig()
        assert c.scale == [1.0, 4.0, 8.0]
        assert c.ori_hist_bins == 36

    def test_invalid_bins(self):
        with pytest.raises(ValidationError):
            SiftConfig(ori_hist_bins=5)


class TestHotSpotterConfig:
    def test_defaults(self):
        c = HotSpotterConfig()
        assert c.knn == 4
        assert c.score_method == "csum"
        assert c.sv_on is True

    def test_invalid_knn(self):
        with pytest.raises(ValidationError):
            HotSpotterConfig(knn=0)

    def test_invalid_score_method(self):
        with pytest.raises(ValidationError):
            HotSpotterConfig(score_method="avg")  # type: ignore


class TestIdentificationConfig:
    def test_defaults(self):
        c = IdentificationConfig()
        assert c.pipeline == "HotSpotter"
        assert isinstance(c.hotspotter, HotSpotterConfig)

    def test_pipeline_enum(self):
        with pytest.raises(ValidationError):
            IdentificationConfig(pipeline="Unknown")  # type: ignore

    def test_serialize_roundtrip(self):
        c = IdentificationConfig(
            pipeline="MiewId",
            hotspotter=HotSpotterConfig(knn=10, sv_on=False),
        )
        data = c.model_dump()
        restored = IdentificationConfig(**data)
        assert restored.pipeline == "MiewId"
        assert restored.hotspotter.knn == 10
        assert restored.hotspotter.sv_on is False
