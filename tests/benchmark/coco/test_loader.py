"""Tests for the COCO subset loader."""

from pathlib import Path

import pytest

from tests.benchmark.coco.loader import CocoLoader

COCO_JSON = Path("tests/test-dataset/annotations/instances_train2020.json")
IMAGE_DIR = Path("tests/test-dataset/images")

_TOTAL_ANNOTS = 19
_VALID_INDIVIDUALS = 3

pytestmark = pytest.mark.skipif(
    not COCO_JSON.exists() or not IMAGE_DIR.exists(),
    reason="benchmark COCO fixture dataset is not present",
)


class TestCocoLoader:
    def test_loads_json(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        assert len(loader._raw.get("annotations", [])) == _TOTAL_ANNOTS

    def test_select_subset_default(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        subset = loader.select_subset()
        assert len(subset.annotations) == _TOTAL_ANNOTS
        assert len(subset.query_indices) == _VALID_INDIVIDUALS
        assert subset.config["species"] is None
        assert subset.config["seed"] == 42

    def test_select_subset_species_filter(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        subset = loader.select_subset(species="zebra_plains")
        assert len(subset.annotations) == _TOTAL_ANNOTS
        for ann in subset.annotations:
            assert ann.species == "zebra_plains"

    def test_select_subset_deterministic(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        a = loader.select_subset(seed=42)
        b = loader.select_subset(seed=42)
        a_ids = [ann.annot_id for ann in a.annotations]
        b_ids = [ann.annot_id for ann in b.annotations]
        assert a_ids == b_ids

    def test_select_subset_different_seed(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        a = loader.select_subset(seed=42)
        b = loader.select_subset(seed=99)
        a_ids = [ann.annot_id for ann in a.annotations]
        b_ids = [ann.annot_id for ann in b.annotations]
        assert a_ids != b_ids

    def test_select_subset_n_queries(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        subset = loader.select_subset(n_queries=2)
        assert len(subset.query_indices) == 2
        assert subset.config["n_queries"] == 2

    def test_select_subset_images_loaded(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        subset = loader.select_subset(n_annots=10)
        for ann in subset.annotations:
            assert len(ann.image) > 0
            assert ann.image[:2] == b"\xff\xd8"

    def test_queries_have_db_matches(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        subset = loader.select_subset(n_queries=3, seed=42)
        q_set = set(subset.query_indices)
        db_indices = [i for i in range(len(subset.annotations)) if i not in q_set]
        for qi in subset.query_indices:
            q_ids = set(subset.annotations[qi].individual_ids)
            found = False
            for di in db_indices:
                if q_ids & set(subset.annotations[di].individual_ids):
                    found = True
                    break
            assert (
                found
            ), f"Query annotation {subset.annotations[qi].annot_id} has no DB match"

    def test_query_not_in_database(self):
        loader = CocoLoader(COCO_JSON, IMAGE_DIR)
        subset = loader.select_subset(n_queries=3, seed=42)
        q_annots = {subset.annotations[qi].annot_id for qi in subset.query_indices}
        for i, ann in enumerate(subset.annotations):
            if i not in subset.query_indices:
                assert ann.annot_id not in q_annots
