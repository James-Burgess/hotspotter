"""COCO dataset loader — reads standard COCO JSON + images and produces subsets."""

from __future__ import annotations

import collections
import json
import pathlib
import random
from dataclasses import dataclass, field

SPECIES_MAP = {
    0: "giraffe_masai",
    1: "zebra_plains",
}


@dataclass
class CocoAnnotation:
    annot_id: int
    image_id: int
    bbox: tuple[int, int, int, int]
    species: str
    individual_ids: list[int]
    image: bytes
    width: int
    height: int


@dataclass
class CocoSubset:
    annotations: list[CocoAnnotation]
    query_indices: list[int]
    config: dict = field(default_factory=dict)


class CocoLoader:
    """Load COCO JSON and images, then select subsets for benchmarking."""

    def __init__(
        self,
        coco_json_path: str | pathlib.Path,
        coco_images_path: str | pathlib.Path,
    ):
        self._images_path = pathlib.Path(coco_images_path)
        with open(coco_json_path) as f:
            self._raw = json.load(f)

        self._cat_name_by_id: dict[int, str] = {}
        for cat in self._raw.get("categories", []):
            self._cat_name_by_id[cat["id"]] = cat["name"]

        self._images_by_id: dict[int, dict] = {}
        for img in self._raw.get("images", []):
            self._images_by_id[img["id"]] = img

    def select_subset(
        self,
        n_annots: int = 100,
        species: str | None = None,
        seed: int = 42,
        n_queries: int = 10,
    ) -> CocoSubset:
        """Select a subset with guaranteed ground-truth matches per query.

        Groups annotations by their primary ``individual_id``, picks
        individuals that appear in at least 2 annotations, then uses one
        annotation as the query and the rest as guaranteed database
        matches.  Remaining slots are filled with random distractors.
        """
        rng = random.Random(seed)

        raw_annots = list(self._raw.get("annotations", []))
        if species:
            raw_annots = [
                a
                for a in raw_annots
                if self._cat_name_by_id.get(a.get("category_id", -1), "") == species
            ]

        # Group by first individual_id (deterministic grouping)
        indiv_groups: dict[int, list[dict]] = collections.defaultdict(list)
        for a in raw_annots:
            ind_ids = a.get("individual_ids", [])
            if ind_ids:
                indiv_groups[ind_ids[0]].append(a)

        # Keep individuals with >= 2 annotations
        valid = {k: v for k, v in indiv_groups.items() if len(v) >= 2}
        ind_ids = sorted(valid.keys())
        rng.shuffle(ind_ids)

        actual_queries = min(n_queries, len(ind_ids))
        query_indivs = ind_ids[:actual_queries]

        # Build selection: query (first annot) + remaining from same individual
        selected_map: dict[int, dict] = {}  # annot_id → raw annot
        query_annot_ids: set[int] = set()
        query_order: list[int] = []  # annot_ids in query order

        for ind_id in query_indivs:
            group = valid[ind_id]
            rng.shuffle(group)
            q_annot = group[0]
            selected_map[q_annot["id"]] = q_annot
            query_annot_ids.add(q_annot["id"])
            query_order.append(q_annot["id"])
            for a in group[1:]:
                if a["id"] not in selected_map:
                    selected_map[a["id"]] = a

        # Fill remaining slots with distractors
        distractor_pool = [a for a in raw_annots if a["id"] not in selected_map]
        rng.shuffle(distractor_pool)

        needed = max(0, n_annots - len(selected_map))
        for a in distractor_pool[:needed]:
            selected_map[a["id"]] = a

        # Build ordered list with queries first
        ordered_ids = list(selected_map.keys())
        query_positions = [ordered_ids.index(qid) for qid in query_order]

        annotations: list[CocoAnnotation] = []
        for a_id in ordered_ids:
            a = selected_map[a_id]
            img_info = self._images_by_id.get(a["image_id"], {})
            img_path = self._resolve_image(img_info.get("file_name", ""))
            image_bytes = img_path.read_bytes() if img_path.exists() else b""

            individual_ids = [int(i) for i in a.get("individual_ids", [])]

            cat_id = a.get("category_id", -1)
            species_name = self._cat_name_by_id.get(cat_id, "")

            bbox_raw = a.get("bbox", [0, 0, 0, 0])
            bbox = (
                int(bbox_raw[0]),
                int(bbox_raw[1]),
                int(bbox_raw[2]),
                int(bbox_raw[3]),
            )

            annotations.append(
                CocoAnnotation(
                    annot_id=a["id"],
                    image_id=a["image_id"],
                    bbox=bbox,
                    species=species_name,
                    individual_ids=individual_ids,
                    image=image_bytes,
                    width=img_info.get("width", 0),
                    height=img_info.get("height", 0),
                )
            )

        return CocoSubset(
            annotations=annotations,
            query_indices=query_positions,
            config={
                "n_annots": len(annotations),
                "n_queries": len(query_positions),
                "species": species,
                "seed": seed,
            },
        )

    def _resolve_image(self, file_name: str) -> pathlib.Path:
        for subdir in ["train2020", "val2020", "test2020"]:
            candidate = self._images_path / subdir / file_name
            if candidate.exists():
                return candidate
        return self._images_path / file_name
