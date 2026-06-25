#!/usr/bin/env python3
"""Run :func:`hotspotter.identify` against a fixture batch of images and bboxes.

Reads a JSON reference batch with ``annotations`` list containing
``file_name``, ``bbox``, ``is_query``, and optional ``individual_ids``,
extracts chips, runs identification, and prints scored results as JSON.

Usage::

    python scripts/run_fixture.py pipeline/tests/reference_batch.json
    python scripts/run_fixture.py pipeline/tests/reference_batch.json --config '{"sv_on": true}'
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import cv2
import numpy as np

from hotspotter.chip import extract_chip
from hotspotter.config import HotSpotterConfig, IdentificationConfig, SiftConfig
from hotspotter.data import AnnotatedImage
from hotspotter.features import extract_features
from hotspotter.pipeline import identify


def load_batch(batch_path: str | Path) -> dict:
    with open(batch_path) as f:
        return json.load(f)


def build_database(
    batch: dict,
    image_dir: str | Path,
) -> tuple[list[AnnotatedImage], list[int], list[str]]:
    """Build an ``AnnotatedImage`` list from the reference batch.

    Returns ``(database, query_indices, annot_filenames)``.
    Query annotations are placed at the start of the ``database`` list
    so that ``identify(query_annot_index=i, database=database)`` works.
    """
    image_dir = Path(image_dir)
    annots = batch["annotations"]

    database: list[AnnotatedImage] = []
    query_indices: list[int] = []
    annot_filenames: list[str] = []

    name_to_uuid: dict[str, uuid.UUID] = {}
    seen_ids: set[str] = set()

    for ann in annots:
        if ann["annot_id"] in seen_ids:
            continue
        img_path = image_dir / ann["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        bbox = ann["bbox"]
        chip = extract_chip(img, bbox)
        features = extract_features(chip, SiftConfig())
        individual_ids = ann.get("individual_ids", [])
        nid = str(individual_ids[0]) if individual_ids else f"name_{ann['annot_id']}"
        if nid not in name_to_uuid:
            name_to_uuid[nid] = uuid.uuid5(uuid.NAMESPACE_DNS, nid)
        if ann.get("is_query") or ann.get("query"):
            query_indices.append(len(database))
        database.append(
            AnnotatedImage(
                annot_uuid=uuid.uuid5(uuid.NAMESPACE_DNS, f"annot_{ann['annot_id']}"),
                name_uuid=name_to_uuid[nid],
                image=chip,
                features=features,
                bbox=tuple(bbox),
            )
        )
        annot_filenames.append(ann["file_name"])
        seen_ids.add(ann["annot_id"])

    return database, query_indices, annot_filenames


def run_identify(
    database: list[AnnotatedImage],
    query_indices: list[int],
    annot_filenames: list[str],
    config_overrides: dict | None = None,
) -> list[dict]:
    """Run :func:`identify` for each query and collect results."""
    hs_kwargs: dict = {}
    if config_overrides:
        hs_kwargs.update(config_overrides)

    if "fg_on" not in hs_kwargs:
        hs_kwargs["fg_on"] = False

    hs_config = HotSpotterConfig(**hs_kwargs)
    id_config = IdentificationConfig(hotspotter=hs_config)

    all_results: list[dict] = []

    for trace_idx, qi in enumerate(query_indices):
        scored = identify(qi, database, id_config, trace_query_index=trace_idx)
        annot = database[qi]

        result = {
            "query_index": qi,
            "query_annot_uuid": str(annot.annot_uuid),
            "query_file": annot_filenames[qi],
            "config": hs_kwargs,
            "matches": [],
        }

        for sm in scored:
            db_file = None
            try:
                db_idx = next(
                    i for i, a in enumerate(database) if a.annot_uuid == sm.annot_uuid
                )
                db_file = (
                    annot_filenames[db_idx] if db_idx < len(annot_filenames) else None
                )
            except StopIteration:
                pass
            result["matches"].append(
                {
                    "annot_uuid": str(sm.annot_uuid),
                    "name_uuid": str(sm.name_uuid) if sm.name_uuid else None,
                    "score": round(float(sm.score), 8),
                    "num_matches": sm.num_matches,
                    "sv_inliers": sm.sv_inliers,
                    "image": db_file,
                }
            )

        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run hotspotter fixture pipeline and print results as JSON."
    )
    parser.add_argument("batch_path", help="Path to reference_batch.json")
    parser.add_argument(
        "--config",
        default=None,
        help='JSON config overrides for HotSpotterConfig (e.g. {"sv_on": false})',
    )
    parser.add_argument("--image-dir", default=None, help="Directory containing images")
    args = parser.parse_args()

    batch = load_batch(args.batch_path)
    image_dir = args.image_dir or str(
        Path(args.batch_path).parent / "assets" / "images"
    )

    database, query_indices, annot_filenames = build_database(batch, image_dir)

    config = json.loads(args.config) if args.config else None
    results = run_identify(database, query_indices, annot_filenames, config)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
