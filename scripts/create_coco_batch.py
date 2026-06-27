#!/usr/bin/env python3
"""Create batch JSON from COCO zebra dataset with real individual IDs.

Selects N individuals that appear in >=2 annotations. First annotation is
the query; remaining annotations from the same individual are positives.
Adds random annotations from other individuals as negatives.
"""

import argparse, json, random, os
from pathlib import Path
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--coco-json",
        default="/app/test-files/zebras/gzgc.coco/annotations/instances_train2020.json",
    )
    p.add_argument(
        "--image-dir", default="/app/test-files/zebras/gzgc.coco/images/train2020"
    )
    p.add_argument("--out-batch", default="/app/batches/zebra_coco.json")
    p.add_argument("--out-img-dir", default="/app/batches/images")
    p.add_argument("--n-queries", type=int, default=30)
    p.add_argument("--negatives-per-query", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-annots", type=int, default=2, help="Min annotations per individual"
    )
    p.add_argument("--category-id", type=int, default=1, help="1=zebra_plains")
    args = p.parse_args()

    with open(args.coco_json) as f:
        coco = json.load(f)

    rng = random.Random(args.seed)

    # Index: image_id -> file_name
    img_map = {img["id"]: img["file_name"] for img in coco["images"]}

    # Filter zebra annotations, index by individual_id
    annots_by_cat = [
        a for a in coco["annotations"] if a["category_id"] == args.category_id
    ]

    # Group by primary individual_id (first in the list)
    by_indiv: dict[int, list[dict]] = defaultdict(list)
    for a in annots_by_cat:
        inds = a.get("individual_ids", [])
        if inds:
            by_indiv[inds[0]].append(a)

    # Filter individuals with >= min_annots
    multi_annots = {k: v for k, v in by_indiv.items() if len(v) >= args.min_annots}
    print(
        f"Zebra annots: {len(annots_by_cat)}, individuals: {len(by_indiv)}, with >=2: {len(multi_annots)}"
    )

    if len(multi_annots) < args.n_queries:
        print(
            f"ERROR: only {len(multi_annots)} individuals with >=2 annots, need {args.n_queries}"
        )
        return 1

    indiv_ids = list(multi_annots.keys())
    rng.shuffle(indiv_ids)
    selected_indivs = indiv_ids[: args.n_queries]

    # Build annotations
    annotations = []
    annot_id = 1
    used_image_ids = set()

    # Consider annots we don't select as negatives
    all_other_annots = []

    for ind_id in selected_indivs:
        annots = multi_annots[ind_id]
        query_annot = annots[0]
        pos_annots = annots[1:]

        # Query
        if query_annot["image_id"] not in used_image_ids:
            annotations.append(
                _build_annot(
                    query_annot, annot_id, query_annot["individual_ids"], True, img_map
                )
            )
            annot_id += 1
            used_image_ids.add(query_annot["image_id"])

        # Positives from same individual
        for pa in pos_annots:
            if pa["image_id"] not in used_image_ids:
                annotations.append(
                    _build_annot(pa, annot_id, pa["individual_ids"], False, img_map)
                )
                annot_id += 1
                used_image_ids.add(pa["image_id"])

    # Negatives: random other annotations
    for i in indiv_ids[args.n_queries :]:
        if i not in selected_indivs:
            all_other_annots.extend(by_indiv[i])

    rng.shuffle(all_other_annots)
    n_neg = args.n_queries * args.negatives_per_query
    for a in all_other_annots[:n_neg]:
        if a["image_id"] not in used_image_ids:
            annotations.append(
                _build_annot(a, annot_id, a["individual_ids"], False, img_map)
            )
            annot_id += 1
            used_image_ids.add(a["image_id"])

    queries = [a for a in annotations if a["is_query"]]
    pos_db = [
        a
        for a in annotations
        if not a["is_query"]
        and any(set(a["individual_ids"]) & set(q["individual_ids"]) for q in queries)
    ]
    neg_db = [
        a
        for a in annotations
        if not a["is_query"]
        and not any(
            set(a["individual_ids"]) & set(q["individual_ids"]) for q in queries
        )
    ]

    print(
        f"Batch: {len(queries)} queries, {len(pos_db)} positives, {len(neg_db)} negatives, {len(annotations)} total"
    )

    batch = {
        "seed": args.seed,
        "n_annots": len(annotations),
        "n_queries": len(queries),
        "annotations": annotations,
    }

    os.makedirs(os.path.dirname(args.out_batch), exist_ok=True)
    os.makedirs(args.out_img_dir, exist_ok=True)
    with open(args.out_batch, "w") as f:
        json.dump(batch, f, indent=2)
    print(f"Wrote {args.out_batch}")

    # Copy images
    copied = 0
    for a in annotations:
        src = Path(args.image_dir) / a["file_name"]
        dst = Path(args.out_img_dir) / a["file_name"]
        if src.exists() and not dst.exists():
            import shutil

            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} images to {args.out_img_dir}")
    return 0


def _build_annot(ann, annot_id, indiv_ids, is_query, img_map):
    fname = img_map.get(ann["image_id"], "")
    return {
        "annot_id": annot_id,
        "image_id": ann["image_id"],
        "file_name": fname,
        "uri": "/app/batches/images/" + fname,
        "bbox": _coco_bbox(ann["bbox"]),
        "individual_ids": indiv_ids,
        "is_query": is_query,
    }


def _coco_bbox(bbox):
    """COCO [x, y, w, h] — already in absolute pixels, no conversion needed."""
    return list(map(int, bbox))


if __name__ == "__main__":
    raise SystemExit(main())
