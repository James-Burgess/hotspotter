#!/usr/bin/env python3
"""Create an N-image batch JSON from YOLO dataset with positive + negative splits."""

import argparse, json, os, random, shutil, struct
from pathlib import Path


def jpeg_dims(path):
    with open(path, "rb") as f:
        f.seek(0)
        while True:
            marker = f.read(2)
            if marker != b"\xff\xd8":
                break
            while True:
                tag = f.read(2)
                if not tag or tag[0] != 0xFF:
                    return None, None
                if tag in (b"\xff\xd8", b"\xff\xd9"):
                    continue
                if tag[0] == 0xFF and 0xC0 <= tag[1] <= 0xC2:
                    f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                length = struct.unpack(">H", f.read(2))[0]
                f.seek(length - 2, 1)
                if tag in (b"\xff\xda", b"\xff\xd9"):
                    return None, None


def yolo_to_abs(bbox_norm, img_w, img_h):
    cx, cy, bw, bh = bbox_norm
    x = max(0, int((cx - bw / 2) * img_w))
    y = max(0, int((cy - bh / 2) * img_h))
    w = min(img_w - x, int(bw * img_w))
    h = min(img_h - y, int(bh * img_h))
    return [x, y, w, h] if w >= 20 and h >= 20 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="/app/images/train")
    p.add_argument("--label-dir", default="/app/labels/train")
    p.add_argument("--out-dir", default="/app/pipeline/tests/assets/batch100")
    p.add_argument("--batch-json", default="/app/pipeline/tests/batch100.json")
    p.add_argument("--n-positive", type=int, default=70, help="Zebra (same individual)")
    p.add_argument("--n-negative1", type=int, default=15, help="Elephant")
    p.add_argument("--n-negative2", type=int, default=15, help="Buffalo")
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--positive-species", type=int, default=3, help="3=zebra")
    p.add_argument("--pos-id", type=int, default=5000)
    p.add_argument("--neg1-id", type=int, default=6000)
    p.add_argument("--neg2-id", type=int, default=7000)
    args = p.parse_args()

    label_dir = Path(args.label_dir)
    image_dir = Path(args.image_dir)
    rng = random.Random(args.seed)

    candidates = []
    for lf in sorted(label_dir.glob("*.txt")):
        lines = lf.read_text().strip().splitlines()
        for li, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            bbox_norm = [float(p) for p in parts[1:5]]
            candidates.append((lf, li, cls, bbox_norm))

    def pick(species, n, indiv_id):
        pool = [c for c in candidates if c[2] == species]
        rng.shuffle(pool)
        if len(pool) < n:
            print(f"WARNING: only {len(pool)} images for species {species}, need {n}")
            n = len(pool)
        return pool[:n], indiv_id

    positive, pos_id = pick(args.positive_species, args.n_positive, args.pos_id)
    neg1, neg1_id = pick(1, args.n_negative1, args.neg1_id)  # elephant
    neg2, neg2_id = pick(0, args.n_negative2, args.neg2_id)  # buffalo

    selected = [
        (c, pos_id, kind)
        for c, kind in (
            [(p, 0) for p in positive] + [(n, 1) for n in neg1] + [(n, 2) for n in neg2]
        )
    ]
    rng.shuffle(selected)

    annotations = []
    annot_id = 100
    os.makedirs(args.out_dir, exist_ok=True)
    used_images = set()

    for (lf, li, cls, bbox_norm), indiv_id, kind in selected:
        img_stem = lf.stem
        img_file = image_dir / f"{img_stem}.jpg"
        if not img_file.exists():
            continue

        # Allow multiple annotations per image
        img_key = f"{img_stem}_{li}"
        if img_key in used_images:
            continue
        used_images.add(img_key)

        w, h = jpeg_dims(str(img_file))
        if w is None:
            continue
        bbox = yolo_to_abs(bbox_norm, w, h)
        if bbox is None:
            continue

        annotations.append(
            {
                "annot_id": annot_id,
                "image_id": annot_id,
                "file_name": f"{img_stem}.jpg",
                "uri": f"/app/batch100_images/{img_stem}.jpg",
                "bbox": bbox,
                "individual_ids": [indiv_id],
                "is_query": False,
            }
        )
        dst = Path(args.out_dir) / f"{img_stem}.jpg"
        if not dst.exists():
            shutil.copy2(img_file, dst)
        annot_id += 1

    # Mark first N as queries
    n_queries = min(args.n_queries, len(annotations) - 1)
    for i in range(n_queries):
        annotations[i]["is_query"] = True

    batch = {
        "seed": args.seed,
        "n_annots": len(annotations),
        "n_queries": n_queries,
        "annotations": annotations,
    }
    with open(args.batch_json, "w") as f:
        json.dump(batch, f, indent=2)

    pos_count = sum(1 for a in annotations if a["individual_ids"][0] == args.pos_id)
    neg1c = sum(1 for a in annotations if a["individual_ids"][0] == args.neg1_id)
    neg2c = sum(1 for a in annotations if a["individual_ids"][0] == args.neg2_id)
    qpos = sum(
        1
        for a in annotations
        if a["is_query"] and a["individual_ids"][0] == args.pos_id
    )
    qneg1 = sum(
        1
        for a in annotations
        if a["is_query"] and a["individual_ids"][0] == args.neg1_id
    )
    qneg2 = sum(
        1
        for a in annotations
        if a["is_query"] and a["individual_ids"][0] == args.neg2_id
    )
    print(f"Batch: {len(annotations)} annots ({n_queries} queries)")
    print(f"  Positive (zebra, id={args.pos_id}): {pos_count} ({qpos} queries)")
    print(f"  Negative1 (elephant, id={args.neg1_id}): {neg1c} ({qneg1} queries)")
    print(f"  Negative2 (buffalo, id={args.neg2_id}): {neg2c} ({qneg2} queries)")
    print(f"  → {args.batch_json}")


if __name__ == "__main__":
    raise SystemExit(main())
