#!/usr/bin/env python3
"""Create a 50-zebra-image batch JSON from YOLO dataset. No external deps."""

import argparse
import json
import os
import random
import shutil
import struct
from pathlib import Path


def jpeg_dims(path):
    """Read JPEG dimensions without any library."""
    with open(path, "rb") as f:
        f.seek(0)
        while True:
            marker = f.read(2)
            if marker != b"\xff\xd8":
                break  # not JPEG
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
    if w < 20 or h < 20:
        return None
    return [x, y, w, h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="/app/images/train")
    parser.add_argument("--label-dir", default="/app/labels/train")
    parser.add_argument("--out-dir", default="/app/batch50_images")
    parser.add_argument("--batch-json", default="/app/batch50.json")
    parser.add_argument("--n-images", type=int, default=50)
    parser.add_argument("--n-queries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--species", type=int, default=3, help="3=zebra, 1=elephant, 0=buffalo, 2=rhino"
    )
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--individual-id", type=int, default=5000)
    args = parser.parse_args()

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

    zebra = [c for c in candidates if c[2] == args.species]
    rng.shuffle(zebra)
    selected = zebra[: args.n_images]

    if args.mixed and len(selected) < args.n_images:
        other = [c for c in candidates if c[2] != args.species]
        rng.shuffle(other)
        selected.extend(other[: args.n_images - len(selected)])

    annotations = []
    annot_id = 100
    os.makedirs(args.out_dir, exist_ok=True)

    for lf, li, cls, bbox_norm in selected:
        img_stem = lf.stem
        img_file = image_dir / f"{img_stem}.jpg"
        if not img_file.exists():
            continue

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
                "uri": f"/app/batch50_images/{img_stem}.jpg",
                "bbox": bbox,
                "individual_ids": [args.individual_id],
                "is_query": False,
            }
        )

        dst = Path(args.out_dir) / f"{img_stem}.jpg"
        if not dst.exists():
            shutil.copy2(img_file, dst)

        annot_id += 1

    n_annots = len(annotations)
    if n_annots < args.n_queries + 1:
        print(
            f"ERROR: only {n_annots} annotations, need {args.n_queries + 1}", flush=True
        )
        return 1

    for i in range(args.n_queries):
        annotations[i]["is_query"] = True

    batch = {
        "seed": args.seed,
        "n_annots": n_annots,
        "n_queries": args.n_queries,
        "annotations": annotations,
    }

    with open(args.batch_json, "w") as f:
        json.dump(batch, f, indent=2)

    print(
        f"Batch: {n_annots} annots ({args.n_queries} queries), class={args.species} -> {args.batch_json}",
        flush=True,
    )
    print(f"Images copied to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
