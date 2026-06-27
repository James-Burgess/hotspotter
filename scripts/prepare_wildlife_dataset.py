#!/usr/bin/env python3
"""Prepare a wildlife-datasets dataset as a Hotspotter/WBIA benchmark batch.

Downloads a dataset from the ``wildlife_datasets`` catalogue, then converts it
into the ``batch.json`` schema consumed by ``run_fixture.py`` (hotspotter) and
``run_wbia_on_batch50.py`` (WBIA), with **single-label individual identity**
ground truth — the clean target for a Top-1/mAP comparison.

Images are pre-chipped (bbox crop applied here) so both pipelines see identical
chips and chipping is removed as a variable. ``bbox`` in the output is the full
chip extent.

Usage:
    python scripts/prepare_wildlife_dataset.py ATRW \\
        --out-batch ../batches/atrw.json \\
        --out-image-dir ../batches/atrw_images \\
        --max-individuals 40 --max-per-individual 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from wildlife_datasets import datasets as wds


def _get_dataset_class(name: str):
    cls = getattr(wds, name, None)
    if cls is None or not isinstance(cls, type):
        print(
            f"ERROR: no dataset class named {name!r} in wildlife_datasets.datasets",
            file=sys.stderr,
        )
        sys.exit(2)
    return cls


def _parse_bbox(value):
    """Best-effort parse of a bbox cell into (x, y, w, h) ints, else None."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        try:
            xs = [int(round(float(v))) for v in value]
        except (TypeError, ValueError):
            return None
        if len(xs) == 4:
            return xs
    if isinstance(value, str):
        s = value.strip().strip("[]()")
        try:
            xs = [int(round(float(p))) for p in s.split(",")]
        except ValueError:
            return None
        if len(xs) == 4:
            return xs
    return None


def _select_subset(df, max_individuals, max_per_individual, min_per_individual):
    """Keep individuals with >=min images, cap counts, return filtered df."""
    col = "identity"
    if col not in df.columns:
        print(
            f"ERROR: dataframe has no 'identity' column: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(2)

    counts = df.groupby(col).size()
    keep_id = counts[counts >= min_per_individual].sort_values(ascending=False)
    if max_individuals is not None:
        keep_id = keep_id.head(max_individuals)
    if keep_id.empty:
        print("ERROR: no individual has >= min_per_individual images", file=sys.stderr)
        sys.exit(2)

    parts = []
    for identity in keep_id.index:
        rows = df[df[col] == identity]
        if max_per_individual is not None:
            rows = rows.head(max_per_individual)
        parts.append(rows)
    return pd.concat(parts).reset_index(drop=True)


def _load_catalogue(cls, root: Path) -> "pd.DataFrame":
    """Load the dataset dataframe via the wildlife-datasets class, falling back
    to a direct CSV read (``reid_list_<split>.csv`` → identity, <split>/path)
    for datasets whose class loader is broken by version mismatches."""
    try:
        dataset = cls(root=str(root), remove_unknown=True)
        return dataset.df.copy()
    except Exception as ex:
        print(f"  class loader failed ({type(ex).__name__}); trying CSV fallback")

    rows = []
    for csv in sorted(root.glob("**/reid_list_*.csv")):
        split = csv.stem.replace("reid_list_", "")  # e.g. 'train', 'test'
        img_subdir = csv.parent.parent / split  # ../<split>/<filename>
        if not img_subdir.is_dir():
            img_subdir = csv.parent / split
        try:
            sub = pd.read_csv(csv, header=None, names=["identity", "filename"])
        except Exception:
            continue
        sub["path"] = sub["filename"].astype(str)
        sub["path"] = sub["path"].apply(
            lambda f: (
                str((img_subdir / f).relative_to(root))
                if (img_subdir / f).exists()
                else f"{split}/{f}"
            )
        )
        rows.append(sub[["identity", "path"]])
    if not rows:
        print(f"ERROR: could not load catalogue from {root}", file=sys.stderr)
        sys.exit(2)
    df = pd.concat(rows).reset_index(drop=True)
    df["identity"] = pd.to_numeric(df["identity"], errors="coerce")
    df = df.dropna(subset=["identity"]).copy()
    df["identity"] = df["identity"].astype(int)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset", help="wildlife_datasets class name, e.g. ATRW")
    p.add_argument(
        "--root",
        default=None,
        help="dataset download root (default: ../wildlife-data/<DATASET>)",
    )
    p.add_argument("--out-batch", required=True, help="output batch.json path")
    p.add_argument("--out-image-dir", required=True, help="output flat chip dir")
    p.add_argument("--max-individuals", type=int, default=40)
    p.add_argument("--max-per-individual", type=int, default=4)
    p.add_argument(
        "--min-per-individual",
        type=int,
        default=2,
        help="require >=N images/individual (need a positive match)",
    )
    p.add_argument("--queries-per-individual", type=int, default=1)
    p.add_argument(
        "--force", action="store_true", help="re-download even if marked present"
    )
    args = p.parse_args(argv)

    cls = _get_dataset_class(args.dataset)
    infra_root = Path(__file__).resolve().parents[2]
    root = Path(args.root) if args.root else infra_root / "wildlife-data" / args.dataset
    root.mkdir(parents=True, exist_ok=True)

    # 1. Download (idempotent via mark file)
    mark = root / ".wildlife_downloaded"
    if not mark.exists() or args.force:
        print(f"Downloading {args.dataset} → {root}")
        cls.download(root=str(root), force=args.force)
        mark.touch()

    # 2. Load catalogue (class first, then CSV fallback for finicky datasets)
    print(f"Loading catalogue from {root}")
    df = _load_catalogue(cls, root)
    print(f"  rows={len(df)}  identities={df['identity'].nunique()}")
    print(f"  columns={list(df.columns)}")

    # 3. Select subset
    sel = _select_subset(
        df, args.max_individuals, args.max_per_individual, args.min_per_individual
    )
    print(f"  selected: {len(sel)} rows, {sel['identity'].nunique()} individuals")

    # 4. Chip + write
    out_batch = Path(args.out_batch)
    out_imgs = Path(args.out_image_dir)
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_batch.parent.mkdir(parents=True, exist_ok=True)

    annotations = []
    annot_id = 1  # 1-indexed so trace `daid` == `annot_id` for both HS and WBIA
    skipped = 0
    for _, row in sel.iterrows():
        rel = str(row["path"])
        src = root / rel
        if not src.exists():
            skipped += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception as ex:
            print(f"  skip unreadable {src.name}: {ex}")
            skipped += 1
            continue

        box = _parse_bbox(row.get("bbox")) if "bbox" in row else None
        if box is not None:
            x, y, w, h = box
            img = img.crop((x, y, x + w, y + h))

        fname = f"{args.dataset.lower()}_{annot_id:06d}.jpg"
        img.save(out_imgs / fname, quality=95)
        W, H = img.size
        annotations.append(
            {
                "annot_id": annot_id,
                "image_id": _as_int(row.get("image_id"), annot_id),
                "file_name": fname,
                "uri": f"/app/batch_images/{fname}",
                "bbox": [0, 0, W, H],
                "individual_ids": [_as_int(row["identity"], str(row["identity"]))],
                "is_query": False,
            }
        )
        annot_id += 1

    # 5. Designate queries (one per individual, first image)
    by_ind = {}
    for a in annotations:
        key = a["individual_ids"][0]
        by_ind.setdefault(key, []).append(a)
    n_queries = 0
    for key, anns in by_ind.items():
        # spread queries across the available images
        nq = min(args.queries_per_individual, len(anns) - 1) if len(anns) > 1 else 0
        for a in anns[:nq]:
            a["is_query"] = True
            n_queries += 1

    batch = {
        "seed": 42,
        "dataset": args.dataset,
        "n_annots": len(annotations),
        "n_queries": n_queries,
        "annotations": annotations,
    }
    out_batch.write_text(json.dumps(batch, indent=2))
    print(
        f"\nWrote {out_batch} ({len(annotations)} annots, {n_queries} queries, "
        f"{len(by_ind)} individuals, skipped {skipped})"
    )


def _as_int(value, default):
    """Return int(value) if possible, else ``default`` (int or str)."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
