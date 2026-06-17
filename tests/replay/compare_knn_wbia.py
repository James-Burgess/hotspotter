"""Compare k-NN distances in WBIA container (standalone, no wbia-core deps)."""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pyhesaff
from pyflann import FLANN


def main():
    fixtures_dir = pathlib.Path(sys.argv[1])
    out = {}

    defaults = pyhesaff.get_hesaff_default_params()

    for npz_path in sorted(fixtures_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        image_bytes_list = list(data["image_bytes"])
        annot_uuids = [str(u) for u in data["annot_uuids"]]
        query_idx = int(data.get("query_idx", 0))

        all_features = {}
        for i, blob in enumerate(image_bytes_list):
            buf = np.frombuffer(blob, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            kpts, vecs = pyhesaff.detect_feats_in_image(img, **dict(defaults))
            all_features[annot_uuids[i]] = vecs.astype(np.float32)

        query_uid = annot_uuids[query_idx]
        query_feats = all_features[query_uid]
        db_uids = [u for u in annot_uuids if u != query_uid]

        # Global index
        all_db_feats = np.vstack([all_features[u] for u in db_uids])
        flann_global = FLANN()
        flann_global.build_index(
            all_db_feats, algorithm="kdtree", trees=4, random_seed=42
        )
        k_plus_1 = 5
        global_labels, global_dists = flann_global.nn_index(
            query_feats, k_plus_1, checks=1028, cores=0
        )

        # Per-annotation indexes
        per_annot_results = {}
        for uid in db_uids:
            flann = FLANN()
            flann.build_index(
                all_features[uid], algorithm="kdtree", trees=4, random_seed=42
            )
            n_neighbors = min(k_plus_1, all_features[uid].shape[0])
            try:
                labels, dists = flann.nn_index(
                    query_feats, n_neighbors, checks=1028, cores=0
                )
            except Exception:
                labels = np.full(
                    (query_feats.shape[0], n_neighbors), -1, dtype=np.int32
                )
                dists = np.full(
                    (query_feats.shape[0], n_neighbors), 1e10, dtype=np.float32
                )
            per_annot_results[uid] = {
                "n_db_feats": int(all_features[uid].shape[0]),
                "dist_stats": {
                    "min": float(dists.min()),
                    "max": float(dists.max()),
                    "mean": float(dists.mean()),
                    "std": float(dists.std()),
                },
                "vdist_stats": {
                    "min": float(dists[:, 0].min()),
                    "max": float(dists[:, 0].max()),
                    "mean": float(dists[:, 0].mean()),
                },
                "ndist_stats": {
                    "min": float(dists[:, -1].min()),
                    "max": float(dists[:, -1].max()),
                    "mean": float(dists[:, -1].mean()),
                },
            }

        out[npz_path.name] = {
            "query_idx": query_idx,
            "query_uid": query_uid,
            "n_query_feats": int(query_feats.shape[0]),
            "db_annot_counts": {u: int(all_features[u].shape[0]) for u in db_uids},
            "global": {
                "dist_stats": {
                    "min": float(global_dists.min()),
                    "max": float(global_dists.max()),
                    "mean": float(global_dists.mean()),
                    "std": float(global_dists.std()),
                },
                "vdist_stats": {
                    "min": float(global_dists[:, 0].min()),
                    "max": float(global_dists[:, 0].max()),
                    "mean": float(global_dists[:, 0].mean()),
                },
                "ndist_stats": {
                    "min": float(global_dists[:, -1].min()),
                    "max": float(global_dists[:, -1].max()),
                    "mean": float(global_dists[:, -1].mean()),
                },
            },
            "per_annot": per_annot_results,
        }

    sys.stdout.write(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
