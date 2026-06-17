"""Compare SIFT features extracted from fixture images in two environments.

Usage (inside Docker container):
    python compare_features.py /tmp/fixtures /tmp/features_output.json

Then copy the JSON output and compare across environments.
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

from wbia_core.config import SiftConfig
from wbia_core.features import extract_features


def main():
    fixtures_dir = pathlib.Path(sys.argv[1])
    outpath = pathlib.Path(sys.argv[2])

    result = {}
    for npz_path in sorted(fixtures_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        image_bytes_list = list(data["image_bytes"])
        species = str(data.get("species", ""))

        fixture_result = []
        for i, blob in enumerate(image_bytes_list):
            buf = np.frombuffer(blob, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            sift_cfg = SiftConfig()
            feats = extract_features(img, sift_cfg)

            # quantizable summary
            kpts = feats.keypoints
            descs = feats.descriptors
            fixture_result.append(
                {
                    "idx": i,
                    "n_kpts": int(kpts.shape[0]),
                    "descriptor_shape": list(descs.shape),
                    "kpt_bounds": {
                        "x": [float(kpts[:, 0].min()), float(kpts[:, 0].max())],
                        "y": [float(kpts[:, 1].min()), float(kpts[:, 1].max())],
                    },
                    "descriptor_stats": {
                        "min": float(descs.min()),
                        "max": float(descs.max()),
                        "mean": float(descs.mean()),
                        "std": float(descs.std()),
                    },
                    "descriptor_first_16": [int(v) for v in descs[0, :16]],
                }
            )

        result[npz_path.name] = {
            "species": species,
            "n_annots": len(fixture_result),
            "annots": fixture_result,
        }

    outpath.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(result)} fixture comparisons to {outpath}")


if __name__ == "__main__":
    main()
