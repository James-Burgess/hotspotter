"""Pure data structures — no DB, no controller, no cache."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeatureSet:
    """Extracted Hessian-affine SIFT features for one image.

    Attributes:
        keypoints: [N, 6] float32 — (x, y, a, b, c, orient).
        descriptors: [N, 128] uint8 — L2-normalized SIFT descriptors.
    """

    keypoints: np.ndarray
    descriptors: np.ndarray

    def __post_init__(self):
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] != 6:
            raise ValueError(f"keypoints must be [N, 6], got {self.keypoints.shape}")
        if self.descriptors.ndim != 2 or self.descriptors.shape[1] != 128:
            raise ValueError(
                f"descriptors must be [N, 128], got {self.descriptors.shape}"
            )
        if self.keypoints.shape[0] != self.descriptors.shape[0]:
            raise ValueError(
                f"keypoints ({self.keypoints.shape[0]}) and descriptors "
                f"({self.descriptors.shape[0]}) must agree on N"
            )

    def __len__(self) -> int:
        return self.keypoints.shape[0]

    def __add__(self, other: FeatureSet) -> FeatureSet:
        return FeatureSet(
            keypoints=np.concatenate([self.keypoints, other.keypoints]),
            descriptors=np.concatenate([self.descriptors, other.descriptors]),
        )

    def __repr__(self) -> str:
        return f"FeatureSet(N={len(self)})"


@dataclass
class AnnotatedImage:
    """An annotation with pre-extracted features for identification.

    Pure data — the caller must provide features.  No database lookups.
    """

    annot_uuid: uuid.UUID
    name_uuid: uuid.UUID | None
    image: np.ndarray  # [H, W] or [H, W, 3] uint8
    features: FeatureSet
    bbox: tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class Match:
    """A single query-feature → database-feature correspondence."""

    qfx: int
    daid: int
    dfx: int
    dist: float
    name_uuid: uuid.UUID | None


@dataclass
class ScoredMatch:
    """An annotation-level identification result."""

    annot_uuid: uuid.UUID
    name_uuid: uuid.UUID | None
    score: float
    num_matches: int = 0
    correspondences: list[tuple[int, int]] = field(default_factory=list)
    sv_inliers: int = 0
    sv_homography: np.ndarray | None = None
