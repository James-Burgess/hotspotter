"""k‑NN search backed by pyflann (FLANN), with faiss fallback."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from wbia_core.data import FeatureSet

# Try pyflann first (primary), fall back to faiss
_HAS_PYFLANN = False
_HAS_FAISS = False

try:
    from pyflann import FLANN as _PyFlann

    _HAS_PYFLANN = True
except ImportError:
    pass

if not _HAS_PYFLANN:
    try:
        import faiss as _faiss

        _HAS_FAISS = True
    except ImportError:
        pass


FLANNIndex = Any


def build_index(
    features: FeatureSet,
    algorithm: str = "kdtree",
    trees: int = 4,
    random_seed: int = 42,
) -> FLANNIndex:
    """Build a FLANN index for *features*.

    Args:
        features: descriptors shape [N, D] uint8.
        algorithm: FLANN algorithm (``"kdtree"``, ``"kmeans"``, ``"linear"``, etc.).
        trees: number of kd-trees (only used with ``"kdtree"``).
        random_seed: seed for deterministic index building.

    Returns:
        Opaque index object (``pyflann.FLANN`` or ``faiss.Index``).
    """
    if _HAS_PYFLANN:
        data = features.descriptors.astype(np.float32, copy=False)
        index = _PyFlann()
        index.build_index(
            data, algorithm=algorithm, trees=trees, random_seed=random_seed
        )
        return index
    elif _HAS_FAISS:
        index = _faiss.IndexFlatL2(features.descriptors.shape[1])
        index.add(features.descriptors.astype(np.float32, copy=False))
        return index
    else:
        raise ImportError("No k‑NN backend available — install pyflann or faiss-cpu")


def build_global_index(
    feature_sets: list[FeatureSet],
    algorithm: str = "kdtree",
    trees: int = 4,
    random_seed: int = 42,
) -> tuple[FLANNIndex, np.ndarray, np.ndarray]:
    """Build a single FLANN index over all descriptors from *feature_sets*.

    Returns:
        ``(index, annot_indices, feat_indices)`` where:
          * *annot_indices* — ``[total_descriptors]`` — which annotation
            each descriptor originated from.
          * *feat_indices* — ``[total_descriptors]`` — which local feature
            index each descriptor corresponds to.
    """
    total_n = sum(len(fs) for fs in feature_sets)
    annot_indices = np.empty(total_n, dtype=np.int32)
    feat_indices = np.empty(total_n, dtype=np.int32)
    chunks = []
    offset = 0
    for i, fs in enumerate(feature_sets):
        n = len(fs)
        annot_indices[offset : offset + n] = i
        feat_indices[offset : offset + n] = np.arange(n, dtype=np.int32)
        chunks.append(fs.descriptors.astype(np.float32, copy=False))
        offset += n

    all_descriptors = np.concatenate(chunks, axis=0)

    if _HAS_PYFLANN:
        index = _PyFlann()
        index.build_index(
            all_descriptors,
            algorithm=algorithm,
            trees=trees,
            random_seed=random_seed,
        )
    elif _HAS_FAISS:
        index = _faiss.IndexFlatL2(all_descriptors.shape[1])
        index.add(all_descriptors)
    else:
        raise ImportError("No k‑NN backend available — install pyflann or faiss-cpu")

    return index, annot_indices, feat_indices


def query_index(
    index: FLANNIndex,
    query_features: FeatureSet,
    k: int,
    checks: int = 1028,
    cores: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Search *index* for the *k* nearest neighbours of each query descriptor.

    Args:
        index: index from :func:`build_index` or :func:`build_global_index`.
        query_features: shape [M, D] uint8.
        k: number of neighbours to return.
        checks: FLANN checks parameter (only used with pyflann).
        cores: FLANN cores parameter, 0 = all (only used with pyflann).

    Returns:
        ``(distances [M, k] float32, labels [M, k] int32)``.
    """
    q = query_features.descriptors.astype(np.float32, copy=False)
    if _HAS_PYFLANN:
        indices, distances = index.nn_index(q, k, checks=checks, cores=cores)
        return distances.astype(np.float32, copy=False), indices
    elif _HAS_FAISS:
        return index.search(q, k)
    else:
        raise ImportError("No k-NN backend available")


def exact_knn(
    query_features: FeatureSet,
    database_features: FeatureSet,
    k: int,
    chunk_size: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-NN search using chunked numpy dot products.

    Computes squared Euclidean distances via::
        |q_i - d_j|^2 = |q_i|^2 + |d_j|^2 - 2 * q_i · d_j

    Query features are processed in *chunk_size* batches to keep memory
    bounded at ``chunk_size × N × 8`` bytes.

    Args:
        query_features: shape [M, D] uint8.
        database_features: shape [N, D] uint8.
        k: number of neighbours to return.
        chunk_size: query batch size (default 500).

    Returns:
        ``(distances [M, k] float32, labels [M, k] int32)``.
    """
    q = query_features.descriptors.astype(np.float64, copy=False)
    d = database_features.descriptors.astype(np.float64, copy=False)
    m = q.shape[0]
    n = d.shape[0]

    d_norm = np.sum(d**2, axis=1)  # [N]

    all_labels = np.empty((m, k), dtype=np.int32)
    all_dists = np.empty((m, k), dtype=np.float32)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        q_chunk = q[start:end]
        q_norm = np.sum(q_chunk**2, axis=1)  # [chunk]
        qd = q_chunk.dot(d.T)  # [chunk, N]
        sq_dists = q_norm[:, None] + d_norm[None, :] - 2.0 * qd
        sq_dists = np.maximum(sq_dists, 0.0)

        if k < n:
            idxs = np.argpartition(sq_dists, k, axis=1)[:, :k]
            top = np.take_along_axis(sq_dists, idxs, axis=1)
            order = np.argsort(top, axis=1)
            all_labels[start:end] = np.take_along_axis(idxs, order, axis=1).astype(
                np.int32
            )
            all_dists[start:end] = np.take_along_axis(top, order, axis=1).astype(
                np.float32
            )
        else:
            all_labels[start:end] = np.tile(
                np.arange(n, dtype=np.int32), (end - start, 1)
            )
            all_dists[start:end] = sq_dists.astype(np.float32)

    return all_dists, all_labels
