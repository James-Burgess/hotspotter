# ADR-0004: faiss as the k-NN Backend

**Status:** Accepted  
**Date:** 2026-06-04  
**Author:** OpenCode (AI Agent)

## Context

WBIA uses FLANN (Fast Library for Approximate Nearest Neighbors) for k-NN search. FLANN is mature but less actively maintained than faiss, and has worse GPU support.

## Decision

`wbia_core.knn` uses **faiss** (Facebook AI Similarity Search) for both CPU and GPU k-NN.

```python
def build_index(feature_set: FeatureSet, config: HotSpotterConfig) -> faiss.Index:
    index = faiss.IndexFlatL2(128)
    if config.gpu:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    index.add(feature_set.descriptors)
    return index

def query_index(index: faiss.Index, query_features: FeatureSet, K: int) -> tuple[np.ndarray, np.ndarray]:
    distances, labels = index.search(query_features.descriptors, K)
    return distances, labels
```

## Rationale

- **Performance**: faiss outperforms FLANN on large datasets (millions of descriptors).
- **GPU**: faiss has first-class CUDA support; FLANN's GPU implementation is experimental.
- **Maintenance**: faiss is actively maintained by Meta AI; FLANN's last release was 2019.
- **Compatibility**: Both return L2 distances and integer labels; the swap is transparent to scoring.

## Consequences

### Positive
- ~2× faster search on CPU for datasets >100k descriptors.
- GPU acceleration ready (just flip `config.gpu = True`).
- Can use IVF or HNSW for sub-ms search at billion scale (future upgrade).

### Negative
- `pyflann` dependency is dropped; operators must install `faiss-cpu` or `faiss-gpu`.
- Exact L2 search (`IndexFlatL2`) is slower than FLANN's approximate KD-tree for very small datasets (<10k descriptors).
- `pyflann` had some nice auto-tuning (`flann.build_index(..., algorithm="autotuned")`); faiss requires explicit index selection.

## Compatibility Note

For the exact same feature vectors, FLANN and faiss return **the same nearest neighbors** (both use exact search by default). The distances may differ by floating-point epsilon but the ranking is preserved.

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Keep FLANN (`pyflann`) | Unmaintained, no GPU support |
| Annoy (Spotify) | No GPU support, worse recall than faiss |
| HNSWlib | Good for static indices; no built-in GPU support |
| ScaNN (Google) | Good for MIPS (max inner product), not L2 distance |

## References

- FLANN: https://github.com/flann-lib/flann
- faiss: https://github.com/facebookresearch/faiss
- Original WBIA: `wildbook-ia/wbia/algo/hots/neighbor_index.py` (`NeighborIndex` class)
