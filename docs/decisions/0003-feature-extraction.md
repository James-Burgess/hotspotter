# ADR-0003: pyhesaff SIFT for Feature Extraction

**Status:** Accepted (revised June 2026)  
**Date:** 2026-06-04 (updated 2026-06-06)  
**Author:** OpenCode (AI Agent)

## Context

Feature extraction in WBIA uses `pyhesaff.detect_feats_in_image()`, a Hesaff-SIFT implementation. This is a critical dependency for algorithmic determinism.

**Revision (2026-06-06):** The PyPI wheel `wbia-pyhesaff` 4.0.0 ships transitive dependency `wbia-vtool` which bundles pre-compiled OpenCV 2.4.5 shared libraries. Loading these alongside system OpenCV 4.x causes a SIGSEGV at import time. We now build `wbia-pyhesaff`, `wbia-vtool`, and `wbia-utool` from git submodule source against the system's `libopencv-dev`.

## Decision

`wbia_core.features` uses `pyhesaff` exclusively. There is no OpenCV SIFT fallback.

```python
def extract_features(image: np.ndarray, config: SiftConfig = SiftConfig()) -> FeatureSet:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    keypoints, descriptors = _h.detect_feats_in_image(image, **hesaff_kwargs)
    return FeatureSet(keypoints=keypoints, descriptors=descriptors)
```

Grayscale images are expanded to 3-channel before passing to pyhesaff (the C extension requires 3D arrays).

Output shape:

- **keypoints**: `[N, 6]` — `(x, y, a, b, c, orient)`
- **descriptors**: `[N, 128]` — `uint8` L2-normalized SIFT descriptors

## Build from submodule source

`wbia-pyhesaff` is a git submodule at `wbia-core/wbia-tpl-pyhesaff/`, alongside its transitive deps `wbia-vtool/` and `wbia-utool/`. The Dockerfile builds them in dependency order:

```dockerfile
COPY . /app
RUN pip install --no-cache-dir --no-deps ./wbia-utool/ \
    && pip install --no-cache-dir --no-deps ./wbia-vtool/ \
    && pip install --no-cache-dir --no-deps ./wbia-tpl-pyhesaff/
RUN pip install --no-cache-dir .
```

Each package is installed with `--no-deps` so pip does not pull stale PyPI wheels. The system's `libopencv-dev` and `cmake` provide the C++ build toolchain. `SETUPTOOLS_SCM_PRETEND_VERSION` is set because the Docker build lacks `.git` metadata.

## Consequences

### Positive
- Feature extraction is a single function call matching WBIA exactly.
- No SIGSEGV — all C++ extensions compiled against the same system OpenCV.
- `FeatureSet` is JSON-serializable for caching.
- Easy to mock in tests.

### Negative
- Requires C++ build toolchain (cmake, ninja, gcc, libopencv-dev) in the Docker image.
- No GPU acceleration (SIFT is CPU-only).
- The `[N, 6]` keypoint shape is unusual; documented explicitly.
- Image must be 3D — grayscale is expanded to 3-channel before extraction.

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| `cv2.SIFT_create()` | Different feature values → different match results |
| PyPI wheel for `wbia-pyhesaff` | Bundles OpenCV 2.4.5 via `wbia-vtool` → SIGSEGV |
| `cv2.ORB_create()` | Binary descriptors, not compatible with LNBNN scoring |
| Learned features (SuperPoint, etc.) | Changes the algorithm entirely; out of scope for v1 |

## References

- pyhesaff source: `wbia-core/wbia-tpl-pyhesaff/` (git submodule)
- vtool source: `wbia-core/wbia-vtool/` (git submodule, dependency)
- utool source: `wbia-core/wbia-utool/` (git submodule, dependency)
