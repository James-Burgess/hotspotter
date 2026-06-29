# Dependency Stripping Plan

Hotspotter currently vendors four wbia-flavoured repos (vtool, pyflann,
pyhesaff, utool) as full submodules. We only use a fraction of their code.
Goal: pull **only the code hotspotter actually calls** directly into this
repo, eliminate the vendored submodules, and drop the utool dependency.

## Current vendored deps

| Vendor | Lines (Python) | Provides | hotspottter uses |
|---|---|---|---|
| `wbia-vtool/` | 41 `.py` files | Full spatial-verification toolkit | `spatial_verification.py`, `sver_c_wrapper.py`, `keypoint.py` (1 func), `linalg.py` (2 funcs), `distance.py` (3 funcs) + C++ `sver.cpp` |
| `wbia-tpl-pyflann/` | 1 package | FLANN Python bindings | `from pyflann import FLANN` (only `flann` KNN backend — parity-only, default is `exact`) |
| `wbia-tpl-pyhesaff/` | 1 package + C ext | Hessian-affine SIFT | `detect_feats_in_image()`, `get_hesaff_default_params()` |
| `wbia-utool/` | ∞ | Utility library | Used by vtool & pyhesaff internally (not by hotspotter directly) |
| `vendor/libflann_wb.so` | 1 binary | WBIA's libflann.so | Copied over pyflann's `.so` at build time |

## Dependency audit — what hotspotter actually calls

### `features.py` → pyhesaff

```python
hesaff_kwargs = _to_hesaff_kwargs(config, _h.get_hesaff_default_params())
keypoints, descriptors = _h.detect_feats_in_image(image, **hesaff_kwargs)
```

**Needs:** `pyhesaff/__init__.py` (exports), `pyhesaff/_pyhesaff.py` (main logic),
compiled C extension (.so). Does NOT need: ctypes_interface, ``__main__``, tests.

### `knn.py` → pyflann (parity-only)

```python
from pyflann import FLANN as _PyFlann
```

Only activated when `knn_backend="flann"` (not default). **Needs:** pyflann
bindings + `libflann.so`. Candidate for optional/extras dependency.

### `spatial.py` → vtool

```python
svtup = sver.spatially_verify_kpts(...)
```

The full call chain is:

```
sver.spatially_verify_kpts()           # spatial_verification.py:945
  → get_best_affine_inliers_()         # :929 → C++ sver_c_wrapper OR Python fallback
  → refine_inliers()                   # :857 → estimate_refined_transform → compute_homog (SVD)
```

**Needed from vtool:**

| File | What we use |
|---|---|
| `spatial_verification.py` | `spatially_verify_kpts`, `get_best_affine_inliers_`, `refine_inliers`, `estimate_refined_transform`, `compute_homog`, `get_affine_inliers`, `try_svd`, `apply_affine_to_points`, `svd` helper |
| `sver_c_wrapper.py` | `get_best_affine_inliers_cpp` (C extension) |
| `keypoint.py` | `get_xys(kpts)` — extracts (x,y) from keypoint array |
| `linalg.py` | `whiten_xy_points()`, `transform_points_with_homography()` |
| `distance.py` | `L2_sqrd()`, `det_distance()`, `ori_distance()` |
| `sver.cpp` + headers | C++ affine-inlier engine (OpenMP) |

**vtool files NOT used by hotspotter** (30 files, ~90% of the package):
`blend, chip, clustering2, confusion, coverage_grid, coverage_kpts, demodata,
depricated, ellipse, exif, features, fontdemo, geometry, _grave, histogram,
image, image_filters, image_shared, inspect_matches, matching,
nearest_neighbors, numpy_utils, _old_matching, other, patch, _pyflann_backend,
quality_classifier, _rhomb_dist, score_normalization, segmentation, symbolic,
trig, util_math, __main__.py`.

### utool — used internally by vtool & pyhesaff

Every vtool `.py` file (21 files) imports `utool`. Hotspotter does NOT import
utool directly. The critical runtime calls in `spatial_verification.py`:

| utool call | Purpose | Replace with |
|---|---|---|
| `ut.get_argflag('--no-c')` | Disable C extension at CLI | Env var or constant |
| `ut.get_argflag('--verb-sver')` | Debug verbosity | Env var |
| `ut.VERBOSE` | Global debug flag | Module constant |
| `ut.printex(ex, msg)` | Exception-print helper | `print(msg, exc_info=True)` or `logging.exception` |
| `ut.get_argval(...)` | CLI arg parsing (main guard) | Only in `if __name__ == '__main__'` — drop the guard |
| `ut.hashstr`, `ut.show_if_requested` | Doctest helpers | Drop (doctests are legacy WBIA) |
| `ut.argparse_dict` | Doctest only | Drop |

**Bottom line:** utool is a ~5-function shim for vtool's runtime. We can inline
these or replace with stdlib calls. pyhesaff's `_pyhesaff.py` imports utool for
verbose-debug output — same pattern, replaceable.

## Extraction strategy

For each vendored dep, extract the minimal files into `src/hotspotter/_vendor/`:

```
src/hotspotter/_vendor/
  __init__.py                      # re-exports for internal use
  sver/
    __init__.py                    # wraps spatially_verify_kpts
    _spatial_verification.py       # from vtool/spatial_verification.py
    _sver_c_wrapper.py             # from vtool/sver_c_wrapper.py
    _keypoint.py                   # from vtool/keypoint.py (get_xys only)
    _linalg.py                     # from vtool/linalg.py (2 funcs)
    _distance.py                   # from vtool/distance.py (3 funcs)
    _sver_cpp/                     # C++ source for affine inliers
  pyhesaff/
    __init__.py                    # from pyhesaff/__init__.py
    _pyhesaff.py                   # from pyhesaff/_pyhesaff.py
  pyflann/                         # from wbia-tpl-pyflann (optional, parity-only)
    __init__.py
    ...
```

### Step 0: Document the FIXME

`spatial_verification.py:964` has a WBIA comment: `FIXME: there is a
non-determenism here`. This is the OpenMP `>=` tie-breaking in `sver.cpp` that
we debugged. Fix it as part of the extraction (change `>=` to `>` or add
index-based tie-breaking) — then remove the FIXME.

### Step 1: Strip vtool → `_vendor/sver/`

1. Copy `spatial_verification.py` → `_vendor/sver/_spatial_verification.py`.
2. Copy `sver_c_wrapper.py` → `_vendor/sver/_sver_c_wrapper.py`.
3. Extract only the used functions from `keypoint.py` → `_vendor/sver/_keypoint.py` (just `get_xys` + its deps).
4. Extract only the used functions from `linalg.py` → `_vendor/sver/_linalg.py` (`whiten_xy_points`, `transform_points_with_homography` + their deps).
5. Extract only the used functions from `distance.py` → `_vendor/sver/_distance.py` (`L2_sqrd`, `det_distance`, `ori_distance`).
6. Copy C++ source (`sver.cpp`, headers, build files) → `_vendor/sver/_sver_cpp/`.
7. Strip utool imports from all copied files (replace with stdlib/inline stubs).
8. Update `source:hotspotter/spatial.py` imports from `from vtool import sver` → `from hotspotter._vendor.sver import _spatial_verification as sver` (or keep the `sver` alias).
9. Update Dockerfile: build `_vendor/sver/_sver_cpp/` in place of `wbia-vtool/`.
10. Remove `wbia-vtool/` submodule.

### Step 2: Strip pyhesaff → `_vendor/pyhesaff/`

1. Copy `pyhesaff/__init__.py` → `_vendor/pyhesaff/__init__.py`.
2. Copy `pyhesaff/_pyhesaff.py` → `_vendor/pyhesaff/_pyhesaff.py`.
3. Strip utool imports (replace with stdlib debug/verbose helpers).
4. Update Dockerfile: build `_vendor/pyhesaff/` C extension in place of `wbia-tpl-pyhesaff/`.
5. Update `source:hotspotter/features.py`: `import pyhesaff` → `from hotspotter._vendor import pyhesaff` (or keep `import pyhesaff` by putting it on the path).
6. Remove `wbia-tpl-pyhesaff/` submodule.

### Step 3: Strip pyflann → `_vendor/pyflann/` (optional)

1. pyflann is only needed for `knn_backend="flann"` (parity comparison).
2. Either: (a) bring it into `_vendor/pyflann/` directly, or (b) make it a
   conditional import (fail gracefully with `ImportError` if not installed —
   already the case in `knn.py`).
3. `libflann.so` → `_vendor/pyflann/lib/libflann.so`.
4. Remove `wbia-tpl-pyflann/` submodule and `vendor/libflann_wb.so`.

### Step 4: Eliminate utool entirely

After Steps 1-2, utool is no longer imported by any file in
`src/hotspotter/` or `_vendor/`. Replacements:

| Original utool call | Replacement |
|---|---|
| `ut.get_argflag('--no-c')` | `os.environ.get("HS_SVER_NO_C", "") == "1"` |
| `ut.get_argflag('--verb-sver')` | `os.environ.get("HS_SVER_VERBOSE", "") == "1"` |
| `ut.VERBOSE` | Module-level `_VERBOSE = bool(os.environ.get("HS_VERBOSE"))` |
| `ut.printex(ex, msg)` | `import traceback; print(msg); traceback.print_exc()` |
| `ut.hashstr(...)` | Was only in doctests — drop |
| `ut.show_if_requested()` | Was only in doctests — drop |
| `ut.argparse_dict(...)` | Was only in doctests — drop |
| `ut.get_argval(...)` | Was only in `if __name__ == '__main__'` guard — drop the guard |

5. Remove `wbia-utool/` submodule.

### Step 5: Update Dockerfile

After all extractions, the Dockerfile simplifies dramatically:

```dockerfile
# ... base image + system deps unchanged ...

COPY . /app
WORKDIR /app

# Build the extracted C++ extensions (sver + pyhesaff)
RUN cd src/hotspotter/_vendor/sver/_sver_cpp \
    && <build commands> \
    && cd ../../pyhesaff \
    && python3 setup.py build_ext --inplace

# Install pure-Python dependencies
RUN pip3 install --no-cache-dir \
    'numpy>=1.24,<2' opencv-contrib-python-headless==4.7.0.72 \
    pydantic scipy pandas pyarrow Pillow faiss-cpu

# No more vendored submodules to install
```

### Step 6: Validation

After each step, confirm:
1. `make test-unit` — 60+ tests pass.
2. `make test-replay` — golden replay bit-exact (pre-SV stages match).
3. `make test-parity` — HS vs WBIA correlation unchanged.
4. `scripts/run_fixture.py` produces identical parquets to the committed golden.

## Risk mitigation

- **Each step is independent** — strip one dep, validate, commit, then the next.
- **Golden replay is the safety net** — if a stripped file changes ANY pre-SV
  output bit, the test screams.
- **git history preserves the old code** — vendored repos remain in history,
  recoverable.
- **pyflann can be deferred or dropped** — `knn_backend="exact"` is the
  default; flann is parity-only and can stay optional.

## Execution order

```
Step 0  — Fix the sver.cpp FIXME (>= → >, remove non-determinism)                  [1 file, rebuild vtool]
Step 1  — Strip vtool → _vendor/sver/                                                [~6 files, add C build]
Step 2  — Strip utool from vtool (replace utool calls in copied files)               [inline edits]
Step 3  — Strip pyhesaff → _vendor/pyhesaff/                                          [~2 files + C ext build]
Step 4  — Strip utool from pyhesaff                                                   [inline edits]
Step 5  — Test: golden replay + unit suite → must be green                           [validation]
Step 6  — Remove wbia-utool/, wbia-vtool/, wbia-tpl-pyhesaff/ submodules             [cleanup]
Step 7  — (Optional) Strip pyflann → _vendor/pyflann/ or make truly optional         [1 package]
Step 8  — Simplify Dockerfile                                                         [1 file]
Step 9  — Final validation: full test suite                                          [validation]
```

## Size impact

| Asset | Before | After |
|---|---|---|
| `wbia-vtool/` (Python) | 41 files, ~15K lines | `_vendor/sver/` 6 files, ~2K lines |
| `wbia-tpl-pyhesaff/` | 1 package + C ext | `_vendor/pyhesaff/` 2 files + C ext |
| `wbia-tpl-pyflann/` | 1 package + .so | Optional / `_vendor/pyflann/` |
| `wbia-utool/` | 1 utility package | Deleted (inlined) |
| Dockerfile | 84 lines, 4 vendored builds | ~25 lines, 2 C-ext builds |
