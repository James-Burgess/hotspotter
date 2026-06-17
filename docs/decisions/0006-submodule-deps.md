# ADR-0006: Submodule-source build for C++ dependencies

**Status:** Accepted  
**Date:** 2026-06-06  
**Author:** OpenCode (AI Agent)

## Context

`wbia-core` depends on `wbia-pyhesaff` which transitively depends on `wbia-vtool` and `wbia-utool`. The PyPI wheels for these packages are stale (last updated 2021) and cause runtime crashes:

- **SIGSEGV on `import vtool`** — `wbia-vtool` 4.0.2 wheel bundles pre-compiled OpenCV 2.4.5 shared libraries. Loading these alongside system OpenCV 4.x (from `libopencv-dev` or `opencv-contrib-python-headless`) causes incompatible symbol resolution.
- **SIGSEGV on `import pyhesaff`** — Same root cause via its transitive dep on `wbia-vtool`.
- **Version detection failure** — `setuptools-scm` cannot detect the version inside Docker containers where `.git` metadata is absent.

Building from source against the target system's `libopencv-dev` eliminates both issues.

## Decision

All three C++/native-extension dependencies are vendored as **git submodules** inside `wbia-core/`:

| Submodule | Repo | C++ extensions |
|---|---|---|
| `wbia-utool` | `github.com:WildMeOrg/wbia-utool.git` | None (pure Python) |
| `wbia-vtool` | `github.com:WildMeOrg/wbia-vtool.git` | `libsver.so` (spatial verification) |
| `wbia-tpl-pyhesaff` | `github.com:WildMeOrg/wbia-tpl-pyhesaff.git` | `libhesaff.so` (Hessian-affine SIFT) |

The Dockerfile builds them in dependency order with `pip install --no-deps`:

```dockerfile
COPY . /app
WORKDIR /app

RUN SETUPTOOLS_SCM_PRETEND_VERSION=4.0.6 \
        pip install --no-cache-dir --no-deps ./wbia-utool/ \
    && SETUPTOOLS_SCM_PRETEND_VERSION=4.0.3 \
        pip install --no-cache-dir --no-deps ./wbia-vtool/ \
    && SETUPTOOLS_SCM_PRETEND_VERSION=4.0.0 \
        pip install --no-cache-dir --no-deps ./wbia-tpl-pyhesaff/

RUN pip install --no-cache-dir . flask gunicorn
```

- **`--no-deps`** prevents pip from pulling stale PyPI wheels for dependencies that are already built from submodule source.
- **`SETUPTOOLS_SCM_PRETEND_VERSION`** provides a version string when `.git` is absent.
- **System OpenCV headers** (`libopencv-dev`) are installed via `apt-get` and used by cmake during the build.

## Rationale

- **Determinism**: C++ extensions compiled against the same OpenCV version as the host system. No more SIGSEGV.
- **Pinnability**: Submodules lock to specific commits. Upgrading is an explicit `git pull` in the submodule.
- **No network at build time**: Submodules are checked out during `git clone --recursive`. The Docker build does not `git clone` anything.
- **Clean layer**: The `COPY . /app` layer copies submodule sources; the `pip install --no-deps` runs compile for each. Both are cached separately.

## Consequences

### Positive
- `import vtool`, `import pyhesaff`, `import wbia_core` all succeed without SIGSEGV.
- All C++ extensions compiled against the same system OpenCV.
- Docker image reproducible — no dependency on PyPI wheel availability.
- Submodules are auditable as plain source in the repo.

### Negative
- Docker image is larger (~2 GB) due to build toolchain (cmake, gcc, libopencv-dev).
- Build time is longer (~4 min) due to C++ compilation.
- Submodule maintenance overhead — versions must be explicitly bumped.
- Multi-arch builds (ARM64, etc.) need the build toolchain on each platform.

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| PyPI wheels only | OpenCV 2.4 / 4.x conflict → SIGSEGV |
| `git clone` inside Dockerfile | Network dependency at build time, no commit pinning |
| Multi-stage build with wheels | Same OpenCV conflict in wheel build stage |
| Skip vtool entirely (pure Python spatial verify) | Loss of `spatial_verification` module; feature drift vs WBIA |
