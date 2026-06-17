# wbia-core Developer Documentation

This directory contains the technical documentation for the `wbia-core` package.

## Structure

- **[decisions/](decisions/)** — Architecture Decision Records (ADRs) documenting key design choices
- **[api/](api/)** — API documentation (generated from docstrings)
- **[development/](development/)** — Setup, testing, and contribution guides
- **[references/](references/)** — Links and notes on the original WBIA algorithm sources

## Quick Start

For package usage, see the [public API documentation](../docs/) in `wildbook-docs`.

## Decision Log

| Date | Decision | File |
|------|----------|------|
| 2026-06-04 | Package structure and module boundaries | [0001-package-structure.md](decisions/0001-package-structure.md) |
| 2026-06-04 | Pure flat Pydantic config replacing QueryParams | [0002-pydantic-config.md](decisions/0002-pydantic-config.md) |
| 2026-06-04 | pyhesaff SIFT for feature extraction | [0003-feature-extraction.md](decisions/0003-feature-extraction.md) |
| 2026-06-04 | faiss knn over FLANN | [0004-knn-backend.md](decisions/0004-knn-backend.md) |
| 2026-06-04 | Stateless pipeline, no IBEISController | [0005-stateless-pipeline.md](decisions/0005-stateless-pipeline.md) |
| 2026-06-06 | Submodule-source build for C++ deps | [0006-submodule-deps.md](decisions/0006-submodule-deps.md) |