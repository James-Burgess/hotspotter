# Hotspotter Developer Documentation

This directory contains technical documentation for the `hotspotter` package
(formerly `wbia-core`) — a reusable HotSpotter algorithm library. Service APIs,
indexes, persistence, and production identification orchestration belong in
`wildlife-id`.

## Structure

- **[decisions/](decisions/)** — Architecture Decision Records (ADRs) documenting key design choices
- **[development/](development/)** — Transition notes, testing guide, parity analysis, discrepancy tracking
- **[logs/](logs/)** — Session logs and investigation notes

## Key Docs

| Doc | Content |
|---|---|
| `development/hotspotter-transition.md` | Source-of-truth checklist: completed + remaining work |
| `development/hotspotter-parity-discrepancies.md` | Measured metrics vs WBIA, root causes, Phase 2 priority |
| `development/phase-1-baseline.md` | Test baseline before Phase 1 rename |
| `development/testing-guide.md` | Test layers, commands, Makefile targets |
| `development/hotspotter_permutations.md` | Per-book WBIA HotSpotter parameter survey |
| `development/wildbook-10.10.2-ia-usage.md` | Deployed Wildbook branch WBIA usage/config drift |
| `development/wildbook-11-rebuild-strategy.md` | Rebuild/reprocess strategy for Wildbook 11 |
| `wbia-oracle-recording.md` | WBIA oracle recording + comparison workflow |

## Quick Start

```bash
# Build
make build

# Unit tests (38)
make test-unit

# Replay tests (84)
make test-replay

# Parity check vs WBIA oracle
make test-parity ORACLE=../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-144646
```

## Decision Log

| Date | Decision | File |
|---|---|---|
| 2026-06-04 | Package structure and module boundaries | [0001-package-structure.md](decisions/0001-package-structure.md) |
| 2026-06-04 | Pure flat Pydantic config replacing QueryParams | [0002-pydantic-config.md](decisions/0002-pydantic-config.md) |
| 2026-06-04 | pyhesaff SIFT for feature extraction | [0003-feature-extraction.md](decisions/0003-feature-extraction.md) |
| 2026-06-04 | pyflann over faiss as primary backend | [0004-knn-backend.md](decisions/0004-knn-backend.md) |
| 2026-06-04 | Stateless pipeline, no IBEISController | [0005-stateless-pipeline.md](decisions/0005-stateless-pipeline.md) |
| 2026-06-06 | Submodule-source build for C++ deps | [0006-submodule-deps.md](decisions/0006-submodule-deps.md) |
