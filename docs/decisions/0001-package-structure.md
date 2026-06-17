# ADR-0001: Package Structure and Module Boundaries

**Status:** Accepted  
**Date:** 2026-06-04  
**Author:** OpenCode (AI Agent)

## Context

The HotSpotter algorithm in WBIA is currently embedded in a monolithic package (`wbia.algo.hots`) with 3000+ line files, deep IBEISController coupling, and no clear separation of concerns. We need to extract the core identification pipeline into a standalone, testable Python package.

## Decision

We adopt a **7-module boundary** corresponding to the pipeline stages:

```
wbia_core/
├── config.py      # Pydantic models for all algorithm parameters
├── data.py        # Data structures: FeatureSet, Match, ScoredMatch
├── features.py    # Feature extraction (pyhesaff SIFT)
├── knn.py         # k-NN search (faiss backend)
├── scoring.py     # LNBNN weighting, name aggregation, scoring
├── spatial.py     # Spatial verification (RANSAC homography)
└── pipeline.py    # Convenience function: identify()
```

Each module is **independent and stateless**.

## Rationale

- **Testability**: Each stage can be unit-tested in isolation.
- **Composability**: Researchers can swap individual stages without rewriting the whole pipeline.
- **Portability**: No imports from `wbia.*` anywhere in the public surface.
- **Determinism**: State = data in → data out; no global cache or mutable controller.

## Consequences

### Positive
- Clean public API: `from wbia_core import identify` or `from wbia_core.scoring import lnbnn_score`.
- No dependency on Wildbook database, PostgreSQL, or ZMQ.
- Can be tested in a notebook without a full Wildbook install.

### Negative
- Need to re-implement some utility functions that previously lived in `wbia.utool`.
- FLANN index building logic must be reimplemented for faiss.
- Original `chip_match.py` (3000 lines) is decomposed — some helper classes are dropped if they are WBIA-specific.

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Single `pipeline.py` file | Too large, hard to test, no composability |
| Keep original `wbia.algo.hots` structure | Deep controller coupling, untestable outside WBIA |
| Put everything in `__init__.py` | Namespace pollution, no clear module boundaries |

## References

- Original pipeline: `wildbook-ia/wbia/algo/hots/pipeline.py` (3000+ lines)
- Original chip match: `wildbook-ia/wbia/algo/hots/chip_match.py` (3000+ lines)
- Design doc: `wildbook-docs/docs/ml-modernization/wbia-core.md`
