# ADR-0005: Stateless Pipeline (No IBEISController)

**Status:** Accepted  
**Date:** 2026-06-04  
**Author:** OpenCode (AI Agent)

## Context

WBIA's HotSpotter pipeline is tightly coupled to `IBEISController`, a God object that coordinates with the database, depcache, ZMQ, and image loading. This makes the algorithm impossible to test or use outside the full WBIA stack.

## Decision

`wbia_core.pipeline` is **stateless**: all inputs are passed as arguments; all outputs are returned as values. No mutable state, no global cache, no database.

```python
def identify(
    query_image: np.ndarray,
    database: list[AnnotatedImage],
    config: IdentificationConfig = IdentificationConfig(),
) -> list[ScoredMatch]:
    """
    Stateless identification. Returns top-k matches withscores ordered by descending score.
    """
    # 1. Extract query features
    query_features = extract_features(query_image, config.hotspotter.sift)

    # 2. Build or use existing index
    index = _build_or_load_index(database, config.hotspotter)

    # 3. k-NN search
    distances, labels = query_index(index, query_features, config.hotspotter.knn)

    # 4. Scoring
    matches = score_matches(query_features, distances, labels, database, config.hotspotter)

    # 5. Spatial verification (optional)
    if config.hotspotter.sv_on:
        matches = spatial_verification(query_features, matches, database, config.hotspotter.sv)

    # 6. Return top N
    return sorted(matches, key=lambda m: m.score, reverse=True)[:config.hotspotter.num_return]
```

## Rationale

- **Determinism**: Same inputs → same outputs, every time. No hidden state.
- **Testability**: Can test the pipeline with synthetic `np.ndarray` images and mock `AnnotatedImage` objects.
- **Portability**: Works in a notebook, a test suite, or a Lambda function.
- **Composability**: Researchers can swap individual stages without subclassing a controller.

## Consequences

### Positive
- `wbia-core` has zero imports from `wbia.*`.
- Cache invalidation is the caller's problem (callers like `wildlife-id` handle it).
- No PostgreSQL connection, no SQLite depcache, no ZMQ.

### Negative
- Callers must manage the `database` (list of `AnnotatedImage`) in memory or on disk.
- No built-in lazy loading; all features must be pre-extracted before calling `identify()`.
- No built-in job queue; callers must parallelize themselves.

## Data Flow

```
Wildbook Java code
    → calls /wildlife-id/api/identify/
        → wildlife-id loads features from pgvector index
        → calls wbia_core.identify(query_image=..., database=...)
            → returns list[ScoredMatch]
        → wildlife-id returns JSON response
```

The `database` parameter is a list of `AnnotatedImage` objects, each containing:
- `annot_uuid` (`uuid.UUID`)
- `name_uuid` (`uuid.UUID`)
- `image` (`np.ndarray` or path)
- `features` (`FeatureSet`, pre-extracted)
- `bbox` (`tuple[int, int, int, int]` — `(x, y, w, h)`)

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Keep IBEISController as a facade | Defeats the purpose of modernization; still requires full WBIA stack |
| Lazy loading inside pipeline | Complicates caching and invalidation; caller should handle it |
| SQLAlchemy models | Adds ORM dependency; caller knows the storage layer, not the algorithm |

## References

- Original pipeline: `wildbook-ia/wbia/algo/hots/pipeline.py` (3000 lines)
- Original controller: `wildbook-ia/wbia/control/IBEISControl.py` (1283 lines)
- Design doc: `wildbook-docs/docs/ml-modernization/wbia-core.md#stateless-pipelinemanship`
