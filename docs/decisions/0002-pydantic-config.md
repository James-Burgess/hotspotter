# ADR-0002: Flat Pydantic Config Replacing QueryParams

**Status:** Accepted  
**Date:** 2026-06-04  
**Author:** OpenCode (AI Agent)

## Context

WBIA's configuration is deeply nested in `QueryParams` and `Config` objects with dozens of interdependent attributes. This makes it impossible to validate a configuration without instantiating the entire WBIA runtime.

## Decision

Replace the nested config with **flat Pydantic models** in `wbia_core.config`.

```python
class HotSpotterConfig(BaseModel):
    knn: int = 7
    checks: int = 768
    method: Literal["faiss"] = "faiss"
    normalizer_rule: Literal["lnbnn"] = "lnbnn"
    score_method: Literal["nsum", "csum"] = "nsum"
    spatial_verification: bool = True
    sv_on: bool = True
    prescore_method: Literal["nsum", "csum"] = "nsum"
    num_return: int = 10
    fg_on: bool = True
    ratio_thresh: float = 1.618
    lnbnn_ratio: float = 1.0
    ln_ratio: float = 0.8

class IdentificationConfig(BaseModel):
    pipeline: Literal["HotSpotter", "MiewId", "CurvRank", "Deepsqueak"] = "HotSpotter"
    hotspotter: Optional[HotSpotterConfig] = Field(default_factory=HotSpotterConfig)
    # ... per-algorithm sub-configs
```

## Rationale

- **Validation**: Pydantic validates types and ranges at import time.
- **Serialization**: Configs can be dumped to/from JSON for reproducibility.
- **Documentation**: Each field is self-documenting via `Field(description=...)`.
- **UX**: Algorithm authors can write `my_config = HotSpotterConfig(knn=5)` instead of manipulating a nested dict.

## Consequences

### Positive
- No more `Config` class with 50+ attributes.
- IDE autocomplete works out of the box.
- Can generate OpenAPI schemas for the REST API automatically.

### Negative
- Need to maintain mapping from old `QueryParams` attributes to new field names.
- Some edge-case flag combinations may be lost if they were not exercised in tests.

## Mapping from Legacy Config

| Legacy Attribute | New Field | Notes |
|---|---|---|
| `query_params.K` | `knn` | Number of nearest neighbors |
| `query_params.Knorm` | — | Removed; always `knn + 1` |
| `query_params.score_method` | `score_method` | `"nsum"` or `"csum"` |
| `query_params.prescore_method` | `prescore_method` | Applied before SV |
| `query_params.sv_on` | `sv_on` | Spatial verification flag |
| `query_params.ratio_thresh` | `ratio_thresh` | Ratio test threshold |
| `query_params.fg_name` | `fg_on` | Foreground flag (bool) |
| `query_params.method` | `method` | Always `"faiss"` |

## References

- Original config: `wildbook-ia/wbia/algo/hots/query_request.py`
- Pydantic docs: https://docs.pydantic.dev/latest/
