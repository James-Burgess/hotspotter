# Hotspotter — Extracted WBIA Identification Pipeline

We are extracting Hotspotter from `../wildbook-ia` and verifying correctness
via oracle testing against `../artifacts/wbia-oracle/`.

**YOU HAVE TO RUN THE CODE IN THE DOCKER FILE. THERE ARE CRAZY DEPS SO RUN EVERYTHING IN DOCKER!**

## Quick commands

```bash
make build                          # Build Docker image
make test-unit                      # Run all unit tests (Docker)
make test-parity ORACLE=../artifacts/wbia-oracle/wildme-wbia-nightly-20260625-173226
                                    # Run full oracle parity comparison
make shell                          # Interactive shell in container
```

## Project structure

```
wbia-core/
  src/hotspotter/       # Pure Python pipeline
    pipeline.py          # Main identify() — KLNBNN → SV → name score
    scoring.py            # LNBNN weights, match-building, per-annot scoring
    name_scoring.py       # fmech/nsumech, csum, canonical alignment
    spatial.py            # RANSAC spatial verification
    config.py             # Pydantic config models
    data.py               # Match, ScoredMatch, FeatureSet, AnnotatedImage
    knn.py                # FLANN/Faiss index build & query
    trace.py              # Parquet trace writer
    chip.py               # Chip extraction (mask, resize)
    features.py            # Hessian-affine SIFT extraction (pyhesaff)
  tests/
    test_name_scoring.py  # Oracle integration + WBIA doctest parity
    test_scoring.py       # LNBNN, filter, match-building
    ...
  scripts/
    compare_to_wbia.py    # Runs all configs, calls compare_wbia_oracles.py
    run_fixture.py         # Hotspotter pipeline runner for a batch
  wbia-tpl-pyflann/       # Vendored pyflann (4.0.5.dev10) — submodule
  wbia-vtool/             # Vendored vtool (spatial_verification)

../artifacts/wbia-oracle/
  wildme-wbia-nightly-20260625-173226/  # Canonical WBIA oracle (26 stages)

../scripts/compare_wbia_oracles.py      # Compares two trace dumps, emits HTML/terminal
```

## Parity investigation → `deeseek-wbia-parity.md`


