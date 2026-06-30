.PHONY: build test test-unit test-benchmark test-benchmark-runner test-parity test-all shell clean recreate-golden-traces

IMAGE := hotspotter:latest
TEST_DATASET := $(CURDIR)/tests/test-dataset
ORACLE_DIR ?= $(CURDIR)/../artifacts/wbia-oracle

# ---- Build ----
# Note: Docker layer caching can silently reuse stale COPY layers after
# Python source changes.  If K2/K6 traces show wrong neighbour columns
# (e.g. 5 instead of 3/7), rebuild with:
#     docker build --no-cache -t $(IMAGE) .
build:
	docker build --no-cache -t $(IMAGE) .

# ---- Unit tests (self-contained, no external mounts needed) ----
test: test-unit
test-unit:
	docker run --rm \
		--entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/ -q"

# ---- Benchmark pytest tests (inside container, needs dataset mount) ----
test-benchmark:
	docker run --rm \
		-v $(TEST_DATASET):/app/tests/test-dataset \
		--entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/benchmark/ -v --ignore=tests/benchmark/test_runner.py"

# Benchmark runner tests (start Docker containers — need socket)
test-benchmark-runner:
	docker run --rm \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v $(TEST_DATASET):/app/tests/test-dataset \
		--network host \
		--entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/benchmark/test_runner.py -v"


# ---- Parity result tests (compare final rankings against WBIA oracle) ----
test-parity-results:
	docker run --rm \
		-v $(CURDIR)/../artifacts/wbia-oracle:/artifacts/wbia-oracle \
		-v $(CURDIR)/../pipeline/tests/reference_batch.json:/app/pipeline/tests/reference_batch.json \
		-v $(CURDIR)/../pipeline/tests/assets/images:/app/pipeline/tests/assets/images \
		-e WBIA_ORACLE_DIR=/artifacts/wbia-oracle \
		-e WBIA_BATCH_PATH=/app/pipeline/tests/reference_batch.json \
		--entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/test_parity_results.py -v -m parity"

# ---- All pytest tests ----
test-all:
	docker run --rm \
		-v $(TEST_DATASET):/app/tests/test-dataset \
		--entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/ -v --ignore=tests/benchmark/test_runner.py -k 'not TestLiveWbiaComparison'"

# ---- Shell in container ----
shell:
	docker run --rm -it \
		-v $(TEST_DATASET):/app/tests/test-dataset \
		--entrypoint bash $(IMAGE)

# ---- Parity comparison against WBIA oracle ----
# Three-way: records WBIA:nightly + WBIA:latest, runs hotspotter,
#           compares all three pairs (apple-apple-orange).
# Phase 1-2: record two WBIA oracles with baseline config (sv_on_true)
# Phase 3:   compare WBIA:nightly vs WBIA:latest (reference parity)
# Phase 4:   compare WBIA:nightly vs hotspotter     (main parity gate)
# Phase 5:   compare WBIA:latest  vs hotspotter     (redundancy)
#
# Pre-requisite: the hotspotter image must be built:
#     make build
#
# Skip recording (use existing oracles):
#     make test-parity SKIP_RECORD=1
test-parity: build
	ORACLE_DIR=$(ORACLE_DIR) python3 scripts/run_parity.py $(if $(SKIP_RECORD),--skip-record)

recreate-golden-traces: build
	docker run --rm \
		-v $(CURDIR)/tests/assets/golden_traces:/app/tests/assets/golden_traces \
		--entrypoint bash $(IMAGE) -c \
		"python tests/generate_goldens.py"

# ---- Clean ----
clean:
	-docker rm -f hotspotter 2>/dev/null
