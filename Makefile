.PHONY: build test test-unit test-benchmark test-benchmark-runner test-replay test-replay-live test-all run shell clean

IMAGE := wbia-core:latest
TEST_DATASET := $(CURDIR)/tests/test-dataset

# ---- Build ----
build:
	docker build -t $(IMAGE) .

# ---- Unit tests (38 tests, <2s, self-contained) ----
test: test-unit
test-unit:
	docker run --rm --entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/ -q --ignore=tests/benchmark --ignore=tests/replay"

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

# ---- Replay tests (84 tests, self-contained, NPZ fixtures in image) ----
test-replay:
	docker run --rm --entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/replay/ -v -k 'not TestLiveWbiaComparison'"

# Live WBIA comparison (1 test, needs WBIA on localhost:5000 + Docker socket)
test-replay-live:
	docker run --rm \
		-v /var/run/docker.sock:/var/run/docker.sock \
		--network host \
		-e WBIA_URL=http://localhost:5000 \
		--entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/replay/ -v -k 'TestLiveWbiaComparison'"

# ---- All pytest tests ----
test-all:
	docker run --rm \
		-v $(TEST_DATASET):/app/tests/test-dataset \
		--entrypoint bash $(IMAGE) -c \
		"pip install pytest -q && python -m pytest tests/ -v --ignore=tests/benchmark/test_runner.py -k 'not TestLiveWbiaComparison'"

# ---- Run sidecar server ----
run: server
server:
	docker run --rm -d --name wbia-core -p 5000:5000 $(IMAGE)

# ---- Shell in container ----
shell:
	docker run --rm -it \
		-v $(TEST_DATASET):/app/tests/test-dataset \
		--entrypoint bash $(IMAGE)

# ---- Clean ----
clean:
	-docker rm -f wbia-core 2>/dev/null
