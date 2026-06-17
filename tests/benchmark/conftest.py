"""Pytest configuration — add the benchmark package to sys.path so that
``from coco.loader import ...`` and ``from targets.base import ...`` work
as they do when running ``python tests/benchmark/run_benchmark.py``."""

import sys
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
