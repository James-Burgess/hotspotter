"""Abstract base for target runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TargetConfig:
    name: str
    image: str
    port: int
    keep_containers: bool = False
    debug_logs_dir: str | None = None
    debug_log_file: str | None = None


@dataclass
class QueryResult:
    query_index: int
    annot_scores: list[dict] = field(default_factory=list)
    timing_ms: float = 0.0
    raw_response: dict | None = None
    error: str | None = None


class TargetRunner(ABC):
    def __init__(self, config: TargetConfig):
        self.config = config

    @abstractmethod
    def start(self) -> dict:
        """Start the container, wait for healthy. Return manifest info dict."""

    @abstractmethod
    def run_query(self, query_index: int, request_body: dict) -> QueryResult:
        """Run one identification query and return results."""

    @abstractmethod
    def stop(self) -> None:
        """Stop and remove the container."""
