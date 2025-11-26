"""Rolling metrics collection for the baseline system."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Tuple
import time


@dataclass
class _RequestRecord:
    timestamp: float
    latency: float


@dataclass
class _TokenRecord:
    timestamp: float
    count: int


class MetricsTracker:
    """Track latency, throughput, and token speed over a sliding time window."""

    def __init__(self, window_seconds: int = 60) -> None:
        self._window = window_seconds
        self._request_records: Deque[_RequestRecord] = deque()
        self._token_records: Deque[_TokenRecord] = deque()
        self._last_latency: float = 0.0
        self._last_tokens: int = 0
        self._lock = Lock()

    def mark_start(self) -> float:
        """Return a timestamp used to compute latency."""

        return time.perf_counter()

    def finalize_request(self, started_at: float, token_count: int) -> float:
        """Record request completion metrics and return latency."""

        latency = time.perf_counter() - started_at
        now = time.time()
        with self._lock:
            self._request_records.append(_RequestRecord(timestamp=now, latency=latency))
            self._token_records.append(_TokenRecord(timestamp=now, count=token_count))
            self._last_latency = latency
            self._last_tokens = token_count
            self._trim(now)
        return latency

    def snapshot(self) -> dict[str, float | int]:
        """Return a metrics snapshot for the frontend."""

        now = time.time()
        with self._lock:
            self._trim(now)
            request_count = len(self._request_records)
            token_count = sum(record.count for record in self._token_records)
            total_latency = sum(record.latency for record in self._request_records)
            window = self._window or 1
            tokens_per_sec = token_count / window
            requests_per_sec = request_count / window
            average_latency = total_latency / request_count if request_count else 0.0
            return {
                "tokens_per_sec": tokens_per_sec,
                "requests_per_sec": requests_per_sec,
                "average_latency": average_latency,
                "last_request_latency": self._last_latency,
                "last_request_tokens": self._last_tokens,
            }

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while self._request_records and self._request_records[0].timestamp < cutoff:
            self._request_records.popleft()
        while self._token_records and self._token_records[0].timestamp < cutoff:
            self._token_records.popleft()
