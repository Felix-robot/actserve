from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


def _percentile(ordered: list[float], percentile: float) -> float | None:
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class ProfileMetric:
    name: str
    unit: str
    count: int
    total: float
    mean: float
    minimum: float
    maximum: float
    p50: float
    p95: float
    p99: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "unit": self.unit,
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    elapsed_ms: float
    metrics: tuple[ProfileMetric, ...]

    def get(self, name: str, unit: str | None = None) -> ProfileMetric | None:
        for metric in self.metrics:
            if metric.name == name and (unit is None or metric.unit == unit):
                return metric
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "actserve.profile.v1",
            "elapsed_ms": self.elapsed_ms,
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


class StageProfiler:
    """Bounded, thread-safe numeric profiler for serving and training stages.

    Observations are aggregated by ``(name, unit)``. Raw observations, model
    inputs, and actions are never retained, keeping snapshots safe to publish.
    """

    def __init__(self, *, sample_limit: int = 100_000) -> None:
        if sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        self.sample_limit = sample_limit
        self._started_ns = time.perf_counter_ns()
        self._samples: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def observe(self, name: str, value: float, *, unit: str) -> None:
        if not name or not unit:
            raise ValueError("profile metric name and unit must be non-empty")
        value = float(value)
        if not math.isfinite(value):
            return
        key = (name, unit)
        with self._lock:
            samples = self._samples[key]
            if len(samples) < self.sample_limit:
                samples.append(value)

    def duration(self, name: str, duration_ms: float) -> None:
        self.observe(name, duration_ms, unit="ms")

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.duration(name, (time.perf_counter_ns() - started_ns) / 1_000_000)

    def snapshot(self) -> ProfileSnapshot:
        with self._lock:
            copied = {key: list(values) for key, values in self._samples.items()}
        metrics = []
        for (name, unit), values in sorted(copied.items()):
            ordered = sorted(values)
            if not ordered:
                continue
            metrics.append(
                ProfileMetric(
                    name=name,
                    unit=unit,
                    count=len(ordered),
                    total=sum(ordered),
                    mean=sum(ordered) / len(ordered),
                    minimum=ordered[0],
                    maximum=ordered[-1],
                    p50=_percentile(ordered, 0.50) or 0.0,
                    p95=_percentile(ordered, 0.95) or 0.0,
                    p99=_percentile(ordered, 0.99) or 0.0,
                )
            )
        return ProfileSnapshot(
            elapsed_ms=(time.perf_counter_ns() - self._started_ns) / 1_000_000,
            metrics=tuple(metrics),
        )

