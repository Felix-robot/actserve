from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .profiler import StageProfiler, _percentile

STANDARD_PHASES = ("data_wait", "forward", "backward", "optimizer", "checkpoint")


@dataclass(frozen=True, slots=True)
class TrainingPhaseMetric:
    name: str
    count: int
    total_ms: float
    mean_ms: float
    p95_ms: float
    share: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "p95_ms": self.p95_ms,
            "share": self.share,
        }


@dataclass(frozen=True, slots=True)
class TrainingProfileSnapshot:
    steps: int
    samples: int
    total_step_ms: float
    mean_step_ms: float
    samples_per_second: float
    bottleneck: str | None
    phases: tuple[TrainingPhaseMetric, ...]
    recommendations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "actserve.training_profile.v1",
            "scope": "timing analysis only; recommendations require workload validation",
            "steps": self.steps,
            "samples": self.samples,
            "total_step_ms": self.total_step_ms,
            "mean_step_ms": self.mean_step_ms,
            "samples_per_second": self.samples_per_second,
            "bottleneck": self.bottleneck,
            "phases": [phase.as_dict() for phase in self.phases],
            "recommendations": list(self.recommendations),
        }


class TrainingStep:
    def __init__(self, profiler: TrainingProfiler, samples: int) -> None:
        self._profiler = profiler
        self._samples = samples
        self._started_ns = time.perf_counter_ns()
        self._phases_ms: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            self._phases_ms[name] = self._phases_ms.get(name, 0.0) + duration_ms

    def record(self, name: str, duration_ms: float) -> None:
        """Record an externally measured phase duration for this step."""

        duration_ms = self._profiler._validate_duration(name, duration_ms)
        self._phases_ms[name] = self._phases_ms.get(name, 0.0) + duration_ms

    def finish(self) -> None:
        step_ms = (time.perf_counter_ns() - self._started_ns) / 1_000_000
        self._profiler.record_step(self._phases_ms, samples=self._samples, step_ms=step_ms)


class TrainingProfiler:
    """Dependency-free training phase profiler that never retains batch data."""

    def __init__(self, *, sample_limit: int = 100_000) -> None:
        self._profiler = StageProfiler(sample_limit=sample_limit)
        self._steps = 0
        self._samples = 0
        self._phase_samples: dict[str, list[float]] = {}
        self._step_samples: list[float] = []
        self._sample_limit = sample_limit

    @contextmanager
    def step(self, *, samples: int = 1) -> Iterator[TrainingStep]:
        if samples < 0:
            raise ValueError("samples must be non-negative")
        step = TrainingStep(self, samples)
        try:
            yield step
        finally:
            step.finish()

    def record_step(
        self,
        phases_ms: Mapping[str, float],
        *,
        samples: int = 1,
        step_ms: float | None = None,
    ) -> None:
        if samples < 0:
            raise ValueError("samples must be non-negative")
        validated = {
            name: self._validate_duration(name, duration)
            for name, duration in phases_ms.items()
        }
        if step_ms is None:
            step_ms = sum(validated.values())
        step_ms = self._validate_duration("step", step_ms)

        self._steps += 1
        self._samples += samples
        if len(self._step_samples) < self._sample_limit:
            self._step_samples.append(step_ms)
        self._profiler.duration("training.step", step_ms)
        for name, duration in validated.items():
            phase_samples = self._phase_samples.setdefault(name, [])
            if len(phase_samples) < self._sample_limit:
                phase_samples.append(duration)
            self._profiler.duration(f"training.{name}", duration)

    def snapshot(self) -> TrainingProfileSnapshot:
        phase_total_ms = sum(sum(values) for values in self._phase_samples.values())
        phases = []
        for name, values in sorted(self._phase_samples.items()):
            ordered = sorted(values)
            total_ms = sum(ordered)
            phases.append(
                TrainingPhaseMetric(
                    name=name,
                    count=len(ordered),
                    total_ms=total_ms,
                    mean_ms=total_ms / len(ordered),
                    p95_ms=_percentile(ordered, 0.95) or 0.0,
                    share=total_ms / phase_total_ms if phase_total_ms else 0.0,
                )
            )
        total_step_ms = sum(self._step_samples)
        bottleneck = max(phases, key=lambda phase: phase.total_ms).name if phases else None
        return TrainingProfileSnapshot(
            steps=self._steps,
            samples=self._samples,
            total_step_ms=total_step_ms,
            mean_step_ms=total_step_ms / self._steps if self._steps else 0.0,
            samples_per_second=(
                self._samples / (total_step_ms / 1000) if total_step_ms else 0.0
            ),
            bottleneck=bottleneck,
            phases=tuple(phases),
            recommendations=self._recommend(phases),
        )

    @staticmethod
    def _validate_duration(name: str, duration_ms: float) -> float:
        if not name:
            raise ValueError("phase name must be non-empty")
        duration_ms = float(duration_ms)
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("phase durations must be finite and non-negative")
        return duration_ms

    @staticmethod
    def _recommend(phases: list[TrainingPhaseMetric]) -> tuple[str, ...]:
        by_name = {phase.name: phase for phase in phases}
        recommendations = []
        if by_name.get("data_wait") and by_name["data_wait"].share >= 0.15:
            recommendations.append(
                "data_wait is material; validate worker count, prefetching, pinned memory, "
                "and preprocessing cache"
            )
        if by_name.get("checkpoint") and by_name["checkpoint"].share >= 0.10:
            recommendations.append(
                "checkpoint time is material; validate asynchronous writes or a longer interval"
            )
        compute_share = sum(
            by_name[name].share for name in ("forward", "backward") if name in by_name
        )
        if compute_share >= 0.70:
            recommendations.append(
                "forward/backward dominate; benchmark mixed precision, compile, and batch size"
            )
        if by_name.get("optimizer") and by_name["optimizer"].share >= 0.15:
            recommendations.append(
                "optimizer time is material; benchmark fused optimizer implementations"
            )
        return tuple(recommendations)
