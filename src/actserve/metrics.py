from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .types import RequestOutcome, ResultStatus


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
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
class MetricsSnapshot:
    submitted: int
    dispatched: int
    batches: int
    completed_on_time: int
    deadline_missed: int
    expired: int
    unserviceable: int
    replaced: int
    out_of_order: int
    overloaded: int
    failed: int
    cancelled: int
    mean_batch_size: float
    queue_p50_ms: float | None
    queue_p95_ms: float | None
    e2e_p50_ms: float | None
    e2e_p95_ms: float | None
    e2e_p99_ms: float | None

    @property
    def useful_actions(self) -> int:
        return self.completed_on_time

    @property
    def useful_action_rate(self) -> float:
        return 0.0 if self.submitted == 0 else self.completed_on_time / self.submitted

    @property
    def deadline_miss_rate(self) -> float:
        attempted = self.completed_on_time + self.deadline_missed
        return 0.0 if attempted == 0 else self.deadline_missed / attempted

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "submitted": self.submitted,
            "dispatched": self.dispatched,
            "batches": self.batches,
            "completed_on_time": self.completed_on_time,
            "deadline_missed": self.deadline_missed,
            "expired": self.expired,
            "unserviceable": self.unserviceable,
            "replaced": self.replaced,
            "out_of_order": self.out_of_order,
            "overloaded": self.overloaded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "mean_batch_size": self.mean_batch_size,
            "useful_action_rate": self.useful_action_rate,
            "deadline_miss_rate": self.deadline_miss_rate,
            "queue_p50_ms": self.queue_p50_ms,
            "queue_p95_ms": self.queue_p95_ms,
            "e2e_p50_ms": self.e2e_p50_ms,
            "e2e_p95_ms": self.e2e_p95_ms,
            "e2e_p99_ms": self.e2e_p99_ms,
        }


class SchedulerMetrics:
    def __init__(self, *, sample_limit: int = 100_000) -> None:
        self.submitted = 0
        self.dispatched = 0
        self.batches = 0
        self.batch_items = 0
        self._status = {status: 0 for status in ResultStatus}
        self._queue_ms: list[float] = []
        self._e2e_ms: list[float] = []
        self._sample_limit = sample_limit

    def record_submit(self) -> None:
        self.submitted += 1

    def record_dispatch(self, batch_size: int) -> None:
        self.dispatched += batch_size
        self.batches += 1
        self.batch_items += batch_size

    def record_outcome(self, outcome: RequestOutcome) -> None:
        self._status[outcome.status] += 1
        dispatched_statuses = {
            ResultStatus.COMPLETED,
            ResultStatus.DEADLINE_MISSED,
            ResultStatus.FAILED,
        }
        if outcome.status in dispatched_statuses and len(self._e2e_ms) < self._sample_limit:
            self._e2e_ms.append(outcome.end_to_end_ms)
        queue_ms = outcome.queue_ms
        if queue_ms is not None and len(self._queue_ms) < self._sample_limit:
            self._queue_ms.append(queue_ms)

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            submitted=self.submitted,
            dispatched=self.dispatched,
            batches=self.batches,
            completed_on_time=self._status[ResultStatus.COMPLETED],
            deadline_missed=self._status[ResultStatus.DEADLINE_MISSED],
            expired=self._status[ResultStatus.EXPIRED],
            unserviceable=self._status[ResultStatus.UNSERVICEABLE],
            replaced=self._status[ResultStatus.REPLACED],
            out_of_order=self._status[ResultStatus.OUT_OF_ORDER],
            overloaded=self._status[ResultStatus.OVERLOADED],
            failed=self._status[ResultStatus.FAILED],
            cancelled=self._status[ResultStatus.CANCELLED],
            mean_batch_size=0.0 if self.batches == 0 else self.batch_items / self.batches,
            queue_p50_ms=_percentile(self._queue_ms, 0.50),
            queue_p95_ms=_percentile(self._queue_ms, 0.95),
            e2e_p50_ms=_percentile(self._e2e_ms, 0.50),
            e2e_p95_ms=_percentile(self._e2e_ms, 0.95),
            e2e_p99_ms=_percentile(self._e2e_ms, 0.99),
        )


def prometheus_text(snapshot: MetricsSnapshot) -> str:
    """Render dependency-free Prometheus exposition text."""

    values = snapshot.as_dict()
    counter_names = {
        "submitted",
        "dispatched",
        "batches",
        "completed_on_time",
        "deadline_missed",
        "expired",
        "unserviceable",
        "replaced",
        "out_of_order",
        "overloaded",
        "failed",
        "cancelled",
    }
    lines = []
    for name, value in values.items():
        if value is None:
            continue
        metric_name = f"actserve_{name}"
        metric_type = "counter" if name in counter_names else "gauge"
        lines.append(f"# TYPE {metric_name} {metric_type}")
        lines.append(f"{metric_name} {value}")
    return "\n".join(lines) + "\n"
