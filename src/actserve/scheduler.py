from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Literal

from .backend import InferenceBackend
from .metrics import SchedulerMetrics
from .profiler import StageProfiler
from .types import InferenceRequest, RequestOutcome, ResultStatus


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    policy: Literal["edf", "fifo"] = "edf"
    max_batch_wait_ms: float = 2.0
    dispatch_guard_ms: float = 1.0
    max_batch_size: int | None = None
    coalesce_sessions: bool = True
    enforce_monotonic_sequence: bool = True
    drop_missed_actions: bool = True
    drop_unserviceable_requests: bool = False
    max_pending_requests: int | None = None

    def __post_init__(self) -> None:
        if self.max_batch_wait_ms < 0 or self.dispatch_guard_ms < 0:
            raise ValueError("scheduler timings must be non-negative")
        if self.max_batch_size is not None and self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.max_pending_requests is not None and self.max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least 1")


@dataclass(order=True, slots=True)
class _Envelope:
    sort_key: tuple[int, int, int]
    order: int
    request: InferenceRequest = field(compare=False)
    future: asyncio.Future[RequestOutcome] = field(compare=False)
    active: bool = field(default=True, compare=False)
    dispatched_ns: int | None = field(default=None, compare=False)


class Scheduler:
    """Deadline-aware, session-coalescing inference scheduler."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: SchedulerConfig | None = None,
        *,
        metrics: SchedulerMetrics | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or SchedulerConfig()
        self.metrics = metrics or SchedulerMetrics()
        self.profiler = profiler
        self._condition = asyncio.Condition()
        self._heap: list[_Envelope] = []
        self._pending_by_session: dict[tuple[str, str], _Envelope] = {}
        self._latest_sequence_by_session: dict[tuple[str, str], int] = {}
        self._counter = itertools.count()
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False

    async def __aenter__(self) -> Scheduler:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @property
    def is_accepting(self) -> bool:
        """Whether the scheduler is running and accepting new requests."""

        return self._accepting and self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        if self._worker is not None:
            return
        self._accepting = True
        self._worker = asyncio.create_task(self._run(), name="actserve-scheduler")

    async def submit(self, request: InferenceRequest) -> RequestOutcome:
        future = await self.enqueue(request)
        return await future

    async def enqueue(self, request: InferenceRequest) -> asyncio.Future[RequestOutcome]:
        if not self._accepting:
            raise RuntimeError("scheduler is not accepting requests; call start() first")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RequestOutcome] = loop.create_future()
        order = next(self._counter)
        if self.config.policy == "edf":
            sort_key = (request.deadline_ns, -request.priority, order)
        else:
            sort_key = (request.received_ns, -request.priority, order)
        envelope = _Envelope(sort_key=sort_key, order=order, request=request, future=future)

        async with self._condition:
            session_key = (request.model, request.session_id)
            latest_sequence = self._latest_sequence_by_session.get(session_key)
            self.metrics.record_submit()
            if (
                self.config.enforce_monotonic_sequence
                and latest_sequence is not None
                and request.sequence_no <= latest_sequence
            ):
                envelope.active = False
                self._finish(envelope, ResultStatus.OUT_OF_ORDER)
                return future
            previous = None
            if self.config.coalesce_sessions:
                candidate = self._pending_by_session.get(session_key)
                if candidate is not None and candidate.active and candidate.dispatched_ns is None:
                    previous = candidate

            self._remove_inactive_locked()
            self._expire_requests_locked()
            at_capacity = (
                self.config.max_pending_requests is not None
                and len(self._heap) >= self.config.max_pending_requests
            )
            if previous is None and at_capacity:
                envelope.active = False
                self._finish(
                    envelope,
                    ResultStatus.OVERLOADED,
                    error="scheduler pending-request capacity is exhausted",
                )
                return future

            self._latest_sequence_by_session[session_key] = request.sequence_no
            if self.config.coalesce_sessions:
                if previous is not None:
                    previous.active = False
                    self._finish(
                        previous,
                        ResultStatus.REPLACED,
                        replaced_by_request_id=request.request_id,
                    )
                self._pending_by_session[session_key] = envelope
            heapq.heappush(self._heap, envelope)
            self._condition.notify()
        return future

    async def close(self, *, cancel_pending: bool = False) -> None:
        worker = self._worker
        if worker is None:
            return
        async with self._condition:
            self._accepting = False
            if cancel_pending:
                for envelope in self._heap:
                    if envelope.active:
                        envelope.active = False
                        self._finish(envelope, ResultStatus.CANCELLED)
                self._heap.clear()
                self._pending_by_session.clear()
            self._condition.notify_all()
        await worker
        self._worker = None

    async def _run(self) -> None:
        while True:
            batch: list[_Envelope] | None = None
            async with self._condition:
                while batch is None:
                    self._remove_inactive_locked()
                    self._expire_requests_locked()
                    self._drop_unserviceable_requests_locked()
                    if not self._heap:
                        if not self._accepting:
                            return
                        await self._condition.wait()
                        continue

                    leader = self._heap[0]
                    now = time.monotonic_ns()
                    batch_window_ns = int(self.config.max_batch_wait_ms * 1_000_000)
                    guard_ns = int(self.config.dispatch_guard_ms * 1_000_000)
                    predicted_ns = self._predicted_latency_ns(1) or 0
                    dispatch_at = min(
                        leader.request.received_ns + batch_window_ns,
                        leader.request.deadline_ns - guard_ns - predicted_ns,
                    )
                    if now < dispatch_at:
                        try:
                            await asyncio.wait_for(
                                self._condition.wait(), timeout=(dispatch_at - now) / 1_000_000_000
                            )
                        except asyncio.TimeoutError:
                            pass
                        continue
                    batch = self._take_batch_locked()

            if batch:
                await self._execute(batch)

    def _take_batch_locked(self) -> list[_Envelope]:
        leader = heapq.heappop(self._heap)
        leader.active = False
        leader.dispatched_ns = time.monotonic_ns()
        self._clear_session_pointer(leader)
        selected = [leader]
        batch_key = self.backend.batch_key(leader.request)
        configured_max = self.config.max_batch_size or self.backend.max_batch_size
        limit = min(configured_max, self.backend.max_batch_size)
        guard_ns = int(self.config.dispatch_guard_ms * 1_000_000)

        remaining: list[_Envelope] = []
        while self._heap:
            envelope = heapq.heappop(self._heap)
            if not envelope.active:
                continue
            compatible = self.backend.batch_key(envelope.request) == batch_key
            projected_size = len(selected) + 1
            predicted_ns = self._predicted_latency_ns(projected_size)
            earliest_deadline = min(
                item.request.deadline_ns for item in [*selected, envelope]
            )
            admission_safe = (
                predicted_ns is None
                or time.monotonic_ns() + predicted_ns + guard_ns <= earliest_deadline
            )
            if len(selected) < limit and compatible and admission_safe:
                envelope.active = False
                envelope.dispatched_ns = time.monotonic_ns()
                self._clear_session_pointer(envelope)
                selected.append(envelope)
            else:
                remaining.append(envelope)
        for envelope in remaining:
            heapq.heappush(self._heap, envelope)
        self.metrics.record_dispatch(len(selected))
        if self.profiler is not None:
            self.profiler.observe("scheduler.batch_size", len(selected), unit="items")
        return selected

    def _predicted_latency_ns(self, batch_size: int) -> int | None:
        estimator = getattr(self.backend, "estimate_batch_latency_ms", None)
        if estimator is None:
            return None
        estimate_ms = estimator(batch_size)
        if estimate_ms is None:
            return None
        if estimate_ms < 0:
            raise ValueError("backend latency estimate must be non-negative")
        return int(estimate_ms * 1_000_000)

    async def _execute(self, batch: list[_Envelope]) -> None:
        requests = [envelope.request for envelope in batch]
        started_ns = time.perf_counter_ns()
        try:
            actions = list(await self.backend.infer_batch(requests))
            if self.profiler is not None:
                self.profiler.duration(
                    "backend.infer",
                    (time.perf_counter_ns() - started_ns) / 1_000_000,
                )
            if len(actions) != len(batch):
                raise RuntimeError(
                    f"backend returned {len(actions)} actions for a batch of {len(batch)} requests"
                )
            for envelope, action in zip(batch, actions, strict=True):
                request = envelope.request
                if (
                    action.request_id != request.request_id
                    or action.session_id != request.session_id
                    or action.sequence_no != request.sequence_no
                    or action.model != request.model
                ):
                    raise RuntimeError(
                        "backend action identity mismatch: refusing to route an action "
                        "to the wrong request or robot session"
                    )
                now = time.monotonic_ns()
                self._record_action_profile(action)
                if now > envelope.request.deadline_ns:
                    self._finish(
                        envelope,
                        ResultStatus.DEADLINE_MISSED,
                        action=None if self.config.drop_missed_actions else action,
                        completed_ns=now,
                    )
                else:
                    self._finish(
                        envelope,
                        ResultStatus.COMPLETED,
                        action=action,
                        completed_ns=now,
                    )
        except Exception as exc:  # backend failures become per-request outcomes
            if self.profiler is not None:
                self.profiler.duration(
                    "backend.infer_failed",
                    (time.perf_counter_ns() - started_ns) / 1_000_000,
                )
            now = time.monotonic_ns()
            for envelope in batch:
                self._finish(
                    envelope,
                    ResultStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                    completed_ns=now,
                )

    def _expire_requests_locked(self) -> None:
        now = time.monotonic_ns()
        for envelope in self._heap:
            if envelope.active and envelope.request.deadline_ns <= now:
                envelope.active = False
                self._clear_session_pointer(envelope)
                self._finish(envelope, ResultStatus.EXPIRED, completed_ns=now)
        self._remove_inactive_locked()

    def _drop_unserviceable_requests_locked(self) -> None:
        """Drop queued work that the backend predicts cannot meet its deadline.

        This is opt-in because an estimator may be absent or intentionally
        optimistic. A conservative backend estimator lets a serial runtime avoid
        spending accelerator time on an action that will be discarded anyway.
        """

        if not self.config.drop_unserviceable_requests:
            return
        predicted_ns = self._predicted_latency_ns(1)
        if predicted_ns is None:
            return
        now = time.monotonic_ns()
        guard_ns = int(self.config.dispatch_guard_ms * 1_000_000)
        for envelope in self._heap:
            if envelope.active and now + predicted_ns + guard_ns > envelope.request.deadline_ns:
                envelope.active = False
                self._clear_session_pointer(envelope)
                self._finish(
                    envelope,
                    ResultStatus.UNSERVICEABLE,
                    completed_ns=now,
                    error="backend latency estimate predicts a deadline miss",
                )
        self._remove_inactive_locked()

    def _remove_inactive_locked(self) -> None:
        if any(not envelope.active for envelope in self._heap):
            self._heap = [envelope for envelope in self._heap if envelope.active]
            heapq.heapify(self._heap)

    def _clear_session_pointer(self, envelope: _Envelope) -> None:
        key = (envelope.request.model, envelope.request.session_id)
        if self._pending_by_session.get(key) is envelope:
            self._pending_by_session.pop(key, None)

    def _finish(
        self,
        envelope: _Envelope,
        status: ResultStatus,
        *,
        action=None,
        completed_ns: int | None = None,
        error: str | None = None,
        replaced_by_request_id: str | None = None,
    ) -> None:
        if envelope.future.done():
            return
        outcome = RequestOutcome(
            request=envelope.request,
            status=status,
            action=action,
            dispatched_ns=envelope.dispatched_ns,
            completed_ns=completed_ns or time.monotonic_ns(),
            error=error,
            replaced_by_request_id=replaced_by_request_id,
        )
        self.metrics.record_outcome(outcome)
        if self.profiler is not None:
            if outcome.queue_ms is not None:
                self.profiler.duration("scheduler.queue", outcome.queue_ms)
            self.profiler.duration("scheduler.end_to_end", outcome.end_to_end_ms)
        envelope.future.set_result(outcome)

    def _record_action_profile(self, action) -> None:
        if self.profiler is None:
            return
        for key, value in action.metadata.items():
            if not key.endswith("_ms") or isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                name = key.removesuffix("_ms").replace("_", ".")
                self.profiler.duration(f"action.{name}", float(value))
