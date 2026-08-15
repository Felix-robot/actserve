from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from .profiler import StageProfiler
from .types import ActionChunk


class ActionQueueEmpty(TimeoutError):
    """No action became available before the caller's deadline."""


class ActionQueueClosed(RuntimeError):
    """The queue is closed and cannot produce another action."""


@dataclass(frozen=True, slots=True)
class ActionQueueConfig:
    low_watermark: int = 5
    handoff_policy: Literal["replace", "append"] = "replace"

    def __post_init__(self) -> None:
        if self.low_watermark < 0:
            raise ValueError("low_watermark must be non-negative")


@dataclass(frozen=True, slots=True)
class QueuedAction:
    action: Any
    request_id: str
    session_id: str
    model: str
    sequence_no: int
    chunk_index: int
    chunk_size: int
    generated_ns: int

    @property
    def age_ms(self) -> float:
        return (time.monotonic_ns() - self.generated_ns) / 1_000_000


@dataclass(frozen=True, slots=True)
class ActionQueueSnapshot:
    remaining_actions: int
    latest_sequence_no: int | None
    refill_inflight: bool
    closed: bool
    accepted_chunks: int
    rejected_chunks: int
    served_actions: int
    replaced_actions: int
    underruns: int

    def as_dict(self) -> dict[str, int | bool | None]:
        return {
            "remaining_actions": self.remaining_actions,
            "latest_sequence_no": self.latest_sequence_no,
            "refill_inflight": self.refill_inflight,
            "closed": self.closed,
            "accepted_chunks": self.accepted_chunks,
            "rejected_chunks": self.rejected_chunks,
            "served_actions": self.served_actions,
            "replaced_actions": self.replaced_actions,
            "underruns": self.underruns,
        }


class AsyncActionQueue:
    """Per-session action buffer with explicit asynchronous refill signaling.

    The queue never invents a fallback action. A caller that reaches an empty
    queue gets ``ActionQueueEmpty`` and must apply its robot-specific safety
    policy outside ActServe.
    """

    def __init__(
        self,
        *,
        session_id: str,
        model: str,
        config: ActionQueueConfig | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if not session_id or not model:
            raise ValueError("session_id and model must be non-empty")
        self.session_id = session_id
        self.model = model
        self.config = config or ActionQueueConfig()
        self.profiler = profiler
        self._condition = asyncio.Condition()
        self._actions: deque[QueuedAction] = deque()
        self._latest_sequence_no: int | None = None
        self._refill_inflight = False
        self._closed = False
        self._accepted_chunks = 0
        self._rejected_chunks = 0
        self._served_actions = 0
        self._replaced_actions = 0
        self._underruns = 0

    @property
    def remaining_actions(self) -> int:
        return len(self._actions)

    @property
    def needs_refill(self) -> bool:
        return (
            not self._closed
            and not self._refill_inflight
            and len(self._actions) <= self.config.low_watermark
        )

    async def put(self, chunk: ActionChunk) -> bool:
        """Publish a chunk, returning false when it is stale or misrouted."""

        async with self._condition:
            accepted = self._put_locked(chunk)
            self._condition.notify_all()
            return accepted

    async def finish_refill(self, chunk: ActionChunk | None = None) -> bool:
        """Release a refill slot and optionally publish its completed chunk."""

        async with self._condition:
            accepted = True
            if chunk is not None:
                accepted = self._put_locked(chunk)
            self._refill_inflight = False
            self._condition.notify_all()
            return accepted

    async def acquire_refill(self) -> bool:
        """Wait for low water and atomically claim the single refill slot.

        Returns false after the queue is closed.
        """

        async with self._condition:
            while not self.needs_refill:
                if self._closed:
                    return False
                await self._condition.wait()
            self._refill_inflight = True
            return True

    async def get(self, *, timeout_ms: float | None = None) -> QueuedAction:
        if timeout_ms is not None and timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

        async def wait_for_action() -> QueuedAction:
            async with self._condition:
                while not self._actions:
                    if self._closed:
                        raise ActionQueueClosed("action queue is closed")
                    await self._condition.wait()
                action = self._actions.popleft()
                self._served_actions += 1
                self._record_depth_locked()
                self._condition.notify_all()
                return action

        try:
            if timeout_ms is None:
                action = await wait_for_action()
            else:
                action = await asyncio.wait_for(wait_for_action(), timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            async with self._condition:
                self._underruns += 1
                if self.profiler is not None:
                    self.profiler.observe("action_queue.underrun", 1, unit="events")
            raise ActionQueueEmpty("no action available before timeout") from exc

        if self.profiler is not None:
            self.profiler.duration("action_queue.action_age", action.age_ms)
        return action

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    async def snapshot(self) -> ActionQueueSnapshot:
        async with self._condition:
            return ActionQueueSnapshot(
                remaining_actions=len(self._actions),
                latest_sequence_no=self._latest_sequence_no,
                refill_inflight=self._refill_inflight,
                closed=self._closed,
                accepted_chunks=self._accepted_chunks,
                rejected_chunks=self._rejected_chunks,
                served_actions=self._served_actions,
                replaced_actions=self._replaced_actions,
                underruns=self._underruns,
            )

    def _put_locked(self, chunk: ActionChunk) -> bool:
        if self._closed:
            raise ActionQueueClosed("cannot publish to a closed action queue")
        if chunk.session_id != self.session_id or chunk.model != self.model:
            self._rejected_chunks += 1
            return False
        if self._latest_sequence_no is not None and chunk.sequence_no <= self._latest_sequence_no:
            self._rejected_chunks += 1
            return False
        actions = list(chunk.actions)
        if not actions:
            self._rejected_chunks += 1
            return False

        if self.config.handoff_policy == "replace":
            self._replaced_actions += len(self._actions)
            self._actions.clear()
        chunk_size = len(actions)
        self._actions.extend(
            QueuedAction(
                action=action,
                request_id=chunk.request_id,
                session_id=chunk.session_id,
                model=chunk.model,
                sequence_no=chunk.sequence_no,
                chunk_index=index,
                chunk_size=chunk_size,
                generated_ns=chunk.generated_ns,
            )
            for index, action in enumerate(actions)
        )
        self._latest_sequence_no = chunk.sequence_no
        self._accepted_chunks += 1
        self._record_depth_locked()
        return True

    def _record_depth_locked(self) -> None:
        if self.profiler is not None:
            self.profiler.observe("action_queue.depth", len(self._actions), unit="actions")

