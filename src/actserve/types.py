from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResultStatus(str, Enum):
    COMPLETED = "completed"
    DEADLINE_MISSED = "deadline_missed"
    EXPIRED = "expired_before_dispatch"
    REPLACED = "replaced_by_newer_observation"
    OUT_OF_ORDER = "out_of_order_observation"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """One observation that may produce an action chunk.

    ``deadline_ns`` and ``received_ns`` use ``time.monotonic_ns``. Wall-clock
    timestamps are deliberately excluded from scheduling decisions.
    """

    session_id: str
    model: str
    observation: Any
    deadline_ns: int
    sequence_no: int
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    received_ns: int = field(default_factory=time.monotonic_ns)
    priority: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def with_timeout(
        cls,
        *,
        session_id: str,
        model: str,
        observation: Any,
        timeout_ms: float,
        sequence_no: int,
        priority: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> InferenceRequest:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        now = time.monotonic_ns()
        return cls(
            session_id=session_id,
            model=model,
            observation=observation,
            deadline_ns=now + int(timeout_ms * 1_000_000),
            sequence_no=sequence_no,
            received_ns=now,
            priority=priority,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class ActionChunk:
    request_id: str
    session_id: str
    sequence_no: int
    actions: Any
    model: str
    generated_ns: int = field(default_factory=time.monotonic_ns)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RequestOutcome:
    request: InferenceRequest
    status: ResultStatus
    action: ActionChunk | None
    dispatched_ns: int | None
    completed_ns: int
    error: str | None = None
    replaced_by_request_id: str | None = None

    @property
    def queue_ms(self) -> float | None:
        if self.dispatched_ns is None:
            return None
        return (self.dispatched_ns - self.request.received_ns) / 1_000_000

    @property
    def end_to_end_ms(self) -> float:
        return (self.completed_ns - self.request.received_ns) / 1_000_000

    @property
    def deadline_lateness_ms(self) -> float:
        return max(0.0, (self.completed_ns - self.request.deadline_ns) / 1_000_000)
