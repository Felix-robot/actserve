from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Hashable, Sequence
from typing import Protocol

from .types import ActionChunk, InferenceRequest


class InferenceBackend(Protocol):
    """Minimal backend contract.

    A backend can wrap PyTorch, TensorRT, vLLM, embodied.cpp, a remote RPC
    service, or a deterministic replay implementation.
    """

    @property
    def max_batch_size(self) -> int: ...

    def batch_key(self, request: InferenceRequest) -> Hashable: ...

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> Sequence[ActionChunk]: ...


class CallableBackend:
    """Adapter for an async Python batch function."""

    def __init__(
        self,
        infer: Callable[[Sequence[InferenceRequest]], Awaitable[Sequence[ActionChunk]]],
        *,
        max_batch_size: int = 1,
        key: Callable[[InferenceRequest], Hashable] | None = None,
        latency_estimator: Callable[[int], float] | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self._infer = infer
        self._max_batch_size = max_batch_size
        self._key = key or (lambda request: request.model)
        self._latency_estimator = latency_estimator

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def batch_key(self, request: InferenceRequest) -> Hashable:
        return self._key(request)

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> Sequence[ActionChunk]:
        return await self._infer(requests)

    def estimate_batch_latency_ms(self, batch_size: int) -> float | None:
        if self._latency_estimator is None:
            return None
        return self._latency_estimator(batch_size)


class SimulatedBackend:
    """Deterministic latency model used by tests and the public benchmark."""

    def __init__(
        self,
        *,
        fixed_ms: float = 20.0,
        per_item_ms: float = 2.0,
        max_batch_size: int = 8,
    ) -> None:
        if fixed_ms < 0 or per_item_ms < 0:
            raise ValueError("latencies must be non-negative")
        self.fixed_ms = fixed_ms
        self.per_item_ms = per_item_ms
        self._max_batch_size = max_batch_size
        self.calls: list[list[str]] = []

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def batch_key(self, request: InferenceRequest) -> Hashable:
        return request.model

    def estimate_batch_latency_ms(self, batch_size: int) -> float:
        return self.fixed_ms + self.per_item_ms * batch_size

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> Sequence[ActionChunk]:
        self.calls.append([request.request_id for request in requests])
        await asyncio.sleep((self.fixed_ms + self.per_item_ms * len(requests)) / 1000)
        return [
            ActionChunk(
                request_id=request.request_id,
                session_id=request.session_id,
                sequence_no=request.sequence_no,
                actions={"echo": request.observation},
                model=request.model,
            )
            for request in requests
        ]
