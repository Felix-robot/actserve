from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from ..types import ActionChunk, InferenceRequest


class HttpJsonBackend:
    """Batch adapter for an existing JSON/HTTP policy inference service.

    ActServe sends one JSON object with a ``requests`` array and expects one
    JSON object with an ``actions`` array. Identity fields remain explicit so
    the scheduler can fail closed before routing an action to a robot session.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        max_batch_size: int = 8,
        timeout_ms: float = 30_000,
        initial_latency_ms: float | None = None,
        latency_window: int = 32,
        latency_safety_factor: float = 1.10,
        headers: Mapping[str, str] | None = None,
        client: Any | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if initial_latency_ms is not None and initial_latency_ms < 0:
            raise ValueError("initial_latency_ms must be non-negative")
        if latency_window < 1:
            raise ValueError("latency_window must be positive")
        if latency_safety_factor < 1:
            raise ValueError("latency_safety_factor must be at least 1")
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install ActServe with the 'http' extra") from exc

        self.endpoint = endpoint
        self._max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.initial_latency_ms = initial_latency_ms
        self.latency_safety_factor = latency_safety_factor
        self._latency_window = latency_window
        self._latencies_ms: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._latency_window)
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_ms / 1000,
            headers=dict(headers or {}),
        )
        self._closed = False

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    async def __aenter__(self) -> HttpJsonBackend:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def batch_key(self, request: InferenceRequest) -> Hashable:
        input_signature = request.metadata.get("input_signature", "default")
        return self.endpoint, request.model, str(input_signature)

    def estimate_batch_latency_ms(self, batch_size: int) -> float | None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        candidates = []
        for observed_size, samples in self._latencies_ms.items():
            if not samples:
                continue
            observed = (
                samples[0]
                if len(samples) == 1
                else statistics.quantiles(samples, n=10, method="inclusive")[8]
            )
            scale = max(1.0, batch_size / observed_size)
            candidates.append(observed * scale)
        estimate = max(candidates) * self.latency_safety_factor if candidates else None
        if self.initial_latency_ms is not None:
            estimate = (
                self.initial_latency_ms
                if estimate is None
                else max(estimate, self.initial_latency_ms)
            )
        return estimate

    async def infer_batch(
        self, requests: Sequence[InferenceRequest]
    ) -> Sequence[ActionChunk]:
        if self._closed:
            raise RuntimeError("HttpJsonBackend is closed")
        if not requests:
            return []
        if len(requests) > self.max_batch_size:
            raise ValueError(
                f"batch has {len(requests)} requests, maximum is {self.max_batch_size}"
            )

        started_ns = time.perf_counter_ns()
        response = await self._client.post(
            self.endpoint,
            json={"requests": [self._request_payload(request) for request in requests]},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
            raise RuntimeError("HTTP backend response must be an object with an actions array")
        actions = payload["actions"]
        if len(actions) != len(requests):
            raise RuntimeError(
                f"HTTP backend returned {len(actions)} actions for {len(requests)} requests"
            )
        chunks = [self._action_chunk(item) for item in actions]
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        self._latencies_ms[len(requests)].append(elapsed_ms)
        return chunks

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _request_payload(request: InferenceRequest) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "model": request.model,
            "sequence_no": request.sequence_no,
            "observation": request.observation,
            "metadata": dict(request.metadata),
        }

    @staticmethod
    def _action_chunk(payload: Any) -> ActionChunk:
        if not isinstance(payload, dict):
            raise RuntimeError("every HTTP backend action must be an object")
        required = ("request_id", "session_id", "model", "sequence_no", "actions")
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeError(f"HTTP backend action is missing fields: {', '.join(missing)}")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise RuntimeError("HTTP backend action metadata must be an object")
        return ActionChunk(
            request_id=str(payload["request_id"]),
            session_id=str(payload["session_id"]),
            model=str(payload["model"]),
            sequence_no=int(payload["sequence_no"]),
            actions=payload["actions"],
            metadata=metadata,
        )
