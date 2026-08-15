from __future__ import annotations

import asyncio
import hashlib
import statistics
import time
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from ..types import ActionChunk, InferenceRequest


class RequestTransport(Protocol):
    """Blocking request/reply transport used on the dedicated backend thread."""

    def request(self, payload: bytes) -> bytes: ...

    def close(self) -> None: ...


class _ZmqRequestTransport:
    def __init__(self, address: str, timeout_ms: int) -> None:
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install ActServe with the 'embodied-cpp' extra") from exc

        self._zmq = zmq
        self._address = address
        self._timeout_ms = timeout_ms
        self._socket = self._connect()

    def _connect(self):
        socket = self._zmq.Context.instance().socket(self._zmq.REQ)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        socket.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        socket.connect(self._address)
        return socket

    def request(self, payload: bytes) -> bytes:
        try:
            self._socket.send(payload)
            return bytes(self._socket.recv())
        except Exception:
            # A timed-out REQ socket cannot send again until it has received a
            # reply. Recreate it so one failed request does not poison the
            # backend for all future sessions.
            self._socket.close(linger=0)
            self._socket = self._connect()
            raise

    def close(self) -> None:
        self._socket.close(linger=0)


def _wire_request_id(request_id: str) -> int:
    digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


class EmbodiedCppVlaBackend:
    """ActServe backend for Embodied.cpp's ZeroMQ/Protobuf VLA server.

    Embodied.cpp exposes a serial REQ/REP endpoint, so this backend deliberately
    reports ``max_batch_size = 1``. ActServe contributes admission control,
    per-session frame coalescing, EDF ordering, and explicit deadline outcomes
    before requests reach the C++ runtime.

    ``protobuf_module`` is the generated ``vla_pb2`` module from Embodied.cpp.
    ``request_builder`` performs model-specific preprocessing and returns a
    ``PredictRequest``. The adapter overwrites its numeric request id so replies
    can be validated before an action is routed to a session.
    """

    max_batch_size = 1

    def __init__(
        self,
        *,
        protobuf_module: Any,
        request_builder: Callable[[InferenceRequest], Any],
        address: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 30_000,
        initial_latency_ms: float | None = None,
        latency_window: int = 32,
        latency_safety_factor: float = 1.10,
        transport_factory: Callable[[], RequestTransport] | None = None,
    ) -> None:
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        if initial_latency_ms is not None and initial_latency_ms < 0:
            raise ValueError("initial_latency_ms must be non-negative")
        if latency_window < 1:
            raise ValueError("latency_window must be positive")
        if latency_safety_factor < 1:
            raise ValueError("latency_safety_factor must be at least 1")

        self.protobuf_module = protobuf_module
        self.request_builder = request_builder
        self.address = address
        self.timeout_ms = timeout_ms
        self.initial_latency_ms = initial_latency_ms
        self.latency_safety_factor = latency_safety_factor
        self._latencies_ms: deque[float] = deque(maxlen=latency_window)
        self._transport_factory = transport_factory or (
            lambda: _ZmqRequestTransport(address, timeout_ms)
        )
        self._transport: RequestTransport | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="actserve-embodied")
        self._closed = False

    async def __aenter__(self) -> EmbodiedCppVlaBackend:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def batch_key(self, request: InferenceRequest) -> Hashable:
        route = request.metadata.get("embodied_cpp_route", self.address)
        return request.model, str(route)

    def estimate_batch_latency_ms(self, batch_size: int) -> float | None:
        if batch_size != 1:
            return None
        samples = list(self._latencies_ms)
        if samples:
            if len(samples) == 1:
                observed = samples[0]
            else:
                observed = statistics.quantiles(samples, n=10, method="inclusive")[8]
            estimate = observed * self.latency_safety_factor
            if self.initial_latency_ms is not None:
                estimate = max(estimate, self.initial_latency_ms)
            return estimate
        return self.initial_latency_ms

    async def infer_batch(
        self, requests: Sequence[InferenceRequest]
    ) -> Sequence[ActionChunk]:
        if self._closed:
            raise RuntimeError("EmbodiedCppVlaBackend is closed")
        if len(requests) != 1:
            raise ValueError("Embodied.cpp VLA server accepts exactly one request at a time")
        request = requests[0]
        loop = asyncio.get_running_loop()
        started_ns = time.perf_counter_ns()
        action = await loop.run_in_executor(self._executor, self._infer_one, request)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        self._latencies_ms.append(elapsed_ms)
        return [action]

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._close_transport)
        self._executor.shutdown(wait=True)

    def _infer_one(self, request: InferenceRequest) -> ActionChunk:
        message = self.request_builder(request)
        if not hasattr(message, "SerializeToString") or not hasattr(message, "request_id"):
            raise TypeError("request_builder must return an Embodied.cpp PredictRequest")
        wire_id = _wire_request_id(request.request_id)
        message.request_id = wire_id
        if self._transport is None:
            self._transport = self._transport_factory()
        payload = self._transport.request(message.SerializeToString())

        response = self.protobuf_module.PredictResponse()
        response.ParseFromString(payload)
        if int(response.request_id) != wire_id:
            raise RuntimeError(
                "Embodied.cpp response request_id mismatch: refusing to route the action"
            )
        if response.error:
            raise RuntimeError(f"Embodied.cpp server error: {response.error}")

        chunk_size = int(response.chunk_size)
        action_dim = int(response.action_dim)
        flat_actions = list(response.action_chunk)
        if chunk_size < 1 or action_dim < 1 or len(flat_actions) != chunk_size * action_dim:
            raise RuntimeError(
                "Embodied.cpp returned an invalid action shape: "
                f"chunk_size={chunk_size}, action_dim={action_dim}, values={len(flat_actions)}"
            )
        actions = [
            flat_actions[index * action_dim : (index + 1) * action_dim]
            for index in range(chunk_size)
        ]
        return ActionChunk(
            request_id=request.request_id,
            session_id=request.session_id,
            sequence_no=request.sequence_no,
            actions=actions,
            model=request.model,
            metadata={
                "runtime": "embodied.cpp",
                "server_total_ms": float(response.latency_ms_total),
                "server_inference_ms": float(response.latency_ms_inference),
                "server_vision_ms": float(response.latency_ms_vision),
                "server_prefill_ms": float(response.latency_ms_prefill),
                "server_denoise_ms": float(response.latency_ms_denoise),
                "chunk_size": chunk_size,
                "action_dim": action_dim,
            },
        )

    def _close_transport(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
