from __future__ import annotations

import asyncio
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..types import ActionChunk, InferenceRequest

DEFAULT_SMOLVLA_MODEL_ID = "lerobot/smolvla_base"
DEFAULT_SMOLVLA_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


class SmolVLABackend:
    """In-process LeRobot SmolVLA backend with non-blocking async dispatch.

    Each request contains one independent observation. Stateful policies with
    more than one observation step are rejected because batching robot-local
    history without an explicit state contract can cross session boundaries.
    """

    def __init__(
        self,
        policy: Any,
        preprocess: Any,
        postprocess: Any,
        *,
        model_name: str,
        max_batch_size: int = 4,
        device: str = "unknown",
        initial_latency_ms: float | None = None,
        latency_window: int = 32,
        latency_safety_factor: float = 1.10,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if initial_latency_ms is not None and initial_latency_ms < 0:
            raise ValueError("initial_latency_ms must be non-negative")
        if latency_window < 1:
            raise ValueError("latency_window must be positive")
        if latency_safety_factor < 1:
            raise ValueError("latency_safety_factor must be at least 1")
        config = getattr(policy, "config", None)
        if config is None:
            raise ValueError("policy must expose a config")
        if getattr(config, "n_obs_steps", None) != 1:
            raise ValueError("SmolVLABackend requires policy.config.n_obs_steps == 1")
        input_features = getattr(config, "input_features", None)
        if not isinstance(input_features, Mapping) or not input_features:
            raise ValueError("policy.config.input_features must be a non-empty mapping")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional installation
            raise RuntimeError("Install PyTorch and lerobot[smolvla] for SmolVLABackend") from exc

        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.model_name = model_name
        self.device = device
        self._max_batch_size = max_batch_size
        self.initial_latency_ms = initial_latency_ms
        self.latency_safety_factor = latency_safety_factor
        self._torch = torch
        self._expected_shapes = {
            name: tuple(feature.shape) for name, feature in input_features.items()
        }
        self._latencies_ms: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=latency_window)
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="actserve-smolvla")
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_SMOLVLA_MODEL_ID,
        *,
        revision: str | None = None,
        device: str = "auto",
        model_name: str | None = None,
        max_batch_size: int = 4,
        initial_latency_ms: float | None = None,
        latency_window: int = 32,
        latency_safety_factor: float = 1.10,
    ) -> SmolVLABackend:
        """Load public or local SmolVLA weights and their matching processors."""

        try:
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.smolvla import SmolVLAPolicy
        except ImportError as exc:  # pragma: no cover - optional installation
            raise RuntimeError("Install PyTorch and lerobot[smolvla] for SmolVLABackend") from exc

        if not model_id:
            raise ValueError("model_id must be non-empty")
        resolved_device = _resolve_device(torch, device)
        policy = SmolVLAPolicy.from_pretrained(model_id, revision=revision)
        policy.to(torch.device(resolved_device))
        policy.eval()
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            model_id,
            pretrained_revision=revision,
            preprocessor_overrides={"device_processor": {"device": resolved_device}},
        )
        return cls(
            policy,
            preprocess,
            postprocess,
            model_name=model_id if model_name is None else model_name,
            max_batch_size=max_batch_size,
            device=resolved_device,
            initial_latency_ms=initial_latency_ms,
            latency_window=latency_window,
            latency_safety_factor=latency_safety_factor,
        )

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    async def __aenter__(self) -> SmolVLABackend:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def batch_key(self, request: InferenceRequest) -> Hashable:
        input_signature = request.metadata.get("input_signature", "default")
        return self.model_name, request.model, str(input_signature)

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
            candidates.append(observed * max(1.0, batch_size / observed_size))
        estimate = max(candidates) * self.latency_safety_factor if candidates else None
        if self.initial_latency_ms is not None:
            estimate = (
                self.initial_latency_ms
                if estimate is None
                else max(estimate, self.initial_latency_ms)
            )
        return estimate

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> Sequence[ActionChunk]:
        if self._closed:
            raise RuntimeError("SmolVLABackend is closed")
        if not requests:
            return []
        if len(requests) > self.max_batch_size:
            raise ValueError(
                f"batch has {len(requests)} requests, maximum is {self.max_batch_size}"
            )
        request_list = list(requests)
        started_ns = time.perf_counter_ns()
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(self._executor, self._infer_sync, request_list)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        self._latencies_ms[len(request_list)].append(elapsed_ms)
        return chunks

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _infer_sync(self, requests: Sequence[InferenceRequest]) -> list[ActionChunk]:
        for request in requests:
            if request.model != self.model_name:
                raise ValueError(
                    f"request model {request.model!r} does not match "
                    f"served model {self.model_name!r}"
                )
        batch = self.preprocess(self._stack_observations(requests))
        self.policy.reset()
        with self._torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
            decoded = self.postprocess(actions).detach().cpu()
        if len(decoded) != len(requests):
            raise RuntimeError(
                f"SmolVLA returned {len(decoded)} action chunks for {len(requests)} requests"
            )
        return [
            ActionChunk(
                request_id=request.request_id,
                session_id=request.session_id,
                sequence_no=request.sequence_no,
                actions=decoded[index].tolist(),
                model=request.model,
                metadata={"backend": "lerobot_smolvla", "device": self.device},
            )
            for index, request in enumerate(requests)
        ]

    def _stack_observations(self, requests: Sequence[InferenceRequest]) -> dict[str, Any]:
        observations = []
        for request in requests:
            if not isinstance(request.observation, Mapping):
                raise TypeError("SmolVLA observations must be mappings")
            observations.append(request.observation)

        batch: dict[str, Any] = {}
        for name, expected_shape in self._expected_shapes.items():
            tensors = []
            for observation in observations:
                if name not in observation:
                    raise ValueError(f"SmolVLA observation is missing {name!r}")
                value = observation[name]
                tensor = (
                    value
                    if self._torch.is_tensor(value)
                    else self._torch.as_tensor(value, dtype=self._torch.float32)
                )
                if tuple(tensor.shape) == expected_shape:
                    tensor = tensor.unsqueeze(0)
                expected_batched_shape = (1, *expected_shape)
                if tuple(tensor.shape) != expected_batched_shape:
                    raise ValueError(
                        f"SmolVLA input {name!r} has shape {tuple(tensor.shape)}, "
                        f"expected {expected_shape} or {expected_batched_shape}"
                    )
                tensor = tensor.to(dtype=self._torch.float32)
                tensors.append(tensor)
            batch[name] = self._torch.cat(tensors, dim=0)

        tasks = []
        for observation in observations:
            task = observation.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("SmolVLA observation 'task' must be a non-empty string")
            tasks.append(task)
        batch["task"] = tasks
        return batch


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    return requested
