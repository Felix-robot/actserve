from __future__ import annotations

from collections.abc import Awaitable, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

from .types import ActionChunk, InferenceRequest


@dataclass(frozen=True, slots=True)
class AdapterRoute:
    """Public routing metadata for one logical policy model."""

    model: str
    backbone: str
    adapter: str | None = None
    input_signature: Hashable = "default"
    adapter_size_mb: float = 0.0

    def __post_init__(self) -> None:
        if not self.model or not self.backbone:
            raise ValueError("model and backbone must be non-empty")
        if self.adapter_size_mb < 0:
            raise ValueError("adapter_size_mb must be non-negative")


@dataclass(frozen=True, slots=True)
class RoutedRequest:
    request: InferenceRequest
    route: AdapterRoute


class AdapterBackend:
    """Route logical policy IDs onto shared backbones without storing inputs.

    Cross-adapter batching is opt-in because it is only valid for runtimes that
    can apply a different adapter per item in one batch.
    """

    def __init__(
        self,
        infer: Callable[[Sequence[RoutedRequest]], Awaitable[Sequence[ActionChunk]]],
        routes: Sequence[AdapterRoute],
        *,
        max_batch_size: int = 8,
        mixed_adapter_batch: bool = False,
        latency_estimator: Callable[[int], float] | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        route_map = {route.model: route for route in routes}
        if len(route_map) != len(routes):
            raise ValueError("route model IDs must be unique")
        if not route_map:
            raise ValueError("at least one adapter route is required")
        self._infer = infer
        self._routes: Mapping[str, AdapterRoute] = route_map
        self._max_batch_size = max_batch_size
        self.mixed_adapter_batch = mixed_adapter_batch
        self._latency_estimator = latency_estimator

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def routes(self) -> tuple[AdapterRoute, ...]:
        return tuple(self._routes.values())

    def route(self, request: InferenceRequest) -> AdapterRoute:
        try:
            return self._routes[request.model]
        except KeyError as exc:
            raise ValueError(f"no adapter route registered for model {request.model!r}") from exc

    def batch_key(self, request: InferenceRequest) -> Hashable:
        route = self.route(request)
        key: tuple[Hashable, ...] = (route.backbone, route.input_signature)
        if not self.mixed_adapter_batch:
            key += (route.adapter,)
        return key

    def estimate_batch_latency_ms(self, batch_size: int) -> float | None:
        if self._latency_estimator is None:
            return None
        return self._latency_estimator(batch_size)

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> Sequence[ActionChunk]:
        routed = [RoutedRequest(request=request, route=self.route(request)) for request in requests]
        return await self._infer(routed)
