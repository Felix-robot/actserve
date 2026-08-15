from __future__ import annotations

from actserve.adapter_backend import AdapterBackend, AdapterRoute
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import ActionChunk, InferenceRequest, ResultStatus


async def test_mixed_adapter_backend_batches_shared_backbone_requests() -> None:
    calls = []

    async def infer(routed):
        calls.append([(item.route.backbone, item.route.adapter) for item in routed])
        return [
            ActionChunk(
                request_id=item.request.request_id,
                session_id=item.request.session_id,
                sequence_no=item.request.sequence_no,
                actions=[item.route.adapter],
                model=item.request.model,
            )
            for item in routed
        ]

    backend = AdapterBackend(
        infer,
        [
            AdapterRoute("pick", "shared-vla", "pick-lora"),
            AdapterRoute("place", "shared-vla", "place-lora"),
        ],
        mixed_adapter_batch=True,
    )
    requests = [
        InferenceRequest.with_timeout(
            session_id=model,
            model=model,
            observation={},
            timeout_ms=100,
            sequence_no=1,
        )
        for model in ("pick", "place")
    ]
    async with Scheduler(
        backend,
        SchedulerConfig(max_batch_wait_ms=5, dispatch_guard_ms=0),
    ) as scheduler:
        futures = [await scheduler.enqueue(request) for request in requests]
        outcomes = [await future for future in futures]

    assert len(calls) == 1
    assert calls[0] == [("shared-vla", "pick-lora"), ("shared-vla", "place-lora")]
    assert [outcome.status for outcome in outcomes] == [
        ResultStatus.COMPLETED,
        ResultStatus.COMPLETED,
    ]


async def test_adapter_is_part_of_batch_key_without_explicit_mixed_support() -> None:
    async def infer(_):
        return []

    backend = AdapterBackend(
        infer,
        [
            AdapterRoute("pick", "shared-vla", "pick-lora"),
            AdapterRoute("place", "shared-vla", "place-lora"),
        ],
    )
    pick = InferenceRequest.with_timeout(
        session_id="one", model="pick", observation={}, timeout_ms=100, sequence_no=1
    )
    place = InferenceRequest.with_timeout(
        session_id="two", model="place", observation={}, timeout_ms=100, sequence_no=1
    )
    assert backend.batch_key(pick) != backend.batch_key(place)


def test_adapter_backend_rejects_unknown_models() -> None:
    async def infer(_):
        return []

    backend = AdapterBackend(infer, [AdapterRoute("pick", "shared-vla")])
    request = InferenceRequest.with_timeout(
        session_id="one", model="unknown", observation={}, timeout_ms=100, sequence_no=1
    )
    try:
        backend.batch_key(request)
    except ValueError as exc:
        assert "no adapter route" in str(exc)
    else:
        raise AssertionError("unknown model route was accepted")
