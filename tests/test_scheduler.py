import asyncio
import time

from actserve.backend import SimulatedBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import ActionChunk, InferenceRequest, ResultStatus


class BlockingBackend:
    max_batch_size = 1

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.order: list[str] = []

    def batch_key(self, request):
        return request.model

    async def infer_batch(self, requests):
        request = requests[0]
        self.order.append(request.request_id)
        self.started.set()
        await self.release.wait()
        return [
            ActionChunk(
                request_id=request.request_id,
                session_id=request.session_id,
                sequence_no=request.sequence_no,
                actions=[request.sequence_no],
                model=request.model,
            )
        ]


def request(session: str, sequence: int, timeout_ms: float = 500) -> InferenceRequest:
    return InferenceRequest.with_timeout(
        session_id=session,
        model="test",
        observation={"sequence": sequence},
        timeout_ms=timeout_ms,
        sequence_no=sequence,
    )


async def test_new_observation_replaces_pending_one() -> None:
    backend = BlockingBackend()
    async with Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0)) as scheduler:
        first_task = asyncio.create_task(scheduler.submit(request("busy", 0)))
        await backend.started.wait()
        second_future = await scheduler.enqueue(request("robot", 1))
        third_request = request("robot", 2)
        third_future = await scheduler.enqueue(third_request)
        second = await second_future
        assert second.status is ResultStatus.REPLACED
        assert second.replaced_by_request_id == third_request.request_id
        backend.release.set()
        first, third = await asyncio.gather(first_task, third_future)
        assert first.status is ResultStatus.COMPLETED
        assert third.status is ResultStatus.COMPLETED
        assert third.request.sequence_no == 2


async def test_edf_runs_earlier_deadline_first() -> None:
    backend = BlockingBackend()
    config = SchedulerConfig(max_batch_wait_ms=0, coalesce_sessions=False)
    async with Scheduler(backend, config) as scheduler:
        blocker = asyncio.create_task(scheduler.submit(request("blocker", 0)))
        await backend.started.wait()
        late = request("late", 1, timeout_ms=450)
        early = request("early", 1, timeout_ms=250)
        late_future = await scheduler.enqueue(late)
        early_future = await scheduler.enqueue(early)
        backend.release.set()
        await asyncio.gather(blocker, early_future, late_future)
        assert backend.order[1:] == [early.request_id, late.request_id]


async def test_expired_request_never_reaches_backend() -> None:
    backend = SimulatedBackend(fixed_ms=0, per_item_ms=0)
    scheduler = Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0))
    await scheduler.start()
    expired = InferenceRequest(
        session_id="robot",
        model="test",
        observation={},
        deadline_ns=time.monotonic_ns() - 1,
        sequence_no=0,
    )
    outcome = await scheduler.submit(expired)
    await scheduler.close()
    assert outcome.status is ResultStatus.EXPIRED
    assert backend.calls == []


async def test_compatible_requests_are_microbatched() -> None:
    backend = SimulatedBackend(fixed_ms=0, per_item_ms=0, max_batch_size=4)
    async with Scheduler(backend, SchedulerConfig(max_batch_wait_ms=10)) as scheduler:
        outcomes = await asyncio.gather(
            *(scheduler.submit(request(f"robot-{index}", 0)) for index in range(4))
        )
    assert all(outcome.status is ResultStatus.COMPLETED for outcome in outcomes)
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 4


async def test_missed_actions_are_dropped() -> None:
    backend = SimulatedBackend(fixed_ms=20, per_item_ms=0)
    async with Scheduler(
        backend,
        SchedulerConfig(max_batch_wait_ms=0, drop_missed_actions=True),
    ) as scheduler:
        outcome = await scheduler.submit(request("robot", 0, timeout_ms=5))
    assert outcome.status is ResultStatus.DEADLINE_MISSED
    assert outcome.action is None


async def test_latency_estimator_prevents_harmful_batch() -> None:
    backend = SimulatedBackend(fixed_ms=10, per_item_ms=10, max_batch_size=2)
    config = SchedulerConfig(max_batch_wait_ms=10, dispatch_guard_ms=0)
    async with Scheduler(backend, config) as scheduler:
        urgent = request("urgent", 0, timeout_ms=35)
        relaxed = request("relaxed", 0, timeout_ms=100)
        outcomes = await asyncio.gather(
            scheduler.submit(urgent),
            scheduler.submit(relaxed),
        )
    assert outcomes[0].status is ResultStatus.COMPLETED
    assert len(backend.calls[0]) == 1


async def test_latency_estimator_can_drop_predicted_deadline_miss() -> None:
    backend = SimulatedBackend(fixed_ms=20, per_item_ms=0)
    config = SchedulerConfig(
        max_batch_wait_ms=0,
        dispatch_guard_ms=1,
        drop_unserviceable_requests=True,
    )
    async with Scheduler(backend, config) as scheduler:
        outcome = await scheduler.submit(request("robot", 0, timeout_ms=5))
    assert outcome.status is ResultStatus.UNSERVICEABLE
    assert backend.calls == []


async def test_out_of_order_observation_is_rejected() -> None:
    backend = SimulatedBackend(fixed_ms=0, per_item_ms=0)
    async with Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0)) as scheduler:
        current = await scheduler.submit(request("robot", 2))
        stale = await scheduler.submit(request("robot", 1))
    assert current.status is ResultStatus.COMPLETED
    assert stale.status is ResultStatus.OUT_OF_ORDER
    assert len(backend.calls) == 1


async def test_backend_identity_mismatch_fails_closed() -> None:
    async def wrong_backend(requests):
        item = requests[0]
        return [
            ActionChunk(
                request_id="wrong-request",
                session_id=item.session_id,
                sequence_no=item.sequence_no,
                actions=[],
                model=item.model,
            )
        ]

    from actserve.backend import CallableBackend

    backend = CallableBackend(wrong_backend)
    async with Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0)) as scheduler:
        outcome = await scheduler.submit(request("robot", 0))
    assert outcome.status is ResultStatus.FAILED
    assert "identity mismatch" in (outcome.error or "")
