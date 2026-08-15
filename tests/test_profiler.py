import json

from actserve.backend import CallableBackend
from actserve.profiler import StageProfiler
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import ActionChunk, InferenceRequest, ResultStatus


def test_profiler_aggregates_bounded_numeric_observations() -> None:
    profiler = StageProfiler(sample_limit=2)
    profiler.observe("gpu.utilization", 10, unit="percent")
    profiler.observe("gpu.utilization", 30, unit="percent")
    profiler.observe("gpu.utilization", 90, unit="percent")
    profiler.observe("gpu.utilization", float("nan"), unit="percent")

    metric = profiler.snapshot().get("gpu.utilization", "percent")
    assert metric is not None
    assert metric.count == 2
    assert metric.mean == 20
    assert metric.minimum == 10
    assert metric.maximum == 30


def test_profile_snapshot_is_json_safe() -> None:
    profiler = StageProfiler()
    with profiler.span("decode"):
        pass
    payload = profiler.snapshot().as_dict()
    assert payload["schema"] == "actserve.profile.v1"
    assert json.loads(json.dumps(payload))["metrics"][0]["name"] == "decode"


async def test_scheduler_records_core_and_backend_phase_timings() -> None:
    async def infer(requests):
        request = requests[0]
        return [
            ActionChunk(
                request_id=request.request_id,
                session_id=request.session_id,
                sequence_no=request.sequence_no,
                actions=[0.0],
                model=request.model,
                metadata={
                    "server_total_ms": 12.0,
                    "server_vision_ms": 3.0,
                    "not_a_timing": 99,
                },
            )
        ]

    profiler = StageProfiler()
    backend = CallableBackend(infer)
    request = InferenceRequest.with_timeout(
        session_id="robot",
        model="test",
        observation={},
        timeout_ms=100,
        sequence_no=1,
    )
    async with Scheduler(
        backend,
        SchedulerConfig(max_batch_wait_ms=0),
        profiler=profiler,
    ) as scheduler:
        outcome = await scheduler.submit(request)

    assert outcome.status is ResultStatus.COMPLETED
    snapshot = profiler.snapshot()
    assert snapshot.get("backend.infer", "ms") is not None
    assert snapshot.get("scheduler.batch_size", "items").mean == 1
    assert snapshot.get("scheduler.queue", "ms") is not None
    assert snapshot.get("scheduler.end_to_end", "ms") is not None
    assert snapshot.get("action.server.total", "ms").mean == 12
    assert snapshot.get("action.server.vision", "ms").mean == 3
    assert snapshot.get("action.not.a.timing", "ms") is None
