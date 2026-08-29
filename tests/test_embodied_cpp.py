import json

from actserve.integrations.embodied_cpp import EmbodiedCppVlaBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import InferenceRequest, ResultStatus


class FakeRequest:
    def __init__(self) -> None:
        self.request_id = 0
        self.value = 0.0

    def SerializeToString(self) -> bytes:
        return json.dumps({"request_id": self.request_id, "value": self.value}).encode()


class FakeResponse:
    def __init__(self) -> None:
        self.request_id = 0
        self.action_chunk = []
        self.chunk_size = 0
        self.action_dim = 0
        self.latency_ms_total = 0.0
        self.latency_ms_inference = 0.0
        self.latency_ms_vision = 0.0
        self.latency_ms_prefill = 0.0
        self.latency_ms_denoise = 0.0
        self.error = ""

    def ParseFromString(self, payload: bytes) -> None:
        values = json.loads(payload)
        for key, value in values.items():
            setattr(self, key, value)


class FakeProtobuf:
    PredictResponse = FakeResponse


class FakeTransport:
    def __init__(self, *, wrong_id: bool = False) -> None:
        self.wrong_id = wrong_id
        self.calls = 0
        self.closed = False

    def request(self, payload: bytes) -> bytes:
        self.calls += 1
        request = json.loads(payload)
        response = {
            "request_id": request["request_id"] + int(self.wrong_id),
            "action_chunk": [request["value"], 2.0],
            "chunk_size": 1,
            "action_dim": 2,
            "latency_ms_total": 4.0,
            "latency_ms_inference": 3.0,
            "latency_ms_vision": 1.0,
            "error": "",
        }
        return json.dumps(response).encode()

    def close(self) -> None:
        self.closed = True


def make_request(timeout_ms: float = 100) -> InferenceRequest:
    return InferenceRequest.with_timeout(
        session_id="robot-1",
        model="hy-vla",
        observation={"value": 7.0},
        timeout_ms=timeout_ms,
        sequence_no=3,
    )


def build_request(request: InferenceRequest) -> FakeRequest:
    message = FakeRequest()
    message.value = request.observation["value"]
    return message


async def test_embodied_cpp_backend_routes_validated_action() -> None:
    transport = FakeTransport()
    async with EmbodiedCppVlaBackend(
        protobuf_module=FakeProtobuf,
        request_builder=build_request,
        transport_factory=lambda: transport,
    ) as backend:
        async with Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0)) as scheduler:
            outcome = await scheduler.submit(make_request())
        assert backend.estimate_batch_latency_ms(1) is not None

    assert outcome.status is ResultStatus.COMPLETED
    assert outcome.action is not None
    assert outcome.action.actions == [[7.0, 2.0]]
    assert outcome.action.metadata["runtime"] == "embodied.cpp"
    assert transport.calls == 1
    assert transport.closed


async def test_embodied_cpp_response_identity_mismatch_fails_closed() -> None:
    transport = FakeTransport(wrong_id=True)
    async with EmbodiedCppVlaBackend(
        protobuf_module=FakeProtobuf,
        request_builder=build_request,
        transport_factory=lambda: transport,
    ) as backend:
        async with Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0)) as scheduler:
            outcome = await scheduler.submit(make_request())

    assert outcome.status is ResultStatus.FAILED
    assert "request_id mismatch" in (outcome.error or "")


async def test_unserviceable_request_never_reaches_embodied_cpp() -> None:
    transport = FakeTransport()
    async with EmbodiedCppVlaBackend(
        protobuf_module=FakeProtobuf,
        request_builder=build_request,
        initial_latency_ms=20,
        transport_factory=lambda: transport,
    ) as backend:
        config = SchedulerConfig(
            max_batch_wait_ms=0,
            dispatch_guard_ms=1,
            drop_unserviceable_requests=True,
        )
        async with Scheduler(backend, config) as scheduler:
            outcome = await scheduler.submit(make_request(timeout_ms=5))

    assert outcome.status is ResultStatus.UNSERVICEABLE
    assert transport.calls == 0
