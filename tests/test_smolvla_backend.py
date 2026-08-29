from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import pytest

from actserve.integrations.smolvla import SmolVLABackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import InferenceRequest, ResultStatus


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    return (len(value), *_shape(value[0]))


class FakeTensor:
    def __init__(self, data: Any) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.data)

    def unsqueeze(self, dim: int) -> FakeTensor:
        assert dim == 0
        return FakeTensor([self.data])

    def to(self, **_: Any) -> FakeTensor:
        return self

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> Any:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> FakeTensor:
        return FakeTensor(self.data[index])


class FakeTorch:
    float32 = object()

    @staticmethod
    def is_tensor(value: Any) -> bool:
        return isinstance(value, FakeTensor)

    @staticmethod
    def as_tensor(value: Any, **_: Any) -> FakeTensor:
        return FakeTensor(value)

    @staticmethod
    def cat(tensors: list[FakeTensor], *, dim: int) -> FakeTensor:
        assert dim == 0
        combined = []
        for tensor in tensors:
            combined.extend(tensor.data)
        return FakeTensor(combined)

    @staticmethod
    def inference_mode():
        return nullcontext()


@dataclass
class FakeFeature:
    shape: tuple[int, ...]


class FakeConfig:
    n_obs_steps = 1
    input_features = {
        "observation.state": FakeFeature((2,)),
        "observation.image": FakeFeature((3, 2, 2)),
    }


class FakePolicy:
    config = FakeConfig()

    def __init__(self, *, delay_seconds: float = 0) -> None:
        self.delay_seconds = delay_seconds
        self.calls: list[dict[str, Any]] = []
        self.thread_names: list[str] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_action_chunk(self, batch: dict[str, Any]) -> FakeTensor:
        self.thread_names.append(threading.current_thread().name)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        self.calls.append(batch)
        states = batch["observation.state"].data
        return FakeTensor([[state] for state in states])


@pytest.fixture(autouse=True)
def fake_torch_module(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)


def make_observation(value: float) -> dict[str, Any]:
    return {
        "observation.state": [value, value + 1],
        "observation.image": [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.9, 1.0], [0.2, 0.3]],
        ],
        "task": f"Move public object {value}",
    }


def make_request(session: str, value: float) -> InferenceRequest:
    return InferenceRequest.with_timeout(
        session_id=session,
        model="public-smolvla",
        observation=make_observation(value),
        timeout_ms=1_000,
        sequence_no=1,
        metadata={"input_signature": "smolvla-public-v1"},
    )


async def test_smolvla_backend_batches_off_loop_and_preserves_identity() -> None:
    policy = FakePolicy(delay_seconds=0.03)
    backend = SmolVLABackend(
        policy,
        lambda batch: batch,
        lambda actions: actions,
        model_name="public-smolvla",
        max_batch_size=2,
        device="test-device",
    )
    requests = [make_request("robot-a", 1.0), make_request("robot-b", 3.0)]
    async with Scheduler(
        backend,
        SchedulerConfig(max_batch_size=2, max_batch_wait_ms=5, dispatch_guard_ms=0),
    ) as scheduler:
        futures = [await scheduler.enqueue(request) for request in requests]
        outcomes = [await future for future in futures]

    assert len(policy.calls) == 1
    assert policy.calls[0]["task"] == [
        "Move public object 1.0",
        "Move public object 3.0",
    ]
    assert policy.thread_names[0].startswith("actserve-smolvla")
    assert policy.reset_calls == 1
    assert [outcome.status for outcome in outcomes] == [
        ResultStatus.COMPLETED,
        ResultStatus.COMPLETED,
    ]
    assert outcomes[0].action is not None
    assert outcomes[0].action.request_id == requests[0].request_id
    assert outcomes[0].action.actions == [[1.0, 2.0]]
    assert outcomes[1].action is not None
    assert outcomes[1].action.session_id == "robot-b"
    assert outcomes[1].action.metadata == {
        "backend": "lerobot_smolvla",
        "device": "test-device",
    }
    assert backend.estimate_batch_latency_ms(2) is not None
    await backend.aclose()


async def test_smolvla_backend_does_not_block_event_loop() -> None:
    backend = SmolVLABackend(
        FakePolicy(delay_seconds=0.05),
        lambda batch: batch,
        lambda actions: actions,
        model_name="public-smolvla",
    )
    inference = asyncio.create_task(backend.infer_batch([make_request("robot-a", 1.0)]))
    await asyncio.sleep(0.005)
    assert not inference.done()
    await inference
    await backend.aclose()


async def test_smolvla_backend_rejects_wrong_model_and_input_shape() -> None:
    backend = SmolVLABackend(
        FakePolicy(),
        lambda batch: batch,
        lambda actions: actions,
        model_name="public-smolvla",
    )
    wrong_model = InferenceRequest.with_timeout(
        session_id="robot-a",
        model="wrong-model",
        observation=make_observation(1.0),
        timeout_ms=1_000,
        sequence_no=1,
    )
    with pytest.raises(ValueError, match="does not match served model"):
        await backend.infer_batch([wrong_model])

    wrong_shape = make_request("robot-a", 1.0)
    wrong_shape.observation["observation.state"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="observation.state.*shape"):
        await backend.infer_batch([wrong_shape])
    await backend.aclose()


def test_smolvla_backend_rejects_stateful_policy() -> None:
    policy = FakePolicy()
    policy.config = FakeConfig()
    policy.config.n_obs_steps = 2
    with pytest.raises(ValueError, match="n_obs_steps == 1"):
        SmolVLABackend(
            policy,
            lambda batch: batch,
            lambda actions: actions,
            model_name="public-smolvla",
        )


def test_smolvla_backend_latency_floor_is_available_before_warmup() -> None:
    backend = SmolVLABackend(
        FakePolicy(),
        lambda batch: batch,
        lambda actions: actions,
        model_name="public-smolvla",
        initial_latency_ms=25,
    )
    assert backend.estimate_batch_latency_ms(1) == 25
    asyncio.run(backend.aclose())
