from __future__ import annotations

import httpx

from actserve.integrations.http_json import HttpJsonBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import InferenceRequest, ResultStatus


def make_request(*, session: str, sequence_no: int = 1) -> InferenceRequest:
    return InferenceRequest.with_timeout(
        session_id=session,
        model="public-policy",
        observation={"state": [1.0, 2.0]},
        timeout_ms=100,
        sequence_no=sequence_no,
    )


async def test_http_backend_batches_and_preserves_action_identity() -> None:
    seen = []

    async def handle(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        seen.append(payload)
        return httpx.Response(
            200,
            json={
                "actions": [
                    {
                        "request_id": item["request_id"],
                        "session_id": item["session_id"],
                        "model": item["model"],
                        "sequence_no": item["sequence_no"],
                        "actions": [[0.1, 0.2]],
                        "metadata": {"server_total_ms": 4.0},
                    }
                    for item in payload["requests"]
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    backend = HttpJsonBackend("http://policy.test/infer", client=client)
    requests = [make_request(session="robot-a"), make_request(session="robot-b")]
    async with Scheduler(
        backend, SchedulerConfig(max_batch_wait_ms=5, dispatch_guard_ms=0)
    ) as scheduler:
        futures = [await scheduler.enqueue(request) for request in requests]
        outcomes = [await future for future in futures]
    await client.aclose()

    assert len(seen) == 1
    assert len(seen[0]["requests"]) == 2
    assert all(outcome.status is ResultStatus.COMPLETED for outcome in outcomes)
    assert outcomes[0].action is not None
    assert outcomes[0].action.actions == [[0.1, 0.2]]


async def test_http_backend_schema_error_becomes_failed_outcome() -> None:
    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not_actions": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    backend = HttpJsonBackend("http://policy.test/infer", client=client)
    async with Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0)) as scheduler:
        outcome = await scheduler.submit(make_request(session="robot-a"))
    await client.aclose()

    assert outcome.status is ResultStatus.FAILED
    assert "actions array" in (outcome.error or "")


async def test_http_backend_rejects_mismatched_identity() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        item = __import__("json").loads(request.content)["requests"][0]
        return httpx.Response(
            200,
            json={
                "actions": [
                    {
                        **item,
                        "session_id": "wrong-robot",
                        "actions": [0.0],
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    backend = HttpJsonBackend("http://policy.test/infer", client=client)
    async with Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0)) as scheduler:
        outcome = await scheduler.submit(make_request(session="robot-a"))
    await client.aclose()

    assert outcome.status is ResultStatus.FAILED
    assert "identity mismatch" in (outcome.error or "")


async def test_http_backend_learns_conservative_batch_latency() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "actions": [
                    {
                        "request_id": item["request_id"],
                        "session_id": item["session_id"],
                        "model": item["model"],
                        "sequence_no": item["sequence_no"],
                        "actions": [0.0],
                    }
                    for item in payload["requests"]
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    backend = HttpJsonBackend(
        "http://policy.test/infer",
        client=client,
        latency_safety_factor=1.25,
    )
    assert backend.estimate_batch_latency_ms(1) is None
    await backend.infer_batch([make_request(session="robot-a")])
    batch_one = backend.estimate_batch_latency_ms(1)
    batch_two = backend.estimate_batch_latency_ms(2)
    await client.aclose()

    assert batch_one is not None and batch_one > 0
    assert batch_two is not None and batch_two >= batch_one * 2


def test_http_backend_can_start_with_conservative_latency_floor() -> None:
    backend = HttpJsonBackend(
        "http://policy.test/infer",
        client=object(),
        initial_latency_ms=25,
    )
    assert backend.estimate_batch_latency_ms(1) == 25
    assert backend.estimate_batch_latency_ms(8) == 25


async def test_http_latency_floor_can_reject_unserviceable_request_before_network() -> None:
    calls = 0

    async def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("unserviceable request reached HTTP backend")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    backend = HttpJsonBackend(
        "http://policy.test/infer",
        client=client,
        initial_latency_ms=50,
    )
    async with Scheduler(
        backend,
        SchedulerConfig(
            max_batch_wait_ms=0,
            drop_unserviceable_requests=True,
        ),
    ) as scheduler:
        outcome = await scheduler.submit(
            InferenceRequest.with_timeout(
                session_id="urgent",
                model="public-policy",
                observation={},
                timeout_ms=5,
                sequence_no=1,
            )
        )
    await client.aclose()

    assert outcome.status is ResultStatus.UNSERVICEABLE
    assert calls == 0
