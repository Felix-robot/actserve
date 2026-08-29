import httpx

from actserve import __version__
from actserve.backend import SimulatedBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.server import create_app
from actserve.types import InferenceRequest


async def test_http_action_round_trip() -> None:
    backend = SimulatedBackend(fixed_ms=0, per_item_ms=0)
    scheduler = Scheduler(backend, SchedulerConfig(max_batch_wait_ms=0))
    app = create_app(scheduler)
    await scheduler.start()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/actions",
            json={
                "session_id": "public-test-robot",
                "model": "simulated-vla",
                "sequence_no": 0,
                "deadline_ms": 100,
                "observation": {"frame": 0},
            },
        )
        metrics = await client.get("/v1/metrics")
    await scheduler.close(cancel_pending=True)
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["action"] == {"echo": {"frame": 0}}
    assert metrics.json()["completed_on_time"] == 1


async def test_server_reports_package_version_and_readiness() -> None:
    scheduler = Scheduler(SimulatedBackend(fixed_ms=0, per_item_ms=0))
    app = create_app(scheduler)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        before = await client.get("/readyz")
        health = await client.get("/healthz")
        await scheduler.start()
        after = await client.get("/readyz")
    await scheduler.close(cancel_pending=True)

    assert app.version == __version__
    assert health.json() == {"status": "ok", "version": __version__}
    assert before.status_code == 503
    assert after.json() == {"status": "ready"}


async def test_optional_bearer_auth_protects_actions_and_metrics() -> None:
    scheduler = Scheduler(SimulatedBackend(fixed_ms=0, per_item_ms=0))
    app = create_app(scheduler, api_key="public-test-token")
    await scheduler.start()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/healthz")
        denied = await client.get("/v1/metrics")
        allowed = await client.get(
            "/v1/metrics", headers={"Authorization": "Bearer public-test-token"}
        )
    await scheduler.close(cancel_pending=True)

    assert health.status_code == 200
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == "Bearer"
    assert allowed.status_code == 200


async def test_server_maps_scheduler_overload_to_retryable_http_429() -> None:
    class NeverRunsBackend:
        max_batch_size = 1

        def batch_key(self, request):
            return request.model

        async def infer_batch(self, requests):
            raise AssertionError("pending overload test must not dispatch")

    scheduler = Scheduler(
        NeverRunsBackend(),
        SchedulerConfig(max_batch_wait_ms=10_000, max_pending_requests=1),
    )
    app = create_app(scheduler)
    await scheduler.start()
    pending = await scheduler.enqueue(
        InferenceRequest.with_timeout(
            session_id="queued",
            model="test",
            observation={},
            timeout_ms=20_000,
            sequence_no=1,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/actions",
            json={
                "session_id": "overflow",
                "model": "test",
                "sequence_no": 1,
                "deadline_ms": 100,
                "observation": {},
            },
        )
    await scheduler.close(cancel_pending=True)
    await pending

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert "capacity" in response.json()["detail"]
