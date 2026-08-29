import httpx

from actserve import __version__
from actserve.backend import SimulatedBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.server import create_app


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
