import httpx

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
