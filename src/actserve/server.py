from contextlib import asynccontextmanager
from typing import Any

from .metrics import prometheus_text
from .scheduler import Scheduler
from .types import InferenceRequest


def create_app(scheduler: Scheduler):
    """Create the optional FastAPI server without making FastAPI a core dependency."""

    try:
        from fastapi import FastAPI, HTTPException, Response
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError("Install ActServe with the 'server' extra") from exc

    class ActionRequest(BaseModel):
        session_id: str = Field(min_length=1)
        model: str = Field(min_length=1)
        sequence_no: int = Field(ge=0)
        deadline_ms: float = Field(gt=0)
        observation: Any
        priority: int = 0
        metadata: dict[str, Any] = Field(default_factory=dict)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.close(cancel_pending=True)

    app = FastAPI(title="ActServe", version="0.4.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/metrics")
    async def metrics() -> dict[str, int | float | None]:
        return scheduler.metrics.snapshot().as_dict()

    @app.get("/metrics", response_class=Response)
    async def prometheus_metrics() -> Response:
        return Response(
            prometheus_text(scheduler.metrics.snapshot()),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/actions")
    async def actions(payload: ActionRequest) -> dict[str, Any]:
        request = InferenceRequest.with_timeout(
            session_id=payload.session_id,
            model=payload.model,
            observation=payload.observation,
            timeout_ms=payload.deadline_ms,
            sequence_no=payload.sequence_no,
            priority=payload.priority,
            metadata=payload.metadata,
        )
        outcome = await scheduler.submit(request)
        if outcome.error:
            raise HTTPException(status_code=503, detail=outcome.error)
        return {
            "request_id": request.request_id,
            "status": outcome.status.value,
            "action": None if outcome.action is None else outcome.action.actions,
            "queue_ms": outcome.queue_ms,
            "end_to_end_ms": outcome.end_to_end_ms,
            "deadline_lateness_ms": outcome.deadline_lateness_ms,
            "replaced_by_request_id": outcome.replaced_by_request_id,
        }

    return app
