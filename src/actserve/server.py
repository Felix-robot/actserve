import hmac
import inspect
from contextlib import asynccontextmanager
from typing import Any

from . import __version__
from .metrics import prometheus_text
from .scheduler import Scheduler
from .types import InferenceRequest


def create_app(scheduler: Scheduler, *, api_key: str | None = None):
    """Create the optional FastAPI server without making FastAPI a core dependency."""

    if api_key == "":
        raise ValueError("api_key must be non-empty when authentication is enabled")

    try:
        from fastapi import FastAPI, Header, HTTPException, Response
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
            close_backend = getattr(scheduler.backend, "aclose", None)
            if close_backend is not None:
                result = close_backend()
                if inspect.isawaitable(result):
                    await result

    app = FastAPI(title="ActServe", version=__version__, lifespan=lifespan)

    def authorize(authorization: str | None) -> None:
        if api_key is None:
            return
        expected = f"Bearer {api_key}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        if not scheduler.is_accepting:
            raise HTTPException(status_code=503, detail="scheduler is not accepting requests")
        return {"status": "ready"}

    @app.get("/v1/metrics")
    async def metrics(
        authorization: str | None = Header(default=None),
    ) -> dict[str, int | float | None]:
        authorize(authorization)
        return scheduler.metrics.snapshot().as_dict()

    @app.get("/metrics", response_class=Response)
    async def prometheus_metrics(
        authorization: str | None = Header(default=None),
    ) -> Response:
        authorize(authorization)
        return Response(
            prometheus_text(scheduler.metrics.snapshot()),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/actions")
    async def actions(
        payload: ActionRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
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
