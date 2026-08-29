"""Minimal JSON policy endpoint for the standalone ActServe smoke test.

Run in one terminal:
    uv run uvicorn examples.http_policy_server:app --port 9000

This public example echoes the observation as an action. It is not a robot
policy and must never be connected to hardware.
"""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


class PolicyRequest(BaseModel):
    request_id: str
    session_id: str
    model: str
    sequence_no: int = Field(ge=0)
    observation: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyBatch(BaseModel):
    requests: list[PolicyRequest] = Field(min_length=1)


app = FastAPI(title="ActServe example policy")


@app.post("/infer")
async def infer(payload: PolicyBatch) -> dict[str, list[dict[str, Any]]]:
    return {
        "actions": [
            {
                "request_id": request.request_id,
                "session_id": request.session_id,
                "model": request.model,
                "sequence_no": request.sequence_no,
                "actions": {"echo": request.observation},
                "metadata": {"example": True},
            }
            for request in payload.requests
        ]
    }
