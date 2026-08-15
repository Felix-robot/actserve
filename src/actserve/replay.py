from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .types import RequestOutcome


def outcome_record(outcome: RequestOutcome) -> dict[str, Any]:
    """Create a JSON-safe, observation-free scheduling record.

    Observations and actions are omitted by default so traces can be shared
    without leaking camera data, instructions, or model outputs.
    """

    return {
        "schema": "actserve.outcome.v1",
        "request_id": outcome.request.request_id,
        "session_id": outcome.request.session_id,
        "model": outcome.request.model,
        "sequence_no": outcome.request.sequence_no,
        "priority": outcome.request.priority,
        "received_ns": outcome.request.received_ns,
        "deadline_ns": outcome.request.deadline_ns,
        "dispatched_ns": outcome.dispatched_ns,
        "completed_ns": outcome.completed_ns,
        "status": outcome.status.value,
        "queue_ms": outcome.queue_ms,
        "end_to_end_ms": outcome.end_to_end_ms,
        "deadline_lateness_ms": outcome.deadline_lateness_ms,
        "error": outcome.error,
        "replaced_by_request_id": outcome.replaced_by_request_id,
    }


def write_outcomes(path: str | Path, outcomes: Iterable[RequestOutcome]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome_record(outcome), sort_keys=True) + "\n")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
