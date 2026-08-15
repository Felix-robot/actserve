from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass

from .backend import SimulatedBackend
from .scheduler import Scheduler, SchedulerConfig
from .types import InferenceRequest, RequestOutcome


@dataclass(frozen=True, slots=True)
class Workload:
    sessions: int = 8
    observations_per_session: int = 20
    observation_hz: float = 20.0
    deadline_ms: float = 100.0
    fixed_backend_ms: float = 25.0
    per_item_backend_ms: float = 2.0


async def run_case(name: str, workload: Workload, config: SchedulerConfig) -> dict:
    backend = SimulatedBackend(
        fixed_ms=workload.fixed_backend_ms,
        per_item_ms=workload.per_item_backend_ms,
        max_batch_size=8,
    )
    outcomes: list[RequestOutcome] = []
    start = time.monotonic()
    async with Scheduler(backend, config) as scheduler:
        tasks: list[asyncio.Task[RequestOutcome]] = []
        interval = 1 / workload.observation_hz
        for sequence_no in range(workload.observations_per_session):
            tick = time.monotonic()
            for session_no in range(workload.sessions):
                request = InferenceRequest.with_timeout(
                    session_id=f"robot-{session_no}",
                    model="public-simulated-vla",
                    observation={"frame": sequence_no},
                    timeout_ms=workload.deadline_ms,
                    sequence_no=sequence_no,
                )
                tasks.append(asyncio.create_task(scheduler.submit(request)))
            sleep_for = interval - (time.monotonic() - tick)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        outcomes.extend(await asyncio.gather(*tasks))
        snapshot = scheduler.metrics.snapshot()
    result = snapshot.as_dict()
    result.update(
        {
            "case": name,
            "wall_seconds": time.monotonic() - start,
            "backend_calls": len(backend.calls),
            "max_observed_batch": max((len(call) for call in backend.calls), default=0),
        }
    )
    return result


async def compare(workload: Workload) -> list[dict]:
    baseline = await run_case(
        "fifo_batch1",
        workload,
        SchedulerConfig(
            policy="fifo",
            max_batch_wait_ms=0,
            max_batch_size=1,
            coalesce_sessions=False,
        ),
    )
    optimized = await run_case(
        "edf_coalescing_microbatch",
        workload,
        SchedulerConfig(
            policy="edf",
            max_batch_wait_ms=2,
            dispatch_guard_ms=2,
            max_batch_size=8,
            coalesce_sessions=True,
        ),
    )
    return [baseline, optimized]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare FIFO and ActServe scheduling")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--observations", type=int, default=20)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--fixed-backend-ms", type=float, default=25.0)
    parser.add_argument("--per-item-backend-ms", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workload = Workload(
        sessions=args.sessions,
        observations_per_session=args.observations,
        observation_hz=args.hz,
        deadline_ms=args.deadline_ms,
        fixed_backend_ms=args.fixed_backend_ms,
        per_item_backend_ms=args.per_item_backend_ms,
    )
    print(json.dumps(asyncio.run(compare(workload)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
