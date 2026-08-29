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
class SerialWorkload:
    sessions: int = 8
    observations_per_session: int = 12
    observation_hz: float = 30.0
    deadline_ms: float = 80.0
    backend_latency_ms: float = 15.0


async def _run_case(
    name: str, workload: SerialWorkload, config: SchedulerConfig
) -> dict[str, int | float | str | None]:
    backend = SimulatedBackend(
        fixed_ms=workload.backend_latency_ms,
        per_item_ms=0,
        max_batch_size=1,
    )
    started = time.monotonic()
    async with Scheduler(backend, config) as scheduler:
        tasks: list[asyncio.Task[RequestOutcome]] = []
        interval = 1 / workload.observation_hz
        for sequence_no in range(workload.observations_per_session):
            tick = time.monotonic()
            for session_no in range(workload.sessions):
                item = InferenceRequest.with_timeout(
                    session_id=f"robot-{session_no}",
                    model="embodied-cpp-serial-vla",
                    observation={"frame": sequence_no},
                    timeout_ms=workload.deadline_ms,
                    sequence_no=sequence_no,
                )
                tasks.append(asyncio.create_task(scheduler.submit(item)))
            remaining = interval - (time.monotonic() - tick)
            if remaining > 0:
                await asyncio.sleep(remaining)
        await asyncio.gather(*tasks)
        result = scheduler.metrics.snapshot().as_dict()
    result.update(
        {
            "case": name,
            "backend_calls": len(backend.calls),
            "wall_seconds": time.monotonic() - started,
        }
    )
    return result


async def compare(workload: SerialWorkload) -> dict:
    direct = await _run_case(
        "embodied_cpp_direct_fifo",
        workload,
        SchedulerConfig(
            policy="fifo",
            max_batch_wait_ms=0,
            max_batch_size=1,
            coalesce_sessions=False,
            drop_unserviceable_requests=False,
        ),
    )
    controlled = await _run_case(
        "actserve_plus_embodied_cpp",
        workload,
        SchedulerConfig(
            policy="edf",
            max_batch_wait_ms=0,
            dispatch_guard_ms=2,
            max_batch_size=1,
            coalesce_sessions=True,
            drop_unserviceable_requests=True,
        ),
    )
    return {
        "schema": "actserve.serial_runtime_benchmark.v1",
        "evidence": "scheduler-only serial-runtime simulation; no model-quality claim",
        "workload": {
            "sessions": workload.sessions,
            "observations_per_session": workload.observations_per_session,
            "observation_hz": workload.observation_hz,
            "deadline_ms": workload.deadline_ms,
            "backend_latency_ms": workload.backend_latency_ms,
        },
        "results": [direct, controlled],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare direct FIFO with ActServe in front of a serial embodied runtime"
    )
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--observations", type=int, default=12)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument("--backend-latency-ms", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workload = SerialWorkload(
        sessions=args.sessions,
        observations_per_session=args.observations,
        observation_hz=args.hz,
        deadline_ms=args.deadline_ms,
        backend_latency_ms=args.backend_latency_ms,
    )
    print(json.dumps(asyncio.run(compare(workload)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
