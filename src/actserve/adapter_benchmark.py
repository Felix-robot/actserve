from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass

from .adapter_backend import AdapterBackend, AdapterRoute
from .scheduler import Scheduler, SchedulerConfig
from .types import ActionChunk, InferenceRequest, ResultStatus


@dataclass(frozen=True, slots=True)
class AdapterArmResult:
    backend_calls: int
    mean_batch_size: float
    completed: int
    wall_ms: float


async def run_arm(
    *,
    adapters: int,
    sessions_per_adapter: int,
    max_batch_size: int,
    fixed_ms: float,
    per_item_ms: float,
    mixed_adapter_batch: bool,
) -> AdapterArmResult:
    calls: list[int] = []

    async def infer(routed):
        calls.append(len(routed))
        await asyncio.sleep((fixed_ms + per_item_ms * len(routed)) / 1000)
        return [
            ActionChunk(
                request_id=item.request.request_id,
                session_id=item.request.session_id,
                sequence_no=item.request.sequence_no,
                actions=[item.route.adapter],
                model=item.request.model,
            )
            for item in routed
        ]

    routes = [
        AdapterRoute(f"task-{index}", "shared-public-backbone", f"adapter-{index}")
        for index in range(adapters)
    ]
    backend = AdapterBackend(
        infer,
        routes,
        max_batch_size=max_batch_size,
        mixed_adapter_batch=mixed_adapter_batch,
    )
    requests = [
        InferenceRequest.with_timeout(
            session_id=f"task-{adapter}-session-{session}",
            model=f"task-{adapter}",
            observation={"synthetic": True},
            timeout_ms=5000,
            sequence_no=1,
        )
        for adapter in range(adapters)
        for session in range(sessions_per_adapter)
    ]
    started = time.perf_counter()
    async with Scheduler(
        backend,
        SchedulerConfig(max_batch_wait_ms=5, dispatch_guard_ms=0),
    ) as scheduler:
        futures = [await scheduler.enqueue(request) for request in requests]
        outcomes = [await future for future in futures]
    return AdapterArmResult(
        backend_calls=len(calls),
        mean_batch_size=sum(calls) / len(calls),
        completed=sum(outcome.status is ResultStatus.COMPLETED for outcome in outcomes),
        wall_ms=(time.perf_counter() - started) * 1000,
    )


async def benchmark(args: argparse.Namespace) -> dict[str, object]:
    isolated = await run_arm(
        adapters=args.adapters,
        sessions_per_adapter=args.sessions_per_adapter,
        max_batch_size=args.max_batch_size,
        fixed_ms=args.fixed_ms,
        per_item_ms=args.per_item_ms,
        mixed_adapter_batch=False,
    )
    shared = await run_arm(
        adapters=args.adapters,
        sessions_per_adapter=args.sessions_per_adapter,
        max_batch_size=args.max_batch_size,
        fixed_ms=args.fixed_ms,
        per_item_ms=args.per_item_ms,
        mixed_adapter_batch=True,
    )
    isolated_memory_mb = args.adapters * (args.backbone_mb + args.adapter_mb)
    shared_memory_mb = args.backbone_mb + args.adapters * args.adapter_mb
    return {
        "schema": "actserve.adapter_benchmark.v1",
        "scope": "synthetic routing and memory model; not real-model or task success",
        "config": {
            "adapters": args.adapters,
            "sessions_per_adapter": args.sessions_per_adapter,
            "max_batch_size": args.max_batch_size,
            "fixed_ms": args.fixed_ms,
            "per_item_ms": args.per_item_ms,
            "backbone_mb": args.backbone_mb,
            "adapter_mb": args.adapter_mb,
        },
        "isolated_models": {
            **asdict(isolated),
            "modeled_memory_mb": isolated_memory_mb,
        },
        "shared_backbone": {
            **asdict(shared),
            "modeled_memory_mb": shared_memory_mb,
        },
        "modeled_memory_reduction_percent": (
            (isolated_memory_mb - shared_memory_mb) / isolated_memory_mb * 100
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actserve benchmark-adapters",
        description="Benchmark isolated adapter routing against shared-backbone batching.",
    )
    parser.add_argument("--adapters", type=int, default=4)
    parser.add_argument("--sessions-per-adapter", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--fixed-ms", type=float, default=20)
    parser.add_argument("--per-item-ms", type=float, default=2)
    parser.add_argument("--backbone-mb", type=float, default=7000)
    parser.add_argument("--adapter-mb", type=float, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    numeric = (
        args.adapters,
        args.sessions_per_adapter,
        args.max_batch_size,
        args.fixed_ms,
        args.per_item_ms,
        args.backbone_mb,
        args.adapter_mb,
    )
    if any(value <= 0 for value in numeric):
        parser.error("all benchmark parameters must be positive")
    print(json.dumps(asyncio.run(benchmark(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
