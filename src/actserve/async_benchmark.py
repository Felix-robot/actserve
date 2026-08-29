from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass

from .action_queue import ActionQueueConfig, ActionQueueEmpty, AsyncActionQueue
from .types import ActionChunk


@dataclass(frozen=True, slots=True)
class LoopResult:
    actions_executed: int
    inference_calls: int
    startup_idle_ticks: int
    steady_idle_ticks: int
    total_idle_ticks: int
    wall_ms: float


async def run_synchronous(
    *, chunks: int, chunk_size: int, inference_ms: float, tick_ms: float
) -> LoopResult:
    started = time.perf_counter()
    idle_ticks = 0
    actions_executed = 0
    for _ in range(chunks):
        await asyncio.sleep(inference_ms / 1000)
        idle_ticks += math.ceil(inference_ms / tick_ms)
        for _ in range(chunk_size):
            await asyncio.sleep(tick_ms / 1000)
            actions_executed += 1
    return LoopResult(
        actions_executed=actions_executed,
        inference_calls=chunks,
        startup_idle_ticks=math.ceil(inference_ms / tick_ms),
        steady_idle_ticks=idle_ticks - math.ceil(inference_ms / tick_ms),
        total_idle_ticks=idle_ticks,
        wall_ms=(time.perf_counter() - started) * 1000,
    )


async def run_asynchronous(
    *,
    chunks: int,
    chunk_size: int,
    inference_ms: float,
    tick_ms: float,
    low_watermark: int,
) -> LoopResult:
    queue = AsyncActionQueue(
        session_id="public-sim",
        model="synthetic-policy",
        config=ActionQueueConfig(low_watermark=low_watermark, handoff_policy="append"),
    )
    total_actions = chunks * chunk_size
    producer_done = asyncio.Event()

    async def produce() -> None:
        try:
            for sequence_no in range(chunks):
                if not await queue.acquire_refill():
                    return
                await asyncio.sleep(inference_ms / 1000)
                await queue.finish_refill(
                    ActionChunk(
                        request_id=f"synthetic-{sequence_no}",
                        session_id="public-sim",
                        sequence_no=sequence_no,
                        actions=list(range(chunk_size)),
                        model="synthetic-policy",
                    )
                )
        finally:
            producer_done.set()

    started = time.perf_counter()
    producer = asyncio.create_task(produce())
    startup_idle_ticks = 0
    steady_idle_ticks = 0
    actions_executed = 0
    first_action_seen = False
    while actions_executed < total_actions:
        await asyncio.sleep(tick_ms / 1000)
        try:
            await queue.get_nowait()
        except ActionQueueEmpty:
            if first_action_seen:
                steady_idle_ticks += 1
            else:
                startup_idle_ticks += 1
        else:
            first_action_seen = True
            actions_executed += 1
        if producer_done.is_set() and queue.remaining_actions == 0:
            break

    await producer
    await queue.close()
    return LoopResult(
        actions_executed=actions_executed,
        inference_calls=chunks,
        startup_idle_ticks=startup_idle_ticks,
        steady_idle_ticks=steady_idle_ticks,
        total_idle_ticks=startup_idle_ticks + steady_idle_ticks,
        wall_ms=(time.perf_counter() - started) * 1000,
    )


async def benchmark(args: argparse.Namespace) -> dict[str, object]:
    synchronous = await run_synchronous(
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        inference_ms=args.inference_ms,
        tick_ms=args.tick_ms,
    )
    asynchronous = await run_asynchronous(
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        inference_ms=args.inference_ms,
        tick_ms=args.tick_ms,
        low_watermark=args.low_watermark,
    )
    return {
        "schema": "actserve.async_benchmark.v1",
        "scope": "synthetic timing benchmark; not closed-loop task success",
        "config": {
            "chunks": args.chunks,
            "chunk_size": args.chunk_size,
            "inference_ms": args.inference_ms,
            "tick_ms": args.tick_ms,
            "low_watermark": args.low_watermark,
        },
        "synchronous": asdict(synchronous),
        "asynchronous": asdict(asynchronous),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actserve benchmark-async",
        description="Compare blocking and asynchronously refilled synthetic action loops.",
    )
    parser.add_argument("--chunks", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--inference-ms", type=float, default=80)
    parser.add_argument("--tick-ms", type=float, default=20)
    parser.add_argument("--low-watermark", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunks <= 0 or args.chunk_size <= 0:
        parser.error("chunks and chunk-size must be positive")
    if args.inference_ms <= 0 or args.tick_ms <= 0:
        parser.error("inference-ms and tick-ms must be positive")
    if not 0 <= args.low_watermark < args.chunk_size:
        parser.error("low-watermark must be in [0, chunk-size)")
    print(json.dumps(asyncio.run(benchmark(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
