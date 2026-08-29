"""Live FIFO-vs-ActServe benchmark for an Embodied.cpp VLA server.

This uses synthetic observations and a public checkpoint. It measures serving
freshness under overload; it is not a closed-loop robotics task evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from actserve.integrations import EmbodiedCppVlaBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import InferenceRequest, ResultStatus


@dataclass(frozen=True, slots=True)
class TraceItem:
    session_id: str
    sequence_no: int
    release_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto-python-dir", type=Path, required=True)
    parser.add_argument("--address", default="tcp://127.0.0.1:5592")
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--arrival-ms", type=float, default=5.0)
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=0.0,
        help="zero chooses 6x the measured warmup latency",
    )
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=32)
    return parser.parse_args()


def load_proto(directory: Path) -> Any:
    sys.path.insert(0, str(directory.resolve()))
    return importlib.import_module("serving.vla_pb2")


def make_observation(args: argparse.Namespace) -> dict[str, Any]:
    image = np.zeros((args.image_size, args.image_size, 3), dtype=np.float32)
    return {
        "image": image.tobytes(),
        "height": args.image_size,
        "width": args.image_size,
        "tokens": [1],
        "state": [0.0] * args.state_dim,
        "noise": [0.0] * (args.chunk_size * args.action_dim),
    }


def build_message(pb: Any, observation: dict[str, Any]) -> Any:
    message = pb.PredictRequest()
    image = message.images.add()
    image.encoding = pb.Image.F32_RGB_01
    image.height = observation["height"]
    image.width = observation["width"]
    image.data = observation["image"]
    message.lang_tokens.extend(observation["tokens"])
    message.state.extend(observation["state"])
    message.noise.extend(observation["noise"])
    return message


def request_once(
    socket: Any, pb: Any, observation: dict[str, Any], request_id: int
) -> tuple[float, np.ndarray[Any, np.dtype[np.float32]]]:
    message = build_message(pb, observation)
    message.request_id = request_id
    started = time.monotonic_ns()
    socket.send(message.SerializeToString())
    body = socket.recv()
    elapsed_ms = (time.monotonic_ns() - started) / 1_000_000
    response = pb.PredictResponse()
    response.ParseFromString(body)
    if response.error:
        raise RuntimeError(response.error)
    if int(response.request_id) != request_id:
        raise RuntimeError("Embodied.cpp response request_id mismatch")
    return elapsed_ms, np.asarray(response.action_chunk, dtype=np.float32)


def make_socket(address: str, timeout_ms: int) -> Any:
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(address)
    return socket


def make_trace(sessions: int, frames: int, arrival_ms: float) -> list[TraceItem]:
    return [
        TraceItem(f"robot-{session}", frame, (frame * sessions + session) * arrival_ms)
        for frame in range(frames)
        for session in range(sessions)
    ]


async def wait_until(start_ns: int, release_ms: float) -> None:
    remaining = start_ns + int(release_ms * 1_000_000) - time.monotonic_ns()
    if remaining > 0:
        await asyncio.sleep(remaining / 1_000_000_000)


async def run_direct(
    args: argparse.Namespace,
    pb: Any,
    observation: dict[str, Any],
    trace: list[TraceItem],
    deadline_ms: float,
) -> tuple[dict[tuple[str, int], np.ndarray[Any, np.dtype[np.float32]]], dict[str, Any]]:
    socket = make_socket(args.address, args.timeout_ms)
    start_ns = time.monotonic_ns()
    actions: dict[tuple[str, int], np.ndarray[Any, np.dtype[np.float32]]] = {}
    latencies: list[float] = []
    on_time = 0
    latest_on_time = 0
    try:
        for index, item in enumerate(trace, start=10_000):
            await wait_until(start_ns, item.release_ms)
            latency_ms, action = await asyncio.to_thread(
                request_once, socket, pb, observation, index
            )
            latencies.append(latency_ms)
            actions[(item.session_id, item.sequence_no)] = action
            deadline_ns = start_ns + int((item.release_ms + deadline_ms) * 1_000_000)
            met = time.monotonic_ns() <= deadline_ns
            on_time += int(met)
            latest_on_time += int(met and item.sequence_no == args.frames - 1)
    finally:
        socket.close(linger=0)
    return actions, {
        "dispatched": len(trace),
        "on_time": on_time,
        "latest_frame_on_time": latest_on_time,
        "p50_request_ms": statistics.median(latencies),
        "p95_request_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
    }


async def run_actserve(
    args: argparse.Namespace,
    pb: Any,
    observation: dict[str, Any],
    trace: list[TraceItem],
    deadline_ms: float,
    initial_latency_ms: float,
    direct_actions: dict[
        tuple[str, int], np.ndarray[Any, np.dtype[np.float32]]
    ],
    parity_tolerance: float,
) -> dict[str, Any]:
    backend = EmbodiedCppVlaBackend(
        protobuf_module=pb,
        request_builder=lambda request: build_message(pb, request.observation),
        address=args.address,
        timeout_ms=args.timeout_ms,
        initial_latency_ms=initial_latency_ms,
    )
    config = SchedulerConfig(
        policy="edf",
        max_batch_wait_ms=1.0,
        coalesce_sessions=True,
        drop_unserviceable_requests=True,
    )
    start_ns = time.monotonic_ns()
    futures = []
    requests = []
    async with backend:
        async with Scheduler(backend, config) as scheduler:
            for item in trace:
                await wait_until(start_ns, item.release_ms)
                received_ns = time.monotonic_ns()
                request = InferenceRequest(
                    session_id=item.session_id,
                    model="public-pi05",
                    observation=observation,
                    deadline_ns=start_ns
                    + int((item.release_ms + deadline_ms) * 1_000_000),
                    sequence_no=item.sequence_no,
                    received_ns=received_ns,
                )
                requests.append(request)
                futures.append(await scheduler.enqueue(request))
            outcomes = await asyncio.gather(*futures)

    counts = {status.value: 0 for status in ResultStatus}
    parity_failures = 0
    max_action_delta = 0.0
    latest_on_time = 0
    for request, outcome in zip(requests, outcomes, strict=True):
        counts[outcome.status.value] += 1
        if outcome.status is ResultStatus.COMPLETED and outcome.action is not None:
            latest_on_time += int(request.sequence_no == args.frames - 1)
            expected = direct_actions[(request.session_id, request.sequence_no)]
            observed = np.asarray(outcome.action.actions, dtype=np.float32).reshape(-1)
            delta = float(np.max(np.abs(observed - expected)))
            max_action_delta = max(max_action_delta, delta)
            parity_failures += int(delta > parity_tolerance)
    return {
        "dispatched": sum(outcome.dispatched_ns is not None for outcome in outcomes),
        "on_time": counts[ResultStatus.COMPLETED.value],
        "latest_frame_on_time": latest_on_time,
        "parity_failures": parity_failures,
        "max_action_delta": max_action_delta,
        "parity_tolerance": parity_tolerance,
        "statuses": counts,
    }


async def main() -> None:
    args = parse_args()
    if args.sessions < 1 or args.frames < 1 or args.arrival_ms < 0:
        raise SystemExit("sessions/frames must be positive and arrival-ms non-negative")
    pb = load_proto(args.proto_python_dir)
    observation = make_observation(args)
    socket = make_socket(args.address, args.timeout_ms)
    try:
        warmup_results = [
            request_once(socket, pb, observation, request_id)
            for request_id in range(1, 6)
        ]
    finally:
        socket.close(linger=0)
    warmups = [result[0] for result in warmup_results]
    warmup_actions = [result[1] for result in warmup_results]
    direct_repeat_max_delta = max(
        float(np.max(np.abs(right - left)))
        for index, left in enumerate(warmup_actions)
        for right in warmup_actions[index + 1 :]
    )
    parity_tolerance = max(1e-5, direct_repeat_max_delta * 1.5)
    warmup_ms = statistics.median(warmups)
    deadline_ms = args.deadline_ms or warmup_ms * 6
    trace = make_trace(args.sessions, args.frames, args.arrival_ms)
    direct_actions, direct = await run_direct(
        args, pb, observation, trace, deadline_ms
    )
    actserve = await run_actserve(
        args,
        pb,
        observation,
        trace,
        deadline_ms,
        warmup_ms,
        direct_actions,
        parity_tolerance,
    )
    print(
        json.dumps(
            {
                "scope": "synthetic serving freshness; not closed-loop task success",
                "trace_requests": len(trace),
                "deadline_ms": deadline_ms,
                "warmup_ms": warmups,
                "direct_repeat_max_action_delta": direct_repeat_max_delta,
                "direct_fifo": direct,
                "actserve": actserve,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
