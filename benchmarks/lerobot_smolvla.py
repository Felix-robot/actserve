"""Real-model SmolVLA batching benchmark with synthetic public observations.

This script deliberately lives outside the core package. Run it in an isolated
LeRobot environment; it downloads only the named public checkpoint and never
uses robot data or sends actions to hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import statistics
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from actserve.backend import CallableBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import ActionChunk, InferenceRequest, ResultStatus

DEFAULT_MODEL_ID = "lerobot/smolvla_base"
DEFAULT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="mps")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-wait-ms", type=float, default=10)
    parser.add_argument("--deadline-ms", type=float, default=120_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--parity-atol", type=float, default=0.01)
    parser.add_argument(
        "--hardware-label",
        help="Human-readable hardware label recorded in the result (for example, 'Apple M5 16GB')",
    )
    parser.add_argument("--output", type=Path)
    return parser


async def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy
    except ImportError as exc:
        raise RuntimeError("run this benchmark in an environment with lerobot[smolvla]") from exc

    if args.sessions < 2 or args.repetitions < 1 or args.warmup < 0:
        raise ValueError("sessions must be >=2, repetitions >=1, and warmup >=0")
    if args.parity_atol < 0 or args.deadline_ms <= 0:
        raise ValueError("parity-atol must be non-negative and deadline-ms positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        device_name = "Apple Metal Performance Shaders"
    else:
        device_name = platform.processor() or platform.machine()

    load_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.model_id, revision=args.revision)
    policy.to(device)
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model_id,
        pretrained_revision=args.revision,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    load_seconds = time.perf_counter() - load_started
    if policy.config.n_obs_steps != 1:
        raise RuntimeError("benchmark requires n_obs_steps=1 for dynamic session batching")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    image_features = sorted(policy.config.image_features)
    state_dim = policy.config.input_features["observation.state"].shape[0]
    observations: list[dict[str, Any]] = []
    noises = []
    for index in range(args.sessions):
        observation: dict[str, Any] = {
            "observation.state": torch.rand(1, state_dim, generator=generator),
            "task": f"Move public object variant {index} to the public target.",
        }
        for key in image_features:
            shape = policy.config.input_features[key].shape
            observation[key] = torch.rand(1, *shape, generator=generator)
        observations.append(observation)
        noises.append(
            torch.randn(
                1,
                policy.config.chunk_size,
                policy.config.max_action_dim,
                generator=generator,
            )
        )

    def make_requests() -> list[InferenceRequest]:
        return [
            InferenceRequest.with_timeout(
                session_id=f"public-session-{index}",
                model=args.model_id,
                observation=observation,
                timeout_ms=args.deadline_ms,
                sequence_no=0,
                metadata={"input_signature": "smolvla-base-public-v1", "sample_index": index},
            )
            for index, observation in enumerate(observations)
        ]

    def stack_observations(requests: Sequence[InferenceRequest]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "observation.state": torch.cat(
                [request.observation["observation.state"].clone() for request in requests]
            ),
            "task": [request.observation["task"] for request in requests],
        }
        for key in image_features:
            batch[key] = torch.cat([request.observation[key].clone() for request in requests])
        return batch

    backend_calls = 0

    async def infer(requests: Sequence[InferenceRequest]) -> Sequence[ActionChunk]:
        nonlocal backend_calls
        backend_calls += 1
        batch = preprocess(stack_observations(requests))
        noise = torch.cat(
            [noises[int(request.metadata["sample_index"])] for request in requests]
        ).to(device)
        policy.reset()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch, noise=noise)
            decoded = postprocess(actions).detach().cpu()
        return [
            ActionChunk(
                request_id=request.request_id,
                session_id=request.session_id,
                sequence_no=request.sequence_no,
                actions=decoded[index].tolist(),
                model=request.model,
            )
            for index, request in enumerate(requests)
        ]

    for _ in range(args.warmup):
        requests = make_requests()
        for request in requests:
            await infer([request])
        await infer(requests)
        _synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    direct_wall_ms = []
    actserve_wall_ms = []
    parity_rmse = []
    parity_max_abs = []
    direct_backend_calls = []
    actserve_backend_calls = []
    measurement_order = []

    async def measure_direct() -> tuple[float, int, list[ActionChunk]]:
        requests = make_requests()
        before_calls = backend_calls
        _synchronize(torch, device)
        started = time.perf_counter()
        chunks = []
        for request in requests:
            chunks.extend(await infer([request]))
        _synchronize(torch, device)
        return (time.perf_counter() - started) * 1000, backend_calls - before_calls, chunks

    async def measure_actserve() -> tuple[float, int, list[Any]]:
        requests = make_requests()
        before_calls = backend_calls
        backend = CallableBackend(infer, max_batch_size=args.sessions)
        _synchronize(torch, device)
        started = time.perf_counter()
        async with Scheduler(
            backend,
            SchedulerConfig(
                max_batch_size=args.sessions,
                max_batch_wait_ms=args.batch_wait_ms,
                dispatch_guard_ms=0,
                coalesce_sessions=False,
            ),
        ) as scheduler:
            outcomes = await asyncio.gather(*(scheduler.submit(request) for request in requests))
        _synchronize(torch, device)
        if any(outcome.status is not ResultStatus.COMPLETED for outcome in outcomes):
            raise RuntimeError(f"ActServe outcomes were not all completed: {outcomes}")
        return (time.perf_counter() - started) * 1000, backend_calls - before_calls, outcomes

    for repetition in range(args.repetitions):
        if repetition % 2 == 0:
            direct_ms, direct_calls, direct_chunks = await measure_direct()
            actserve_ms, actserve_calls, outcomes = await measure_actserve()
            measurement_order.append("direct_then_actserve")
        else:
            actserve_ms, actserve_calls, outcomes = await measure_actserve()
            direct_ms, direct_calls, direct_chunks = await measure_direct()
            measurement_order.append("actserve_then_direct")
        direct_wall_ms.append(direct_ms)
        direct_backend_calls.append(direct_calls)
        actserve_wall_ms.append(actserve_ms)
        actserve_backend_calls.append(actserve_calls)

        for direct, outcome in zip(direct_chunks, outcomes, strict=True):
            assert outcome.action is not None
            direct_tensor = torch.tensor(direct.actions)
            batched_tensor = torch.tensor(outcome.action.actions)
            delta = direct_tensor - batched_tensor
            parity_rmse.append(float(torch.sqrt(torch.mean(delta.square()))))
            parity_max_abs.append(float(torch.max(torch.abs(delta))))

    accelerator_memory: dict[str, float] = {}
    if device.type == "cuda":
        accelerator_memory["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        accelerator_memory["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
    elif device.type == "mps":
        accelerator_memory["current_allocated_mb"] = torch.mps.current_allocated_memory() / 2**20
        accelerator_memory["driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 2**20

    paired_speedups = [
        direct_ms / actserve_ms
        for direct_ms, actserve_ms in zip(direct_wall_ms, actserve_wall_ms, strict=True)
    ]
    result = {
        "schema": "actserve.lerobot_smolvla_benchmark.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "real public model with synthetic observations; not closed-loop task success",
        "measurement": "steady state after batch-1 and full-batch shape warmup",
        "model": {"id": args.model_id, "revision": args.revision},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lerobot": version("lerobot"),
            "actserve": version("actserve"),
            "device": str(device),
            "device_name": device_name,
            "hardware_label": args.hardware_label,
            "accelerator_memory_snapshot": accelerator_memory,
        },
        "config": {
            "sessions": args.sessions,
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "warmup_batch_sizes": [1, args.sessions],
            "batch_wait_ms": args.batch_wait_ms,
            "deadline_ms": args.deadline_ms,
            "seed": args.seed,
            "explicit_noise": True,
            "measurement_order": measurement_order,
        },
        "load_seconds": load_seconds,
        "direct_serial": {
            **_summary(direct_wall_ms),
            "samples_ms": direct_wall_ms,
            "backend_calls": direct_backend_calls,
        },
        "actserve_batched": {
            **_summary(actserve_wall_ms),
            "samples_ms": actserve_wall_ms,
            "backend_calls": actserve_backend_calls,
        },
        "speedup_mean": statistics.mean(direct_wall_ms) / statistics.mean(actserve_wall_ms),
        "paired_speedup": {
            "mean": statistics.mean(paired_speedups),
            "median": statistics.median(paired_speedups),
            "min": min(paired_speedups),
            "max": max(paired_speedups),
            "samples": paired_speedups,
        },
        "parity": {
            "rmse_max": max(parity_rmse),
            "max_abs": max(parity_max_abs),
            "atol": args.parity_atol,
            "passed": max(parity_max_abs) <= args.parity_atol,
        },
    }
    if not result["parity"]["passed"]:
        raise RuntimeError(f"batched action parity failed: {result['parity']}")
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
