from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

from .scheduler import Scheduler, SchedulerConfig
from .types import ActionChunk, InferenceRequest, RequestOutcome


def _build_policy(torch: Any, *, width: int, layers: int, action_dim: int):
    nn = torch.nn

    class SyntheticVisionPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Conv2d(3, width, kernel_size=16, stride=16)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=8,
                dim_feedforward=width * 4,
                dropout=0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, action_dim)

        def forward(self, images):
            tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
            tokens = self.blocks(tokens)
            return self.head(self.norm(tokens.mean(dim=1)))

    return SyntheticVisionPolicy()


class TorchVisionBackend:
    """Public synthetic CUDA backend for measuring real batching behavior."""

    def __init__(
        self,
        *,
        device: str,
        max_batch_size: int,
        image_size: int,
        width: int,
        layers: int,
        action_dim: int = 64,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - requires optional CUDA environment
            raise RuntimeError(
                "The CUDA benchmark requires a CUDA-enabled PyTorch install"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.torch = torch
        self.device = torch.device(device)
        self.image_size = image_size
        self._max_batch_size = max_batch_size
        self.model = _build_policy(
            torch,
            width=width,
            layers=layers,
            action_dim=action_dim,
        ).to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()
        self.calls = 0
        self.items = 0
        self.latency_table_ms: dict[int, float] = {}

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def batch_key(self, request: InferenceRequest) -> str:
        return request.model

    def estimate_batch_latency_ms(self, batch_size: int) -> float | None:
        return self.latency_table_ms.get(batch_size)

    def calibrate(self, *, repetitions: int = 7) -> dict[int, float]:
        torch = self.torch
        table: dict[int, float] = {}
        for batch_size in range(1, self.max_batch_size + 1):
            print(
                f"[actserve] calibrating batch_size={batch_size}",
                file=sys.stderr,
                flush=True,
            )
            host = torch.randn(
                batch_size,
                3,
                self.image_size,
                self.image_size,
                dtype=torch.float32,
            ).pin_memory()
            timings = []
            for iteration in range(repetitions + 3):
                torch.cuda.synchronize(self.device)
                started = time.perf_counter_ns()
                images = host.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                with torch.inference_mode():
                    output = self.model(images)
                output.float().cpu()
                torch.cuda.synchronize(self.device)
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                if iteration >= 3:
                    timings.append(elapsed_ms)
            # A conservative estimate avoids admitting a batch on a median-only promise.
            table[batch_size] = statistics.quantiles(timings, n=10)[-1] * 1.10
        self.latency_table_ms = table
        return table

    async def infer_batch(self, requests):
        torch = self.torch
        self.calls += 1
        self.items += len(requests)
        host = torch.stack([request.observation for request in requests])
        images = host.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
        with torch.inference_mode():
            output = self.model(images)
        actions = output.float().cpu()
        torch.cuda.synchronize(self.device)
        return [
            ActionChunk(
                request_id=request.request_id,
                session_id=request.session_id,
                sequence_no=request.sequence_no,
                actions=actions[index].tolist(),
                model=request.model,
            )
            for index, request in enumerate(requests)
        ]


@dataclass(frozen=True, slots=True)
class CudaWorkload:
    sessions: int = 16
    observations_per_session: int = 20
    observation_hz: float = 30.0
    deadline_ms: float = 100.0


async def _run_case(
    name: str,
    backend: TorchVisionBackend,
    workload: CudaWorkload,
    frames: list[Any],
    config: SchedulerConfig,
    timeout_seconds: float,
) -> dict[str, Any]:
    backend.calls = 0
    backend.items = 0
    started = time.monotonic()
    print(f"[actserve] starting case={name}", file=sys.stderr, flush=True)
    async with Scheduler(backend, config) as scheduler:
        tasks: list[asyncio.Task[RequestOutcome]] = []
        interval = 1 / workload.observation_hz
        for sequence_no in range(workload.observations_per_session):
            tick = time.monotonic()
            for session_no in range(workload.sessions):
                request = InferenceRequest.with_timeout(
                    session_id=f"robot-{session_no}",
                    model="synthetic-vit-policy",
                    observation=frames[session_no],
                    timeout_ms=workload.deadline_ms,
                    sequence_no=sequence_no,
                )
                tasks.append(asyncio.create_task(scheduler.submit(request)))
            remaining = interval - (time.monotonic() - tick)
            if remaining > 0:
                await asyncio.sleep(remaining)
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_seconds)
        except TimeoutError as exc:
            snapshot = scheduler.metrics.snapshot().as_dict()
            raise RuntimeError(
                f"case {name} timed out after {timeout_seconds}s; metrics={snapshot}"
            ) from exc
        snapshot = scheduler.metrics.snapshot()
    print(f"[actserve] completed case={name}", file=sys.stderr, flush=True)
    result = snapshot.as_dict()
    result.update(
        {
            "case": name,
            "wall_seconds": time.monotonic() - started,
            "backend_calls": backend.calls,
            "backend_items": backend.items,
        }
    )
    return result


async def compare(args: argparse.Namespace) -> dict[str, Any]:
    batch_sizes = _parse_batch_sizes(args.batch_sizes, args.max_batch_size)
    backend = TorchVisionBackend(
        device=args.device,
        max_batch_size=max(batch_sizes),
        image_size=args.image_size,
        width=args.width,
        layers=args.layers,
    )
    calibration = backend.calibrate(repetitions=args.calibration_repetitions)
    torch = backend.torch
    frames = [
        torch.randn(3, args.image_size, args.image_size, dtype=torch.float32).pin_memory()
        for _ in range(args.sessions)
    ]
    workload = CudaWorkload(
        sessions=args.sessions,
        observations_per_session=args.observations,
        observation_hz=args.hz,
        deadline_ms=args.deadline_ms,
    )
    baseline = await _run_case(
        "fifo_batch1",
        backend,
        workload,
        frames,
        SchedulerConfig(
            policy="fifo",
            max_batch_wait_ms=0,
            max_batch_size=1,
            coalesce_sessions=False,
        ),
        args.case_timeout_seconds,
    )
    optimized = []
    for batch_size in batch_sizes:
        result = await _run_case(
            f"edf_coalescing_microbatch_{batch_size}",
            backend,
            workload,
            frames,
            SchedulerConfig(
                policy="edf",
                max_batch_wait_ms=args.batch_wait_ms,
                dispatch_guard_ms=args.dispatch_guard_ms,
                max_batch_size=batch_size,
                coalesce_sessions=True,
            ),
            args.case_timeout_seconds,
        )
        result["configured_max_batch_size"] = batch_size
        optimized.append(result)
    recommendation = max(
        optimized,
        key=lambda result: (
            result["completed_on_time"],
            -result["e2e_p95_ms"],
            -result["backend_calls"],
        ),
    )
    return {
        "schema": "actserve.cuda_benchmark.v1",
        "device": torch.cuda.get_device_name(backend.device),
        "torch_version": torch.__version__,
        "model": {
            "image_size": args.image_size,
            "width": args.width,
            "layers": args.layers,
        },
        "workload": {
            "sessions": workload.sessions,
            "observations_per_session": workload.observations_per_session,
            "observation_hz": workload.observation_hz,
            "deadline_ms": workload.deadline_ms,
        },
        "calibrated_p90_guarded_ms": calibration,
        "recommendation": {
            "max_batch_size": recommendation["configured_max_batch_size"],
            "selection_order": [
                "completed_on_time desc",
                "e2e_p95_ms asc",
                "backend_calls asc",
            ],
        },
        "results": [baseline, *optimized],
    }


def _parse_batch_sizes(value: str | None, default: int) -> list[int]:
    if value is None:
        return [default]
    try:
        sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not sizes or sizes[0] < 1:
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the public ActServe CUDA benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--observations", type=int, default=20)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument(
        "--batch-sizes",
        help="comma-separated batch ceilings to benchmark and rank after one calibration",
    )
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--dispatch-guard-ms", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--calibration-repetitions", type=int, default=7)
    parser.add_argument("--case-timeout-seconds", type=float, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(asyncio.run(compare(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
