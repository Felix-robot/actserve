from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .profiler import StageProfiler


@dataclass(frozen=True, slots=True)
class GpuSample:
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    power_watts: float


def parse_gpu_sample(line: str) -> GpuSample:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 4:
        raise ValueError(f"expected four nvidia-smi fields, got {len(fields)}")
    return GpuSample(*(float(field) for field in fields))


async def _probe_gpu(index: str) -> GpuSample:
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"nvidia-smi exited with {process.returncode}")
    first_line = stdout.decode("utf-8", errors="replace").splitlines()[0]
    return parse_gpu_sample(first_line)


async def _sample_gpu(
    profiler: StageProfiler,
    *,
    index: str,
    interval_ms: float,
    stopped: asyncio.Event,
) -> str | None:
    while not stopped.is_set():
        try:
            sample = await _probe_gpu(index)
        except (FileNotFoundError, IndexError, RuntimeError, ValueError) as exc:
            return f"{type(exc).__name__}: {exc}"
        profiler.observe("gpu.utilization", sample.utilization_percent, unit="percent")
        profiler.observe("gpu.memory_used", sample.memory_used_mib, unit="MiB")
        profiler.observe("gpu.memory_total", sample.memory_total_mib, unit="MiB")
        profiler.observe("gpu.power", sample.power_watts, unit="W")
        try:
            await asyncio.wait_for(stopped.wait(), timeout=interval_ms / 1000)
        except asyncio.TimeoutError:
            pass
    return None


async def profile_command(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after '--'")

    profiler = StageProfiler(sample_limit=args.sample_limit)
    stopped = asyncio.Event()
    sampler = None
    if args.gpu is not None:
        sampler = asyncio.create_task(
            _sample_gpu(
                profiler,
                index=args.gpu,
                interval_ms=args.interval_ms,
                stopped=stopped,
            )
        )

    started_ns = time.perf_counter_ns()
    process = await asyncio.create_subprocess_exec(*command)
    returncode = await process.wait()
    profiler.duration("process.wall", (time.perf_counter_ns() - started_ns) / 1_000_000)
    profiler.observe("process.exit_code", returncode, unit="code")
    stopped.set()
    gpu_error = await sampler if sampler is not None else None

    visible_command = (
        command
        if args.include_command
        else [Path(command[0]).name, "<arguments redacted>"]
    )
    report = {
        "schema": "actserve.command_profile.v1",
        "command": visible_command,
        "returncode": returncode,
        "gpu": args.gpu,
        "gpu_sample_error": gpu_error,
        "profile": profiler.snapshot().as_dict(),
    }
    return returncode, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actserve profile",
        description="Profile an arbitrary command without retaining model inputs or outputs.",
    )
    parser.add_argument("--gpu", help="nvidia-smi GPU index or UUID to sample")
    parser.add_argument("--interval-ms", type=float, default=500.0)
    parser.add_argument("--sample-limit", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-command",
        action="store_true",
        help="include command arguments in the report (redacted by default)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    if args.sample_limit < 1:
        parser.error("--sample-limit must be positive")
    try:
        returncode, report = asyncio.run(profile_command(args))
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return returncode
