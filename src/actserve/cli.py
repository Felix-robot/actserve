from __future__ import annotations

import argparse
import sys

from . import __version__
from .async_benchmark import main as async_benchmark_main
from .benchmark import main as benchmark_main
from .cuda_benchmark import main as cuda_benchmark_main
from .profile_cli import main as profile_main
from .serial_benchmark import main as serial_benchmark_main


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "benchmark":
        return benchmark_main(argv[1:])
    if argv and argv[0] == "benchmark-async":
        return async_benchmark_main(argv[1:])
    if argv and argv[0] == "benchmark-cuda":
        return cuda_benchmark_main(argv[1:])
    if argv and argv[0] == "benchmark-serial":
        return serial_benchmark_main(argv[1:])
    if argv and argv[0] == "profile":
        return profile_main(argv[1:])
    parser = argparse.ArgumentParser(prog="actserve")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "command",
        nargs="?",
        help=(
            "available commands: benchmark, benchmark-async, benchmark-cuda, "
            "benchmark-serial, profile"
        ),
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover
