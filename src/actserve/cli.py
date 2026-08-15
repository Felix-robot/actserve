from __future__ import annotations

import argparse
import sys

from . import __version__
from .benchmark import main as benchmark_main


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "benchmark":
        return benchmark_main(argv[1:])
    parser = argparse.ArgumentParser(prog="actserve")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", help="available command: benchmark")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover
