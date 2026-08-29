from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import Any

from .integrations.smolvla import (
    DEFAULT_SMOLVLA_MODEL_ID,
    DEFAULT_SMOLVLA_REVISION,
    SmolVLABackend,
)
from .scheduler import Scheduler, SchedulerConfig
from .server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actserve serve-smolvla",
        description="Load LeRobot SmolVLA and serve it through the ActServe control plane.",
    )
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument(
        "--revision",
        help=(
            "model and processor revision; the tested public revision is used "
            "when model-id is lerobot/smolvla_base"
        ),
    )
    parser.add_argument("--served-model-name")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--dispatch-guard-ms", type=float, default=1.0)
    parser.add_argument(
        "--initial-backend-latency-ms",
        type=float,
        help="conservative latency floor used before enough local observations exist",
    )
    parser.add_argument("--latency-window", type=int, default=32)
    parser.add_argument("--latency-safety-factor", type=float, default=1.10)
    parser.add_argument(
        "--drop-unserviceable-requests",
        action="store_true",
        help="reject requests whose learned latency estimate predicts a deadline miss",
    )
    parser.add_argument(
        "--max-pending-requests",
        type=int,
        default=1024,
        help="maximum queued requests before new sessions receive HTTP 429",
    )
    parser.add_argument(
        "--api-key-env",
        help="environment variable containing the bearer token required by ActServe",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def create_smolvla_server_app(
    args: argparse.Namespace, *, environ: Mapping[str, str] | None = None
) -> Any:
    _require_fastapi()
    environ = os.environ if environ is None else environ
    api_key = _secret_from_env(args.api_key_env, environ)
    revision = args.revision
    if revision is None and args.model_id == DEFAULT_SMOLVLA_MODEL_ID:
        revision = DEFAULT_SMOLVLA_REVISION
    model_name = args.served_model_name or args.model_id
    backend = SmolVLABackend.from_pretrained(
        args.model_id,
        revision=revision,
        device=args.device,
        model_name=model_name,
        max_batch_size=args.max_batch_size,
        initial_latency_ms=args.initial_backend_latency_ms,
        latency_window=args.latency_window,
        latency_safety_factor=args.latency_safety_factor,
    )
    scheduler = Scheduler(
        backend,
        SchedulerConfig(
            max_batch_size=args.max_batch_size,
            max_batch_wait_ms=args.max_batch_wait_ms,
            dispatch_guard_ms=args.dispatch_guard_ms,
            max_pending_requests=args.max_pending_requests,
            drop_unserviceable_requests=args.drop_unserviceable_requests,
        ),
    )
    return create_app(scheduler, api_key=api_key)


def _secret_from_env(name: str | None, environ: Mapping[str, str]) -> str | None:
    if name is None:
        return None
    value = environ.get(name)
    if not value:
        raise ValueError(f"environment variable {name!r} is missing or empty")
    return value


def _require_fastapi() -> None:
    try:
        import fastapi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install ActServe with the 'server' extra") from exc
    del fastapi


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if args.max_batch_size < 1:
        parser.error("max-batch-size must be at least 1")
    if args.max_pending_requests < 1:
        parser.error("max-pending-requests must be at least 1")
    if args.latency_window < 1:
        parser.error("latency-window must be at least 1")
    if args.latency_safety_factor < 1:
        parser.error("latency-safety-factor must be at least 1")
    if args.initial_backend_latency_ms is not None and args.initial_backend_latency_ms < 0:
        parser.error("initial-backend-latency-ms must be non-negative")
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install ActServe with the 'server' extra") from exc
    try:
        app = create_smolvla_server_app(args)
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
