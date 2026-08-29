from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import Any

from .integrations.http_json import HttpJsonBackend
from .scheduler import Scheduler, SchedulerConfig
from .server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actserve serve",
        description="Serve ActServe in front of an existing JSON/HTTP policy backend.",
    )
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--dispatch-guard-ms", type=float, default=1.0)
    parser.add_argument("--backend-timeout-ms", type=float, default=30_000)
    parser.add_argument(
        "--initial-backend-latency-ms",
        type=float,
        help="conservative latency floor used before enough HTTP observations exist",
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
        "--backend-token-env",
        help="environment variable containing the bearer token for the policy backend",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def create_server_app(
    args: argparse.Namespace, *, environ: Mapping[str, str] | None = None
) -> Any:
    environ = os.environ if environ is None else environ
    api_key = _secret_from_env(args.api_key_env, environ)
    backend_token = _secret_from_env(args.backend_token_env, environ)
    headers = None if backend_token is None else {"Authorization": f"Bearer {backend_token}"}
    backend = HttpJsonBackend(
        args.backend_url,
        max_batch_size=args.max_batch_size,
        timeout_ms=args.backend_timeout_ms,
        initial_latency_ms=args.initial_backend_latency_ms,
        latency_window=args.latency_window,
        latency_safety_factor=args.latency_safety_factor,
        headers=headers,
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if args.max_pending_requests < 1:
        parser.error("max-pending-requests must be at least 1")
    if args.latency_window < 1:
        parser.error("latency-window must be at least 1")
    if args.latency_safety_factor < 1:
        parser.error("latency-safety-factor must be at least 1")
    if args.initial_backend_latency_ms is not None and args.initial_backend_latency_ms < 0:
        parser.error("initial-backend-latency-ms must be non-negative")
    try:
        app = create_server_app(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install ActServe with the 'server' extra") from exc
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
