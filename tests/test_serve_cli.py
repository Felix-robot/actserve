from __future__ import annotations

import pytest

from actserve.serve_cli import build_parser, create_server_app


def test_serve_app_reads_tokens_from_environment_without_exposing_values() -> None:
    args = build_parser().parse_args(
        [
            "--backend-url",
            "http://127.0.0.1:9000/infer",
            "--api-key-env",
            "ACTSERVE_API_KEY",
            "--backend-token-env",
            "POLICY_API_KEY",
        ]
    )
    app = create_server_app(
        args,
        environ={"ACTSERVE_API_KEY": "front-secret", "POLICY_API_KEY": "back-secret"},
    )

    assert app.title == "ActServe"
    assert "front-secret" not in repr(app)
    assert "back-secret" not in repr(app)


def test_serve_app_rejects_missing_secret_environment_variable() -> None:
    args = build_parser().parse_args(
        [
            "--backend-url",
            "http://127.0.0.1:9000/infer",
            "--api-key-env",
            "MISSING_TOKEN",
        ]
    )
    with pytest.raises(ValueError, match="MISSING_TOKEN"):
        create_server_app(args, environ={})
