from __future__ import annotations

import pytest

import actserve.smolvla_serve_cli as smolvla_serve_cli
from actserve.integrations.smolvla import (
    DEFAULT_SMOLVLA_MODEL_ID,
    DEFAULT_SMOLVLA_REVISION,
    SmolVLABackend,
)
from actserve.smolvla_serve_cli import build_parser, create_smolvla_server_app


class FakeBackend:
    max_batch_size = 4

    def batch_key(self, request):
        return request.model

    async def infer_batch(self, requests):
        return []

    async def aclose(self) -> None:
        return None


def test_smolvla_server_uses_tested_revision_and_environment_token(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(cls, model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return FakeBackend()

    monkeypatch.setattr(SmolVLABackend, "from_pretrained", classmethod(fake_from_pretrained))
    args = build_parser().parse_args(["--api-key-env", "ACTSERVE_API_KEY"])
    app = create_smolvla_server_app(args, environ={"ACTSERVE_API_KEY": "test-secret"})

    assert app.title == "ActServe"
    assert "test-secret" not in repr(app)
    assert captured["model_id"] == DEFAULT_SMOLVLA_MODEL_ID
    assert captured["revision"] == DEFAULT_SMOLVLA_REVISION
    assert captured["model_name"] == DEFAULT_SMOLVLA_MODEL_ID
    assert captured["device"] == "auto"


def test_custom_smolvla_model_does_not_inherit_public_revision(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(cls, model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return FakeBackend()

    monkeypatch.setattr(SmolVLABackend, "from_pretrained", classmethod(fake_from_pretrained))
    args = build_parser().parse_args(
        ["--model-id", "/models/custom-smolvla", "--served-model-name", "custom"]
    )
    create_smolvla_server_app(args, environ={})

    assert captured["revision"] is None
    assert captured["model_name"] == "custom"


def test_smolvla_server_rejects_missing_secret_environment_variable() -> None:
    args = build_parser().parse_args(["--api-key-env", "MISSING_TOKEN"])
    with pytest.raises(ValueError, match="MISSING_TOKEN"):
        create_smolvla_server_app(args, environ={})


def test_smolvla_server_checks_fastapi_before_loading_model(monkeypatch) -> None:
    loaded = False

    def fail_fastapi() -> None:
        raise RuntimeError("server extra is missing")

    def fake_from_pretrained(cls, model_id, **kwargs):
        nonlocal loaded
        loaded = True
        return FakeBackend()

    monkeypatch.setattr(smolvla_serve_cli, "_require_fastapi", fail_fastapi)
    monkeypatch.setattr(SmolVLABackend, "from_pretrained", classmethod(fake_from_pretrained))
    with pytest.raises(RuntimeError, match="server extra is missing"):
        create_smolvla_server_app(build_parser().parse_args([]), environ={})
    assert not loaded
