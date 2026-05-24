from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_compose_defines_api_service_only() -> None:
    compose = _load_compose()

    assert "services" in compose
    assert set(compose["services"]) == {"api"}


def test_api_service_builds_from_repo_root_and_exposes_port() -> None:
    compose = _load_compose()
    api = compose["services"]["api"]

    assert api["build"]["context"] == "."
    assert api["build"]["dockerfile"] == "Dockerfile"
    assert "8000:8000" in api["ports"]


def test_api_service_uses_mock_first_env_defaults() -> None:
    compose = _load_compose()
    env = compose["services"]["api"]["environment"]

    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONPATH"] == "/app"
    assert env["AGENT_ENABLE_LLM"] == "false"
    assert env["AGENT_DEFAULT_PROVIDER"] == "mock"
    assert env["LLM_PROVIDER_ORDER"] == "mock,gemini,grok"
    assert env["LLM_ENABLE_FALLBACK"] == "true"


def test_api_service_has_healthcheck_and_no_raw_mounts() -> None:
    compose = _load_compose()
    api = compose["services"]["api"]

    assert "healthcheck" in api
    assert "frontend" not in compose["services"]
    assert "volumes" not in api
