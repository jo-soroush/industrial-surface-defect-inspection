from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_compose_defines_api_and_frontend_services() -> None:
    compose = _load_compose()

    assert "services" in compose
    assert set(compose["services"]) == {"api", "frontend"}


def test_api_service_builds_from_repo_root_and_exposes_port() -> None:
    compose = _load_compose()
    api = compose["services"]["api"]

    assert api["build"]["context"] == "."
    assert api["build"]["dockerfile"] == "Dockerfile.api"
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
    assert "volumes" not in api


def test_frontend_service_uses_streamlit_and_api_service_dns() -> None:
    compose = _load_compose()
    frontend = compose["services"]["frontend"]

    assert frontend["build"]["context"] == "."
    assert frontend["build"]["dockerfile"] == "Dockerfile.frontend"
    assert "8501:8501" in frontend["ports"]
    assert frontend["depends_on"] == ["api"]
    assert frontend["environment"]["STREAMLIT_API_BASE_URL"] == "http://api:8000"
    assert "localhost" not in frontend["environment"]["STREAMLIT_API_BASE_URL"]
    assert frontend["command"] == [
        "streamlit",
        "run",
        "frontend/streamlit_app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
    ]
    assert "volumes" not in frontend


def test_compose_frontend_does_not_request_agent_env_defaults() -> None:
    compose = _load_compose()
    frontend = compose["services"]["frontend"]

    assert "AGENT_ENABLE_LLM" not in frontend["environment"]
    assert "AGENT_DEFAULT_PROVIDER" not in frontend["environment"]
    assert "LLM_PROVIDER_ORDER" not in frontend["environment"]
    assert "LLM_ENABLE_FALLBACK" not in frontend["environment"]
