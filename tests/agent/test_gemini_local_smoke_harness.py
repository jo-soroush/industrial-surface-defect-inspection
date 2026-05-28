"""Tests for the explicit local Gemini smoke harness."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
from pathlib import Path

from src.inspection_ai.agent.gemini_provider import GeminiRealGenerationResult
from src.inspection_ai.agent.provider_contracts import build_provider_response
from src.inspection_ai.agent.provider_router import AgentProviderRouter


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "agent" / "run_gemini_local_smoke.py"


def load_harness_module():
    module_name = "run_gemini_local_smoke_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, HARNESS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subprocess_env_without_pythonpath() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def _build_success_result() -> GeminiRealGenerationResult:
    response = build_provider_response(
        answer="This is a safe mocked Gemini answer. Manual review still applies.",
        provider_used="gemini",
        fallback_used=False,
        fallback_reason=None,
        grounding_status="grounded",
        safety_status="pass",
        limitations=["manual review still applies"],
        evidence_used=[{"source": "test.fake", "value": "safe"}],
        provider_error=None,
    )
    return GeminiRealGenerationResult(
        provider_response=response,
        status="pass",
        safe_to_send=True,
        safe_to_display=True,
        provider_error=None,
        fallback_reason=None,
        client_name="fake-sdk",
    )


def _build_failure_result() -> GeminiRealGenerationResult:
    response = build_provider_response(
        answer="This should never be printed verbatim.",
        provider_used="mock",
        fallback_used=True,
        fallback_reason="Gemini real provider raised a provider error; mock fallback remains the safe path.",
        grounding_status="grounded",
        safety_status="blocked",
        limitations=["manual review still applies"],
        evidence_used=[{"source": "test.fake", "value": "safe"}],
        provider_error="Gemini real provider raised a provider error.",
    )
    return GeminiRealGenerationResult(
        provider_response=response,
        status="provider_error",
        safe_to_send=False,
        safe_to_display=False,
        provider_error="Gemini real provider raised a provider error.",
        fallback_reason=response.fallback_reason,
        client_name="fake-sdk",
    )


def test_harness_import_does_not_read_gemini_api_key(monkeypatch) -> None:
    def _fail_getenv(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("GEMINI_API_KEY should not be read during import")

    monkeypatch.setattr(os, "getenv", _fail_getenv)
    module = load_harness_module()

    assert module.build_parser is not None
    assert module.main is not None


def test_harness_import_does_not_add_banned_sdk_or_network_modules() -> None:
    banned_roots = {"google", "openai", "requests", "httpx", "urllib"}
    before = {
        name for name in sys.modules if name.split(".", 1)[0] in banned_roots
    }

    module = load_harness_module()

    after = {
        name for name in sys.modules if name.split(".", 1)[0] in banned_roots
    }

    assert after == before
    assert module.main is not None

    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert "provider_router" not in source
    assert "from google import genai" not in source
    assert "google.genai" not in source
    assert "google.generativeai" not in source
    assert "import google" not in source
    assert "requests" not in source
    assert "httpx" not in source
    assert "urllib" not in source


def test_default_dry_run_exits_successfully_without_key_access(monkeypatch, capsys) -> None:
    def _fail_getenv(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("GEMINI_API_KEY should not be read in dry-run mode")

    monkeypatch.setattr(os, "getenv", _fail_getenv)
    module = load_harness_module()

    exit_code = module.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "gemini_local_smoke_status=DRY_RUN" in captured.out
    assert "no_real_gemini_api_call_was_made=true" in captured.out
    assert "gemini_api_key_read=false" in captured.out
    assert "normal_agent_route=mock_first" in captured.out


def test_dry_run_output_says_no_real_api_call_was_made(capsys) -> None:
    module = load_harness_module()

    exit_code = module.main(["--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "gemini_local_smoke_status=DRY_RUN" in captured.out
    assert "no_real_gemini_api_call_was_made=true" in captured.out
    assert "provider_routing_activation=disabled" in captured.out
    assert "future_real_smoke_requires_explicit_user_approval=true" in captured.out


def test_missing_confirmation_flag_blocks_non_dry_run_before_key_access(monkeypatch, capsys) -> None:
    def _fail_getenv(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("GEMINI_API_KEY should not be read before confirmation")

    monkeypatch.setattr(os, "getenv", _fail_getenv)
    module = load_harness_module()

    exit_code = module.main(["--execute", "--question", "sanitized"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "gemini_local_smoke_status=BLOCKED" in captured.out
    assert "reason=missing_confirmation_flag" in captured.out
    assert "no_real_gemini_api_call_was_made=true" in captured.out


def test_confirmation_flag_reaches_explicit_execution_helper_with_fake_seam(monkeypatch, capsys) -> None:
    module = load_harness_module()

    def _fail_router(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("AgentProviderRouter should not be involved in the harness smoke path")

    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(module, "generate_with_real_gemini_provider", lambda *a, **k: _build_success_result())
    monkeypatch.setattr(AgentProviderRouter, "health", _fail_router)
    monkeypatch.setattr(AgentProviderRouter, "explain", _fail_router)

    exit_code = module.main(
        [
            "--execute",
            "--i-understand-this-calls-gemini",
            "--question",
            "Explain the current image inspection result in a safe way.",
            "--page-id",
            "image_inspection",
            "--section-id",
            "final_decision",
            "--component-id",
            "image_inspection_ai_explanation_panel",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "gemini_local_smoke_status=SUCCESS" in captured.out
    assert "result_status=pass" in captured.out
    assert "provider_used=gemini" in captured.out
    assert "fallback_used=false" in captured.out
    assert "grounding_status=grounded" in captured.out
    assert "safety_status=pass" in captured.out
    assert "request_summary=page_id=image_inspection;section_id=final_decision;component_id=image_inspection_ai_explanation_panel;question_sanitized=true" in captured.out
    assert "response_summary=manual_review_visible=true;sanitized=true;raw_response_hidden=true" in captured.out
    assert "cleanup_reminder=unset_temporary_key_and_restore_mock_defaults;export AGENT_ENABLE_LLM=false;export AGENT_DEFAULT_PROVIDER=mock" in captured.out
    assert "present-but-disabled" not in captured.out
    assert "GEMINI_API_KEY" not in captured.out
    assert "Explain the current image inspection result in a safe way." not in captured.out
    assert "raw_provider_response" not in captured.out


def test_confirmation_flag_fake_failure_output_is_sanitized(monkeypatch, capsys) -> None:
    module = load_harness_module()
    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(module, "generate_with_real_gemini_provider", lambda *a, **k: _build_failure_result())

    exit_code = module.main(
        [
            "--execute",
            "--i-understand-this-calls-gemini",
            "--question",
            "super-secret-test-question",
            "--page-id",
            "image_inspection",
            "--section-id",
            "final_decision",
            "--component-id",
            "image_inspection_ai_explanation_panel",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "gemini_local_smoke_status=FAILED" in captured.out
    assert "result_status=provider_error" in captured.out
    assert "provider_used=mock" in captured.out
    assert "fallback_used=true" in captured.out
    assert "grounding_status=grounded" in captured.out
    assert "safety_status=blocked" in captured.out
    assert "super-secret-test-question" not in captured.out
    assert "present-but-disabled" not in captured.out
    assert "GEMINI_API_KEY" not in captured.out
    assert "This should never be printed verbatim." not in captured.out
    assert "response_summary=manual_review_visible=true;sanitized=true;raw_response_hidden=true" in captured.out
    assert "cleanup_reminder=unset_temporary_key_and_restore_mock_defaults;export AGENT_ENABLE_LLM=false;export AGENT_DEFAULT_PROVIDER=mock" in captured.out


def test_confirmation_flag_path_does_not_write_files(monkeypatch) -> None:
    module = load_harness_module()

    def _fail_write(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("The smoke harness must not write files")

    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(module, "generate_with_real_gemini_provider", lambda *a, **k: _build_success_result())
    monkeypatch.setattr(pathlib.Path, "write_text", _fail_write)
    monkeypatch.setattr(pathlib.Path, "write_bytes", _fail_write)

    exit_code = module.main(
        [
            "--execute",
            "--i-understand-this-calls-gemini",
            "--question",
            "Explain the current image inspection result in a safe way.",
            "--page-id",
            "image_inspection",
            "--section-id",
            "final_decision",
            "--component-id",
            "image_inspection_ai_explanation_panel",
        ]
    )

    assert exit_code == 0


def test_cli_runs_from_repo_root_without_pythonpath_in_dry_run_mode() -> None:
    completed = subprocess.run(
        [sys.executable, str(HARNESS_PATH)],
        cwd=REPO_ROOT,
        env=_subprocess_env_without_pythonpath(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "gemini_local_smoke_status=DRY_RUN" in completed.stdout
    assert "no_real_gemini_api_call_was_made=true" in completed.stdout
    assert "GEMINI_API_KEY" not in completed.stdout


def test_cli_runs_from_repo_root_without_pythonpath_in_blocked_mode() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(HARNESS_PATH),
            "--execute",
            "--question",
            "sanitized",
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env_without_pythonpath(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "gemini_local_smoke_status=BLOCKED" in completed.stdout
    assert "reason=missing_confirmation_flag" in completed.stdout
    assert "GEMINI_API_KEY" not in completed.stdout
