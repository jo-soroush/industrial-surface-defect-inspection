"""Tests for the explicit local Gemini smoke harness."""

from __future__ import annotations

import importlib.util
import ast
import os
import pathlib
import subprocess
import sys
from pathlib import Path

import pytest

from src.inspection_ai.agent.gemini_provider import GeminiRealGenerationResult
from src.inspection_ai.agent.gemini_provider import GeminiSdkLoadResult
from src.inspection_ai.agent.context_builder import build_grounding_context as build_real_grounding_context
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


def _build_failure_result(
    status: str = "provider_error",
    *,
    provider_used: str = "mock",
    fallback_used: bool = True,
    grounding_status: str = "grounded",
    safety_status: str = "blocked",
    safety_block_reason: str | None = None,
    fallback_reason: str = "Gemini real provider raised a provider error; mock fallback remains the safe path.",
    provider_error: str = "Gemini real provider raised a provider error.",
) -> GeminiRealGenerationResult:
    response = build_provider_response(
        answer="This should never be printed verbatim.",
        provider_used=provider_used,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        grounding_status=grounding_status,
        safety_status=safety_status,
        limitations=["manual review still applies"],
        evidence_used=[{"source": "test.fake", "value": "safe"}],
        provider_error=provider_error,
    )
    return GeminiRealGenerationResult(
        provider_response=response,
        status=status,
        safe_to_send=False,
        safe_to_display=False,
        provider_error=provider_error,
        fallback_reason=response.fallback_reason,
        safety_block_reason=safety_block_reason,
        client_name="fake-sdk",
    )


def _build_real_result_with_status(
    status: str,
    *,
    provider_used: str = "gemini",
    fallback_used: bool = False,
    safe_to_send: bool = True,
    safe_to_display: bool = True,
    safety_status: str | None = None,
) -> GeminiRealGenerationResult:
    response = build_provider_response(
        answer="This is a safe mocked Gemini answer. Manual review still applies.",
        provider_used=provider_used,
        fallback_used=fallback_used,
        fallback_reason=None,
        grounding_status="grounded",
        safety_status=safety_status if safety_status is not None else ("pass" if status == "pass" else "blocked"),
        limitations=["manual review still applies"],
        evidence_used=[{"source": "test.fake", "value": "safe"}],
        provider_error=None,
    )
    return GeminiRealGenerationResult(
        provider_response=response,
        status=status,
        safe_to_send=safe_to_send,
        safe_to_display=safe_to_display,
        provider_error=None,
        fallback_reason=None,
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
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint({"google", "openai", "requests", "httpx", "urllib"})


def test_default_dry_run_exits_successfully_without_key_access(monkeypatch, capsys) -> None:
    def _fail_getenv(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("GEMINI_API_KEY should not be read in dry-run mode")

    monkeypatch.setattr(os, "getenv", _fail_getenv)
    module = load_harness_module()

    exit_code = module.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "gemini_local_smoke_status=DRY_RUN" in captured.out
    assert "smoke_model=gemini-2.5-flash" in captured.out
    assert "no_real_gemini_api_call_was_made=true" in captured.out
    assert "gemini_api_key_read=false" in captured.out
    assert "normal_agent_route=mock_first" in captured.out


def test_dry_run_output_says_no_real_api_call_was_made(capsys) -> None:
    module = load_harness_module()

    exit_code = module.main(["--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "gemini_local_smoke_status=DRY_RUN" in captured.out
    assert "smoke_model=gemini-2.5-flash" in captured.out
    assert "no_real_gemini_api_call_was_made=true" in captured.out
    assert "provider_routing_activation=disabled" in captured.out
    assert "future_real_smoke_requires_explicit_user_approval=true" in captured.out
    assert "error_category=" not in captured.out


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
    assert "smoke_model=gemini-2.5-flash" in captured.out
    assert "no_real_gemini_api_call_was_made=true" in captured.out
    assert "error_category=" not in captured.out


def test_smoke_question_helper_adds_evidence_only_instructions() -> None:
    module = load_harness_module()

    wrapped = module._smoke_question_with_evidence_only_instructions(
        "Explain the current image inspection result in a safe way."
    )

    assert "Use only the provided evidence." in wrapped
    assert "Use no numbers." in wrapped
    assert "Do not mention any numeric value, metric, score, threshold, probability, percentage, count, model name, model version, run ID, or confidence value unless it is explicitly requested and present exactly in the allowed evidence." in wrapped
    assert "Do not invent metrics or convert numeric values to other formats." in wrapped
    assert "Do not claim production readiness, deployment readiness, HTTPS/domain readiness, or real LLM readiness." in wrapped
    assert "Explain only the qualitative decision and manual-review boundary." in wrapped
    assert "If evidence is insufficient, say manual review is required." in wrapped
    assert "User question: Explain the current image inspection result in a safe way." in wrapped


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
    assert "smoke_success_level=full" in captured.out
    assert "smoke_model=gemini-2.5-flash" in captured.out
    assert "request_summary=page_id=image_inspection;section_id=final_decision;component_id=image_inspection_ai_explanation_panel;question_sanitized=true" in captured.out
    assert "response_summary=manual_review_visible=true;sanitized=true;raw_response_hidden=true" in captured.out
    assert "cleanup_reminder=unset_temporary_key_and_restore_mock_defaults;export AGENT_ENABLE_LLM=false;export AGENT_DEFAULT_PROVIDER=mock" in captured.out
    assert "present-but-disabled" not in captured.out
    assert "GEMINI_API_KEY" not in captured.out
    assert "Explain the current image inspection result in a safe way." not in captured.out
    assert "raw_provider_response" not in captured.out
    assert "sdk_missing" not in captured.out


def test_confirmation_flag_limited_safe_result_is_reported_as_success(monkeypatch, capsys) -> None:
    module = load_harness_module()

    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(
        module,
        "generate_with_real_gemini_provider",
        lambda *a, **k: _build_real_result_with_status(
            "limited",
            safety_status="limited",
            safe_to_send=True,
            safe_to_display=True,
        ),
    )

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
    assert "gemini_local_smoke_status=SUCCESS_LIMITED" in captured.out
    assert "smoke_success_level=limited" in captured.out
    assert "result_status=limited" in captured.out
    assert "provider_used=gemini" in captured.out
    assert "fallback_used=false" in captured.out
    assert "grounding_status=grounded" in captured.out
    assert "safety_status=limited" in captured.out
    assert "This is a safe mocked Gemini answer." not in captured.out


@pytest.mark.parametrize(
    "result_factory",
    [
        lambda: _build_real_result_with_status(
            "limited",
            provider_used="gemini",
            fallback_used=True,
            safety_status="limited",
        ),
        lambda: _build_real_result_with_status(
            "limited",
            provider_used="mock",
            fallback_used=False,
            safety_status="limited",
        ),
        lambda: _build_real_result_with_status(
            "limited",
            provider_used="gemini",
            fallback_used=False,
            safe_to_send=False,
            safety_status="limited",
        ),
        lambda: _build_real_result_with_status(
            "limited",
            provider_used="gemini",
            fallback_used=False,
            safe_to_display=False,
            safety_status="limited",
        ),
    ],
)
def test_limited_result_must_meet_all_success_conditions(result_factory) -> None:
    module = load_harness_module()
    result = result_factory()

    assert module._is_successful_smoke_result(result) is False
    assert module._is_limited_success_smoke_result(result) is False


def test_explicit_execute_path_uses_minimal_grounded_smoke_context(monkeypatch) -> None:
    module = load_harness_module()
    captured_kwargs: dict[str, object] = {}
    real_build_grounding_context = build_real_grounding_context

    def _capture_grounding_context(*args, **kwargs):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return real_build_grounding_context(*args, **kwargs)

    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(module, "generate_with_real_gemini_provider", lambda *a, **k: _build_success_result())
    monkeypatch.setattr(module, "build_grounding_context", _capture_grounding_context)

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
    assert captured_kwargs["visible_context"] == module._minimal_smoke_visible_context()
    assert captured_kwargs["inspection_response"] == module._minimal_smoke_inspection_response()
    assert captured_kwargs["include_raw_evidence"] is False
    assert captured_kwargs["visible_context"]
    assert captured_kwargs["inspection_response"]
    wrapped_question = captured_kwargs["question"]
    assert isinstance(wrapped_question, str)
    assert "Use only the provided evidence." in wrapped_question
    assert "Use no numbers." in wrapped_question
    assert "Do not mention any numeric value, metric, score, threshold, probability, percentage, count, model name, model version, run ID, or confidence value unless it is explicitly requested and present exactly in the allowed evidence." in wrapped_question
    assert "Do not invent metrics or convert numeric values to other formats." in wrapped_question
    assert "Do not claim production readiness, deployment readiness, HTTPS/domain readiness, or real LLM readiness." in wrapped_question
    assert "Explain only the qualitative decision and manual-review boundary." in wrapped_question
    assert "If evidence is insufficient, say manual review is required." in wrapped_question
    assert "Explain the current image inspection result in a safe way." in wrapped_question


def test_minimal_smoke_context_builds_grounded_context_without_secrets() -> None:
    module = load_harness_module()
    visible_context = module._minimal_smoke_visible_context()
    inspection_response = module._minimal_smoke_inspection_response()

    context = build_real_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        component_id="image_inspection_ai_explanation_panel",
        question="Explain the current image inspection result in a safe way.",
        visible_context=visible_context,
        inspection_response=inspection_response,
        include_raw_evidence=False,
    )

    joined = "\n".join(
        [
            repr(visible_context),
            repr(inspection_response),
            repr(context.evidence_used),
            repr(context.limitations),
        ]
    )

    assert context.grounding_status == "grounded"
    assert visible_context["page_title"] == "Image Inspection"
    assert inspection_response["decision"]["final_decision"] == "manual_review_required"
    assert inspection_response["traceability"]["source_endpoint"] == "local_smoke_synthetic_context"
    assert "classification" not in inspection_response
    assert "detection" not in inspection_response
    assert "anomaly" not in inspection_response
    assert "GEMINI_API_KEY" not in joined
    assert "/Users/" not in joined
    assert "raw_image" not in joined
    assert "0.72" not in joined
    assert "0.50" not in joined
    assert "0.21" not in joined


def test_confirmation_flag_passes_sdk_readiness_loader_and_module_loader(monkeypatch) -> None:
    module = load_harness_module()
    captured_kwargs: dict[str, object] = {}

    def _capture_generate(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _build_real_result_with_status("pass")

    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(module, "generate_with_real_gemini_provider", _capture_generate)

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
    assert callable(captured_kwargs["sdk_loader"])
    assert callable(captured_kwargs["sdk_module_loader"])
    assert captured_kwargs["sdk_module_loader"] is module._load_google_genai_module
    assert captured_kwargs["config"].model_name == "gemini-2.5-flash"
    readiness = captured_kwargs["sdk_loader"]()
    assert isinstance(readiness, GeminiSdkLoadResult)
    assert readiness.checked is True
    assert readiness.sdk_available is True
    assert readiness.status == "available"
    assert readiness.sdk_name == "google-genai"
    assert readiness.import_style == "from google import genai"


def test_sdk_readiness_helper_reports_available_when_google_genai_import_succeeds(monkeypatch) -> None:
    module = load_harness_module()

    class _FakeSdkModule:
        Client = object

    monkeypatch.setattr(module, "_load_google_genai_module", lambda: _FakeSdkModule())

    result = module._load_gemini_sdk_readiness_result()

    assert result.checked is True
    assert result.sdk_available is True
    assert result.status == "available"
    assert result.sdk_name == "google-genai"
    assert result.import_style == "from google import genai"


def test_sdk_readiness_helper_reports_load_error_when_google_genai_import_fails(monkeypatch) -> None:
    module = load_harness_module()

    def _raise_import_error():
        raise ModuleNotFoundError("google.genai missing in test seam")

    monkeypatch.setattr(module, "_load_google_genai_module", _raise_import_error)

    result = module._load_gemini_sdk_readiness_result()

    assert result.checked is True
    assert result.sdk_available is False
    assert result.status == "load_error"
    assert result.sdk_name == "google-genai"
    assert result.import_style == "from google import genai"
    assert "google.genai import failed" in result.reason


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
    assert "error_category=provider_error" in captured.out
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
    assert "sdk_missing" not in captured.out
    assert "safety_stage=" not in captured.out
    assert "safety_block_reason=" not in captured.out
    assert "provider_error_stage=" in captured.out
    assert "provider_error_reason=" in captured.out


def test_confirmation_flag_fake_provider_error_output_includes_stage_and_reason(monkeypatch, capsys) -> None:
    module = load_harness_module()
    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(
        module,
        "generate_with_real_gemini_provider",
        lambda *a, **k: _build_failure_result(
            status="provider_error",
            safety_status="pass",
            fallback_reason="Gemini real provider raised a provider error; mock fallback remains the safe path.",
            provider_error="Gemini real provider raised a provider error.",
        ),
    )

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

    assert exit_code == 1
    assert "gemini_local_smoke_status=FAILED" in captured.out
    assert "result_status=provider_error" in captured.out
    assert "error_category=provider_error" in captured.out
    assert "provider_error_stage=client_invocation" in captured.out
    assert "provider_error_reason=invoke_raised_provider_error" in captured.out
    assert "Gemini real provider raised a provider error." not in captured.out
    assert "GEMINI_API_KEY" not in captured.out
    assert "provider_used=mock" in captured.out
    assert "fallback_used=true" in captured.out
    assert "grounding_status=grounded" in captured.out
    assert "safety_status=pass" in captured.out


def test_confirmation_flag_fake_safety_block_output_includes_stage_and_reason(monkeypatch, capsys) -> None:
    module = load_harness_module()
    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(
        module,
        "generate_with_real_gemini_provider",
        lambda *a, **k: _build_failure_result(
            status="blocked",
            safety_status="blocked",
            fallback_reason="Gemini real provider output was blocked by the safety guard; mock fallback remains the safe path.",
            provider_error="Gemini real provider output was blocked by the safety guard.",
        ),
    )

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

    assert exit_code == 1
    assert "gemini_local_smoke_status=FAILED" in captured.out
    assert "result_status=blocked" in captured.out
    assert "error_category=safety_blocked" in captured.out
    assert "safety_status=blocked" in captured.out
    assert "safety_stage=post_generation" in captured.out
    assert "safety_block_reason=safety_guard_blocked" in captured.out
    assert "provider_error_stage=" not in captured.out
    assert "provider_error_reason=" not in captured.out
    assert "Gemini real provider output was blocked by the safety guard." not in captured.out
    assert "This should never be printed verbatim." not in captured.out
    assert "GEMINI_API_KEY" not in captured.out


@pytest.mark.parametrize(
    ("safety_block_reason", "expected_reason"),
    [
        ("invented_metric_like_output", "invented_metric_like_output"),
        ("unsupported_readiness_claim", "unsupported_readiness_claim"),
        ("provider_connected_claim", "provider_connected_claim"),
        ("secret_or_path_leak", "secret_or_path_leak"),
        ("unknown", "unknown"),
    ],
)
def test_confirmation_flag_fake_safety_block_output_exposes_specific_sanitized_reason(
    monkeypatch,
    capsys,
    safety_block_reason: str,
    expected_reason: str,
) -> None:
    module = load_harness_module()
    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(
        module,
        "generate_with_real_gemini_provider",
        lambda *a, **k: _build_failure_result(
            status="blocked",
            safety_status="blocked",
            safety_block_reason=safety_block_reason,
            fallback_reason="Gemini real provider output was blocked by the safety guard; mock fallback remains the safe path.",
            provider_error="Gemini real provider output was blocked by the safety guard.",
        ),
    )

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

    assert exit_code == 1
    assert "gemini_local_smoke_status=FAILED" in captured.out
    assert "result_status=blocked" in captured.out
    assert "error_category=safety_blocked" in captured.out
    assert "safety_status=blocked" in captured.out
    assert "safety_stage=post_generation" in captured.out
    assert f"safety_block_reason={expected_reason}" in captured.out
    assert "Gemini real provider output was blocked by the safety guard." not in captured.out
    assert "This should never be printed verbatim." not in captured.out
    assert "GEMINI_API_KEY" not in captured.out


def test_failure_lines_map_known_error_statuses_to_sanitized_categories() -> None:
    module = load_harness_module()
    request = _build_real_provider_request(module)

    expected_categories = {
        "provider_error": "provider_error",
        "timeout": "timeout",
        "rate_limit": "rate_limited",
        "empty": "empty_response",
        "malformed": "malformed_response",
        "load_error": "sdk_load_error",
        "sdk_missing": "sdk_missing",
    }

    for status, expected_category in expected_categories.items():
        result = _build_failure_result(status=status)
        lines = module.build_failure_lines(
            request=request,
            result=result,
            smoke_model_name="gemini-2.5-flash",
        )
        joined = "\n".join(lines)
        assert f"error_category={expected_category}" in joined
        assert "Gemini real provider raised a provider error." not in joined
        assert "present-but-disabled" not in joined
        if status == "provider_error":
            assert "provider_error_stage=client_invocation" in joined
            assert "provider_error_reason=invoke_raised_provider_error" in joined
        else:
            assert "provider_error_stage=" not in joined
            assert "provider_error_reason=" not in joined


def test_failure_lines_map_blocked_safety_status_to_sanitized_category() -> None:
    module = load_harness_module()
    request = _build_real_provider_request(module)
    result = _build_failure_result(status="blocked")
    lines = module.build_failure_lines(
        request=request,
        result=result,
        smoke_model_name="gemini-2.5-flash",
    )

    assert "error_category=safety_blocked" in "\n".join(lines)


def test_failure_lines_map_unavailable_fallback_to_sanitized_category() -> None:
    module = load_harness_module()
    request = _build_real_provider_request(module)
    result = _build_failure_result(
        status="failed",
        provider_used="mock",
        fallback_used=True,
        grounding_status="insufficient_evidence",
        safety_status="pass",
        fallback_reason="missing key",
    )
    lines = module.build_failure_lines(
        request=request,
        result=result,
        smoke_model_name="gemini-2.5-flash",
    )

    assert "error_category=unavailable" in "\n".join(lines)


def test_failure_lines_map_client_creation_provider_error_to_sanitized_category() -> None:
    module = load_harness_module()
    request = _build_real_provider_request(module)
    result = _build_failure_result(
        status="provider_error",
        provider_error="Gemini client creation failed.",
        fallback_reason="Gemini client creation failed; mock fallback remains the safe path.",
    )

    lines = module.build_failure_lines(
        request=request,
        result=result,
        smoke_model_name="gemini-2.5-flash",
    )

    joined = "\n".join(lines)
    assert "error_category=provider_error" in joined
    assert "provider_error_stage=client_creation" in joined
    assert "provider_error_reason=client_creation_failed" in joined


def test_failure_lines_map_service_unavailable_provider_error_to_sanitized_category() -> None:
    module = load_harness_module()
    request = _build_real_provider_request(module)
    result = _build_failure_result(
        status="provider_error",
        provider_error="Gemini real provider service unavailable.",
        fallback_reason="Gemini real provider service unavailable; mock fallback remains the safe path.",
    )

    lines = module.build_failure_lines(
        request=request,
        result=result,
        smoke_model_name="gemini-2.5-flash",
    )

    joined = "\n".join(lines)
    assert "error_category=provider_error" in joined
    assert "provider_error_stage=client_invocation" in joined
    assert "provider_error_reason=service_unavailable" in joined
    assert "503" not in joined


def test_default_smoke_model_name_is_smoke_only_and_can_be_overridden(monkeypatch) -> None:
    module = load_harness_module()
    captured_model_names: list[str] = []

    def _capture_generate(*args, **kwargs):
        captured_model_names.append(kwargs["config"].model_name)
        return _build_success_result()

    monkeypatch.setattr(module, "_resolve_gemini_api_key", lambda: "present-but-disabled")
    monkeypatch.setattr(module, "generate_with_real_gemini_provider", _capture_generate)

    default_exit_code = module.main(
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
    override_exit_code = module.main(
        [
            "--execute",
            "--i-understand-this-calls-gemini",
            "--model-name",
            "gemini-3-flash-preview",
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

    assert default_exit_code == 0
    assert override_exit_code == 0
    assert captured_model_names == ["gemini-2.5-flash", "gemini-3-flash-preview"]


def test_invalid_smoke_model_name_is_blocked_safely(capsys) -> None:
    module = load_harness_module()

    exit_code = module.main(
        [
            "--execute",
            "--model-name",
            "bad model name",
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

    assert exit_code == 2
    assert "gemini_local_smoke_status=BLOCKED" in captured.out
    assert "reason=invalid_smoke_model_name" in captured.out
    assert "smoke_model_valid=false" in captured.out
    assert "bad model name" not in captured.out
    assert "GEMINI_API_KEY" not in captured.out


def _build_real_provider_request(module):
    grounding_context = module.build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        component_id="image_inspection_ai_explanation_panel",
        question="Explain the current image inspection result in a safe way.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {"final_decision": "good", "rule_id": "manual_check_rule"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )
    pre_guard = module.guard_pre_generation_context(grounding_context)
    return module.build_provider_request(
        provider_name="gemini",
        grounding_context=grounding_context,
        sanitized_context=pre_guard.sanitized_context,
        safety_status=pre_guard.status,
        llm_enabled=True,
    )


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
