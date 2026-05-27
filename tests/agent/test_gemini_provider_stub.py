"""Tests for the Gemini Phase G1 provider stub."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.inspection_ai.agent.context_builder import build_grounding_context
from src.inspection_ai.agent.gemini_provider import (
    GeminiProviderConfig,
    GeminiProviderDisabledError,
    GeminiProviderStub,
    evaluate_gemini_provider_readiness,
)
from src.inspection_ai.agent.provider_contracts import (
    ProviderRuntimeSettings,
    build_provider_request,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STUB_PATH = REPO_ROOT / "src/inspection_ai/agent/gemini_provider.py"


def test_gemini_provider_stub_can_be_constructed_without_sdk_or_api_key() -> None:
    stub = GeminiProviderStub()

    assert stub.config.enabled is False
    assert stub.config.api_key_present is False
    assert stub.config.stub_only is True
    assert stub.config.model_name == "gemini-g1-stub"


def test_gemini_provider_stub_refuses_real_generation() -> None:
    stub = GeminiProviderStub()
    context = build_grounding_context(
        page_id="classification",
        section_id="detailed_metrics",
        component_id="classification_threshold_curve_chart",
        question="Explain /Users/jo.soroush/secret.key",
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )
    request = build_provider_request(
        provider_name="gemini",
        grounding_context=context,
        sanitized_context={
            "page_id": "classification",
            "section_id": "detailed_metrics",
            "component_id": "classification_threshold_curve_chart",
            "question": "[REDACTED_SECRET]",
        },
        safety_status="pass",
        llm_enabled=True,
    )

    assert request.provider_name == "gemini"
    assert request.safety_status == "pass"
    assert request.llm_enabled is True
    assert request.component_id == "classification_threshold_curve_chart"
    assert request.question == "[REDACTED_SECRET]"
    assert request.sanitized_context["page_id"] == "classification"
    assert request.sanitized_context["question"] == "[REDACTED_SECRET]"
    assert request.grounding_context["question"] == "[REDACTED_SECRET]"
    assert "/Users/jo.soroush/secret.key" not in request.question
    assert "/Users/jo.soroush/secret.key" not in str(request.grounding_context)

    with pytest.raises(GeminiProviderDisabledError, match="Phase G1"):
        stub.explain(request)


def test_gemini_provider_stub_reports_disabled_state_even_with_key_and_llm_enabled() -> None:
    stub = GeminiProviderStub(
        config=GeminiProviderConfig(enabled=True, api_key_present=True)
    )

    readiness = stub.readiness()

    assert readiness.provider_name == "gemini"
    assert readiness.availability.available is False
    assert readiness.availability.status == "disabled"
    assert readiness.availability.configured is True
    assert "Phase G1" in readiness.availability.reason
    assert "secret" not in readiness.availability.reason.lower()
    assert "API key presence" in readiness.availability.reason
    assert readiness.fallback_policy.allow_mock_fallback is True
    assert readiness.fallback_policy.fallback_provider == "mock"
    assert any("Phase G1" in warning for warning in readiness.warnings)


def test_gemini_provider_readiness_keeps_gemini_unavailable_for_runtime_settings() -> None:
    readiness = evaluate_gemini_provider_readiness(
        ProviderRuntimeSettings(
            enable_llm=True,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=False,
            openai_key_present=False,
        )
    )

    assert readiness.availability.available is False
    assert readiness.availability.status == "disabled"
    assert readiness.fallback_policy.allow_mock_fallback is True
    assert readiness.fallback_policy.fallback_provider == "mock"
    assert "Phase G1" in readiness.availability.reason


def test_gemini_provider_stub_does_not_import_provider_sdks_or_network_libraries() -> None:
    tree = ast.parse(STUB_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    banned_roots = {"google", "openai", "requests", "httpx", "urllib"}
    assert imported_roots.isdisjoint(banned_roots)


def test_gemini_stub_module_text_does_not_reference_provider_sdk_imports() -> None:
    text = STUB_PATH.read_text(encoding="utf-8").lower()

    assert "google.generativeai" not in text
    assert "google.genai" not in text
    assert "openai" not in text
    assert "requests" not in text
    assert "httpx" not in text
