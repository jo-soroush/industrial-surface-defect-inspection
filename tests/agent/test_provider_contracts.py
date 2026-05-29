"""Tests for the provider contract and readiness layer."""

from __future__ import annotations

from src.inspection_ai.agent.context_builder import build_grounding_context
from src.inspection_ai.agent.provider_contracts import (
    AgentProviderResponse,
    AgentProviderStatus,
    ProviderFallbackPolicy,
    ProviderReadinessResult,
    ProviderRuntimeSettings,
    build_provider_request,
    build_provider_response,
    evaluate_provider_readiness,
)


def test_provider_contract_objects_can_be_constructed() -> None:
    context = build_grounding_context(
        page_id="classification",
        section_id="detailed_metrics",
        component_id="classification_threshold_curve_chart",
        question="What does this chart mean?",
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )

    request = build_provider_request(
        provider_name="mock",
        grounding_context=context,
        sanitized_context={"page_id": "classification"},
        safety_status="pass",
        llm_enabled=False,
    )
    response = build_provider_response(
        answer="Safe answer",
        provider_used="mock",
        fallback_used=True,
        fallback_reason="Mock fallback is the MVP fallback.",
        provider_error_stage="readiness",
        provider_error_reason="provider_error",
        safety_block_reason="unknown",
        grounding_status="grounded",
        safety_status="pass",
        limitations=["Manual review still applies."],
        evidence_used=[{"source": "request.page_id", "value": "classification"}],
    )

    assert request.provider_name == "mock"
    assert request.component_id == "classification_threshold_curve_chart"
    assert request.question == "What does this chart mean?"
    assert request.sanitized_context["page_id"] == "classification"
    assert request.grounding_context["page_id"] == "classification"
    assert request.grounding_context["component_id"] == "classification_threshold_curve_chart"
    assert isinstance(response, AgentProviderResponse)
    assert response.raw_provider_response_allowed is False
    assert response.fallback_reason == "Mock fallback is the MVP fallback."
    assert response.provider_error_stage == "readiness"
    assert response.provider_error_reason == "provider_error"
    assert response.safety_block_reason == "unknown"
    assert response.provider_used == "mock"


def test_provider_request_uses_sanitized_question_when_available() -> None:
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
        provider_name="mock",
        grounding_context=context,
        sanitized_context={
            "page_id": "classification",
            "question": "[REDACTED_SECRET]",
        },
        safety_status="pass",
        llm_enabled=False,
    )

    assert request.question == "[REDACTED_SECRET]"
    assert request.grounding_context["question"] == "[REDACTED_SECRET]"
    assert "/Users/jo.soroush/secret.key" not in request.question


def test_provider_status_objects_can_be_constructed() -> None:
    availability = AgentProviderStatus(
        provider_name="gemini",
        configured=False,
        available=False,
        status="unavailable",
        reason="API key missing.",
        warnings=("Gemini is unavailable because its API key is missing.",),
    )
    fallback_policy = ProviderFallbackPolicy(
        allow_mock_fallback=True,
        fallback_provider="mock",
        fallback_reason="Gemini is unavailable; mock fallback remains the safe path.",
    )
    readiness = ProviderReadinessResult(
        provider_name="gemini",
        availability=availability,
        fallback_policy=fallback_policy,
        warnings=("Gemini is unavailable because its API key is missing.",),
    )

    assert readiness.provider_name == "gemini"
    assert readiness.availability.status == "unavailable"
    assert readiness.fallback_policy.allow_mock_fallback is True
    assert "API key missing" in readiness.availability.reason


def test_provider_readiness_keeps_mock_available_and_real_providers_unavailable_without_keys() -> None:
    readiness = evaluate_provider_readiness(
        ProviderRuntimeSettings(
            enable_llm=False,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=False,
            grok_key_present=False,
            openai_key_present=False,
        )
    )

    assert readiness["mock"].availability.available is True
    assert readiness["mock"].availability.status == "available"
    assert readiness["gemini"].availability.available is False
    assert readiness["grok"].availability.available is False
    assert readiness["gemini"].availability.status == "disabled"
    assert readiness["grok"].availability.status == "disabled"
    assert "Gemini" in readiness["gemini"].warnings[0]
    assert "Grok" in readiness["grok"].warnings[0]


def test_provider_readiness_does_not_expose_secret_values(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-secret-value-123")
    monkeypatch.setenv("GROK_API_KEY", "xoxb-secret-value-456")

    readiness = evaluate_provider_readiness(
        ProviderRuntimeSettings(
            enable_llm=False,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=True,
            openai_key_present=False,
        )
    )

    rendered = " ".join(
        warning
        for result in readiness.values()
        for warning in result.warnings
    )

    assert "sk-secret-value-123" not in rendered
    assert "xoxb-secret-value-456" not in rendered
    assert "API key is missing" not in rendered


def test_provider_readiness_allows_ready_for_future_use_when_llm_enabled_and_keys_present() -> None:
    readiness = evaluate_provider_readiness(
        ProviderRuntimeSettings(
            enable_llm=True,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=True,
            openai_key_present=True,
        )
    )

    assert readiness["mock"].availability.available is True
    assert readiness["gemini"].availability.available is False
    assert readiness["grok"].availability.available is False
    assert readiness["openai"].availability.available is False
    assert readiness["gemini"].fallback_policy.fallback_provider == "mock"
    assert readiness["openai"].fallback_policy.allow_mock_fallback is True
