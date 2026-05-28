"""Tests for the agent provider router and mock fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.inspection_ai.agent.context_builder import build_grounding_context
from src.inspection_ai.agent.gemini_provider import GeminiSdkLoadResult, evaluate_gemini_g3_readiness
import src.inspection_ai.agent.provider_router as provider_router_module
from src.inspection_ai.agent.provider_router import AgentProviderRouter, AgentProviderSettings
from src.inspection_ai.agent.provider_contracts import build_provider_response


def test_health_reports_mock_first_mvp_state() -> None:
    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=False,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key=None,
            grok_api_key=None,
        )
    )

    health = router.health()
    assert health.status == "ok"
    assert health.agent_ready is True
    assert health.llm_enabled is False
    assert health.default_provider == "mock"
    assert health.provider_order == ["mock", "gemini", "grok"]
    assert health.available_providers == ["mock"]
    assert health.fallback_available is True
    assert health.grounding_ready is True
    assert any("phase g1" in warning.lower() for warning in health.warnings)
    assert any("gemini readiness:" in warning.lower() for warning in health.warnings)
    gemini_readiness = router.gemini_readiness()
    assert gemini_readiness.status == "disabled"
    assert gemini_readiness.available is False
    assert gemini_readiness.gates.llm_enabled is False
    assert gemini_readiness.gates.api_key_present is False
    assert gemini_readiness.gates.sdk_checked is False
    assert gemini_readiness.gates.sdk_status == "not_checked"
    assert gemini_readiness.gates.activation_allowed is False
    assert gemini_readiness.gates.real_provider_implemented is False


def test_missing_provider_keys_do_not_break_mock_health(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("LLM_ENABLE_FALLBACK", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,grok,mock")

    settings = AgentProviderSettings.from_env()
    router = AgentProviderRouter(settings)
    health = router.health()

    assert settings.enable_llm is True
    assert settings.gemini_api_key is None
    assert settings.grok_api_key is None
    assert health.available_providers == ["mock"]
    assert health.default_provider == "mock"
    assert health.provider_order == ["gemini", "grok", "mock"]
    assert health.status == "ok"
    assert health.llm_enabled is False
    assert health.fallback_available is True
    assert any("mock fallback" in warning.lower() for warning in health.warnings)
    assert any("mandatory" in warning.lower() for warning in health.warnings)
    assert any("intentionally disabled" in warning.lower() for warning in health.warnings)
    assert any("gemini" in warning.lower() for warning in health.warnings)
    assert any("grok" in warning.lower() for warning in health.warnings)
    assert any("phase g1" in warning.lower() for warning in health.warnings)
    assert any("gemini readiness:" in warning.lower() for warning in health.warnings)

    gemini_readiness = router.gemini_readiness()
    assert gemini_readiness.status == "unavailable"
    assert gemini_readiness.available is False
    assert gemini_readiness.gates.llm_enabled is True
    assert gemini_readiness.gates.api_key_present is False
    assert gemini_readiness.gates.sdk_checked is False
    assert gemini_readiness.gates.sdk_status == "not_checked"
    assert gemini_readiness.gates.activation_allowed is False
    assert gemini_readiness.gates.real_provider_implemented is False


def test_settings_from_env_reads_explicit_real_provider_runtime_flag(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("AGENT_ENABLE_REAL_PROVIDER_RUNTIME", "true")
    monkeypatch.setenv("AGENT_DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,mock")
    monkeypatch.setenv("GEMINI_API_KEY", "present")
    monkeypatch.delenv("GROK_API_KEY", raising=False)

    settings = AgentProviderSettings.from_env()

    assert settings.enable_llm is True
    assert settings.enable_real_provider_runtime is True
    assert settings.default_provider == "gemini"
    assert settings.provider_order == ("gemini", "mock")
    assert settings.gemini_api_key == "present"


def test_gemini_health_metadata_does_not_expose_raw_key_values(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-test-key")
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mock,gemini,grok")

    router = AgentProviderRouter()
    health = router.health()
    gemini_readiness = router.gemini_readiness()

    joined_warnings = " ".join(health.warnings).lower()
    assert "super-secret-test-key" not in joined_warnings
    assert "super-secret-test-key" not in repr(gemini_readiness).lower()
    assert gemini_readiness.gates.api_key_present is True
    assert any("api_key_present=true" in warning.lower() for warning in health.warnings)
    assert any("gemini readiness:" in warning.lower() for warning in health.warnings)


def test_health_stays_mock_first_with_fake_key_present_and_llm_requested(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-disabled")
    monkeypatch.setenv("GROK_API_KEY", "present-but-disabled")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,grok,mock")

    router = AgentProviderRouter()
    health = router.health()
    gemini_readiness = router.gemini_readiness()

    joined_warnings = " ".join(health.warnings).lower()
    assert health.status == "ok"
    assert health.llm_enabled is False
    assert health.default_provider == "mock"
    assert health.available_providers == ["mock"]
    assert health.fallback_available is True
    assert "present-but-disabled" not in joined_warnings
    assert "present-but-disabled" not in repr(gemini_readiness).lower()
    assert gemini_readiness.gates.api_key_present is True
    assert gemini_readiness.gates.activation_allowed is False
    assert any("gemini" in warning.lower() for warning in health.warnings)
    assert any("grok" in warning.lower() for warning in health.warnings)
    assert any("api_key_present=true" in warning.lower() for warning in health.warnings)


def test_gemini_route_decision_stays_mock_when_llm_disabled_even_with_key() -> None:
    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=False,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present-but-disabled",
            grok_api_key=None,
        )
    )

    decision = router.gemini_route_decision(requested_provider="gemini")

    assert decision.requested_provider == "gemini"
    assert decision.selected_provider == "mock"
    assert decision.should_route_to_gemini is False
    assert decision.fallback_used is True
    assert "mock fallback" in decision.fallback_reason.lower()
    assert decision.llm_enabled is False
    assert decision.api_key_present is True
    assert decision.activation_allowed is False


def test_gemini_route_decision_stays_mock_when_key_is_missing() -> None:
    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key=None,
            grok_api_key=None,
        )
    )

    decision = router.gemini_route_decision(requested_provider="gemini")

    assert decision.requested_provider == "gemini"
    assert decision.selected_provider == "mock"
    assert decision.should_route_to_gemini is False
    assert decision.fallback_used is True
    assert "mock fallback" in decision.fallback_reason.lower()
    assert decision.llm_enabled is True
    assert decision.api_key_present is False
    assert decision.sdk_checked is False
    assert decision.activation_allowed is False


def test_gemini_route_decision_stays_mock_when_real_provider_is_not_implemented() -> None:
    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present",
            grok_api_key=None,
        )
    )

    decision = router.gemini_route_decision(requested_provider="gemini")

    assert decision.requested_provider == "gemini"
    assert decision.selected_provider == "mock"
    assert decision.should_route_to_gemini is False
    assert decision.real_provider_implemented is False
    assert decision.activation_allowed is False
    assert decision.fallback_used is True
    assert "gated" in decision.reason.lower()


def test_gemini_route_decision_stays_mock_when_sdk_is_available_but_activation_is_disabled() -> None:
    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present",
            grok_api_key=None,
        )
    )
    readiness = evaluate_gemini_g3_readiness(router._provider_runtime_settings(), sdk_available=True)
    decision = router.gemini_route_decision(requested_provider="gemini", readiness=readiness)

    assert decision.requested_provider == "gemini"
    assert decision.selected_provider == "mock"
    assert decision.should_route_to_gemini is False
    assert decision.sdk_available is True
    assert decision.activation_allowed is False
    assert decision.fallback_used is True


def test_gemini_route_decision_can_route_when_runtime_is_explicitly_enabled_and_sdk_is_available(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_router_module,
        "_load_runtime_gemini_sdk_status",
        lambda: GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai SDK import succeeded.",
        ),
    )

    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            enable_real_provider_runtime=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present",
            grok_api_key=None,
        )
    )

    readiness = router.gemini_readiness()
    decision = router.gemini_route_decision(requested_provider="gemini", readiness=readiness)

    assert readiness.gates.sdk_available is True
    assert readiness.gates.real_provider_implemented is True
    assert readiness.gates.activation_allowed is True
    assert decision.requested_provider == "gemini"
    assert decision.selected_provider == "gemini"
    assert decision.should_route_to_gemini is True
    assert decision.fallback_used is False
    assert "allowed" in decision.reason.lower()


def test_router_explain_stays_mock_first_with_fake_key_present(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-disabled")
    monkeypatch.setenv("GROK_API_KEY", "present-but-disabled")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,grok,mock")

    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Why is this image defective?",
        visible_context={},
        inspection_response={
            "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert response.grounding_status == "grounded"
    assert "manual review" in response.answer.lower()
    assert "present-but-disabled" not in response.answer.lower()


def test_mock_provider_grounded_answer_mentions_manual_review() -> None:
    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Why is this image defective?",
        visible_context={},
        inspection_response={
            "decision": {
                "final_decision": "defective",
                "decision_level": "review",
                "rule_id": "classification_detection_agree_v0",
            },
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert response.page_id == "image_inspection"
    assert response.section_id == "final_decision"
    assert response.grounding_status == "grounded"
    assert "manual review" in response.answer.lower()
    assert any(item.source == "inspection_response.decision.final_decision" for item in response.evidence_used)
    assert any("production-ready" in limitation.lower() for limitation in response.limitations)
    assert any("mock backend agent" in limitation.lower() for limitation in response.limitations)
    assert any("external llm" in limitation.lower() for limitation in response.limitations)
    assert any("no real llm provider call" in limitation.lower() for limitation in response.limitations)
    assert all("no backend agent" not in limitation.lower() for limitation in response.limitations)
    assert all("planned / not active" not in limitation.lower() for limitation in response.limitations)
    assert all("no backend agent or llm call" not in limitation.lower() for limitation in response.limitations)


def test_mock_provider_rejects_forbidden_claim_requests() -> None:
    router = AgentProviderRouter()
    deploy_context = build_grounding_context(
        page_id="safety",
        section_id="boundaries",
        question="Can I deploy this safely?",
        visible_context={"summary": "Safety boundaries"},
        inspection_response={},
        include_raw_evidence=False,
    )
    review_context = build_grounding_context(
        page_id="safety",
        section_id="manual_review",
        question="Can this replace human review?",
        visible_context={"summary": "Safety boundaries"},
        inspection_response={},
        include_raw_evidence=False,
    )

    deploy_response = router.explain(deploy_context)
    review_response = router.explain(review_context)

    assert deploy_response.grounding_status == "unsupported"
    assert review_response.grounding_status == "unsupported"
    assert "deployment" in deploy_response.answer.lower()
    assert "manual review still applies" in review_response.answer.lower()
    assert deploy_response.provider_used == "mock"
    assert review_response.provider_used == "mock"


def test_component_image_inspection_mock_answer_mentions_decision_and_manual_review() -> None:
    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        component_id="image_inspection_ai_explanation_panel",
        question="Explain this inspection result.",
        visible_context={},
        inspection_response={
            "request_id": "request-0001",
            "decision": {
                "final_decision": "good",
                "rule_id": "manual_check_rule",
                "recommended_action": "manual_review",
            },
            "classification": {"predicted_label": "good"},
            "detection": {"predicted_box_count": 0},
            "anomaly": {"quality_status": "review_required_weak_evidence"},
            "traceability": {"source_endpoint": "/inspect/image"},
            "warnings": [],
            "limitations": ["manual review required"],
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert response.component_id == "image_inspection_ai_explanation_panel"
    assert "image inspection ai explanation panel" in response.answer.lower()
    assert "good" in response.answer.lower()
    assert "manual review" in response.answer.lower()
    assert "mock/offline" in response.answer.lower()


def test_router_explain_routes_to_gemini_with_explicit_runtime_gate_and_fake_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_router_module,
        "_load_runtime_gemini_sdk_status",
        lambda: GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai SDK import succeeded.",
        ),
    )

    calls: list[dict[str, object]] = []

    def fake_generate_with_real_gemini_provider(
        request,
        *,
        settings,
        config,
        sdk_loader=None,
        sdk_module_loader=None,
        client_factory=None,
        allowed_evidence_values=None,
    ):
        calls.append(
            {
                "provider_name": request.provider_name,
                "enable_llm": settings.enable_llm,
                "runtime_gate": settings.enable_real_provider_runtime,
                "config_runtime_gate": config.real_provider_implemented,
                "sdk_import_allowed": config.sdk_import_allowed,
                "sdk_loader_present": sdk_loader is not None,
                "sdk_module_loader_present": sdk_module_loader is not None,
                "client_factory_present": client_factory is not None,
                "allowed_evidence_values": list(allowed_evidence_values or []),
            }
        )
        return SimpleNamespace(
            provider_response=build_provider_response(
                answer="Gemini gated answer. Manual review still applies.",
                provider_used="gemini",
                fallback_used=False,
                fallback_reason=None,
                provider_error_stage=None,
                provider_error_reason=None,
                grounding_status="grounded",
                safety_status="pass",
                limitations=["Manual review still applies."],
                evidence_used=[{"source": "inspection_response.decision.final_decision", "value": "defective"}],
            ),
            status="pass",
            safe_to_send=True,
            safe_to_display=True,
            provider_error=None,
            fallback_reason=None,
        )

    monkeypatch.setattr(
        provider_router_module,
        "generate_with_real_gemini_provider",
        fake_generate_with_real_gemini_provider,
    )

    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            enable_real_provider_runtime=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present",
            grok_api_key=None,
        )
    )
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {
                "final_decision": "defective",
                "rule_id": "manual_check_rule",
            },
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert len(calls) == 1
    assert calls[0]["provider_name"] == "gemini"
    assert calls[0]["enable_llm"] is True
    assert calls[0]["runtime_gate"] is True
    assert calls[0]["config_runtime_gate"] is True
    assert calls[0]["sdk_loader_present"] is True
    assert calls[0]["sdk_module_loader_present"] is True
    assert calls[0]["client_factory_present"] is False
    assert response.provider_used == "gemini"
    assert response.fallback_used is False
    assert response.fallback_reason is None
    assert response.provider_error_stage is None
    assert response.provider_error_reason is None
    assert response.grounding_status == "grounded"
    assert "gemini gated answer" in response.answer.lower()
    assert "manual review" in response.answer.lower()


def test_router_explain_stays_mock_when_runtime_enabled_but_sdk_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_router_module,
        "_load_runtime_gemini_sdk_status",
        lambda: GeminiSdkLoadResult(
            checked=True,
            sdk_available=False,
            status="missing",
            reason="google-genai SDK is missing.",
            error_category="missing",
        ),
    )

    def fake_generate_with_real_gemini_provider(*args, **kwargs):
        raise AssertionError("Gemini should not be called when the SDK is missing.")

    monkeypatch.setattr(
        provider_router_module,
        "generate_with_real_gemini_provider",
        fake_generate_with_real_gemini_provider,
    )

    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            enable_real_provider_runtime=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present",
            grok_api_key=None,
        )
    )
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert response.provider_error_stage == "readiness"
    assert response.provider_error_reason == "sdk_missing"
    assert "manual review" in response.answer.lower()


def test_router_explain_stays_mock_when_runtime_enabled_but_fallback_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_router_module,
        "_load_runtime_gemini_sdk_status",
        lambda: GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai SDK import succeeded.",
        ),
    )

    def fake_generate_with_real_gemini_provider(*args, **kwargs):
        raise AssertionError("Gemini should not be called when fallback is disabled.")

    monkeypatch.setattr(
        provider_router_module,
        "generate_with_real_gemini_provider",
        fake_generate_with_real_gemini_provider,
    )

    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            enable_real_provider_runtime=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=False,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present",
            grok_api_key=None,
        )
    )
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert "manual review" in response.answer.lower()


@pytest.mark.parametrize(
    ("boundary_status", "fallback_reason"),
    [
        ("provider_error", "Gemini real provider service unavailable; mock fallback remains the safe path."),
        ("rate_limit", "Gemini real provider rate limited; mock fallback remains the safe path."),
        ("timeout", "Gemini real provider timed out; mock fallback remains the safe path."),
    ],
)
def test_router_explain_falls_back_safely_when_real_provider_boundary_returns_mock_result(
    monkeypatch,
    boundary_status: str,
    fallback_reason: str,
) -> None:
    monkeypatch.setattr(
        provider_router_module,
        "_load_runtime_gemini_sdk_status",
        lambda: GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai SDK import succeeded.",
        ),
    )

    calls: list[str] = []

    def fake_generate_with_real_gemini_provider(*args, **kwargs):
        calls.append("called")
        return SimpleNamespace(
            provider_response=build_provider_response(
                answer="Mock fallback remains the safe path. Manual review still applies.",
                provider_used="mock",
                fallback_used=True,
                fallback_reason=fallback_reason,
                provider_error_stage="client_invocation",
                provider_error_reason=boundary_status,
                grounding_status="grounded",
                safety_status="pass",
                limitations=["Manual review still applies."],
                evidence_used=[],
            ),
            status=boundary_status,
            safe_to_send=True,
            safe_to_display=True,
            provider_error=f"Gemini real provider {boundary_status.replace('_', ' ')}.",
            fallback_reason=fallback_reason,
        )

    monkeypatch.setattr(
        provider_router_module,
        "generate_with_real_gemini_provider",
        fake_generate_with_real_gemini_provider,
    )

    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            enable_real_provider_runtime=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present",
            grok_api_key=None,
        )
    )
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert calls == ["called"]
    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert response.fallback_reason == fallback_reason
    assert response.provider_error_stage == "client_invocation"
    assert response.provider_error_reason == boundary_status
    assert "mock fallback" in response.answer.lower()
    assert "manual review" in response.answer.lower()


def test_router_explain_keeps_safety_guard_before_gemini_route(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_router_module,
        "_load_runtime_gemini_sdk_status",
        lambda: GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai SDK import succeeded.",
        ),
    )

    def fake_generate_with_real_gemini_provider(*args, **kwargs):
        raise AssertionError("Gemini should not be called when the safety guard blocks the request.")

    monkeypatch.setattr(
        provider_router_module,
        "generate_with_real_gemini_provider",
        fake_generate_with_real_gemini_provider,
    )

    router = AgentProviderRouter(
        AgentProviderSettings(
            enable_llm=True,
            enable_real_provider_runtime=True,
            default_provider="gemini",
            provider_order=("gemini", "mock"),
            enable_fallback=True,
            timeout_seconds=20,
            max_retries=1,
            gemini_api_key="present",
            grok_api_key=None,
        )
    )
    grounding_context = build_grounding_context(
        page_id="safety",
        section_id="boundaries",
        question="Can I deploy this safely?",
        visible_context={"summary": "Safety boundaries"},
        inspection_response={},
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert response.fallback_reason is not None
    assert "mock fallback" in response.fallback_reason.lower()
    assert response.provider_error_stage == "pre_generation"
    assert response.provider_error_reason == "safety_blocked"
    assert "deployment" in response.answer.lower()
    assert "manual review" in response.answer.lower()


def test_component_detection_confidence_mock_answer_mentions_confidence_and_review() -> None:
    response = _component_response(
        page_id="detection",
        section_id="visual_evidence",
        component_id="detection_confidence_chart",
        question="What does this confidence chart mean?",
    )

    assert response.provider_used == "mock"
    assert response.component_id == "detection_confidence_chart"
    assert "confidence" in response.answer.lower()
    assert "yolo detection evidence" in response.answer.lower()
    assert "do not replace review" in response.answer.lower()


def test_component_anomaly_threshold_mock_answer_mentions_weak_review_only_boundary() -> None:
    response = _component_response(
        page_id="anomaly",
        section_id="visual_evidence",
        component_id="anomaly_threshold_behavior_chart",
        question="What does this anomaly threshold mean?",
    )

    assert response.provider_used == "mock"
    assert response.component_id == "anomaly_threshold_behavior_chart"
    assert "weak/review-only" in response.answer.lower()
    assert "supporting review evidence" in response.answer.lower()


def test_component_classification_threshold_mock_answer_mentions_validation_threshold() -> None:
    response = _component_response(
        page_id="classification",
        section_id="detailed_metrics",
        component_id="classification_threshold_curve_chart",
        question="What does this threshold chart mean?",
    )

    assert response.provider_used == "mock"
    assert response.component_id == "classification_threshold_curve_chart"
    assert "validation evidence" in response.answer.lower()
    assert "threshold" in response.answer.lower()
    assert "production-ready" not in response.answer.lower()
    assert "deployment-safe" not in response.answer.lower()


def test_component_mock_answers_do_not_claim_readiness_or_provider_integration() -> None:
    responses = [
        _component_response("classification", "detailed_metrics", "classification_threshold_curve_chart"),
        _component_response("anomaly", "visual_evidence", "anomaly_threshold_behavior_chart"),
        _component_response("detection", "visual_evidence", "detection_confidence_chart"),
    ]

    for response in responses:
        normalized_answer = response.answer.lower()
        assert response.provider_used == "mock"
        assert response.fallback_used is True
        assert response.provider_error_stage is None
        assert response.provider_error_reason is None
        assert "production-ready" not in normalized_answer
        assert "deployment-safe" not in normalized_answer
        assert "gemini" not in normalized_answer
        assert "grok" not in normalized_answer
        assert "openai" not in normalized_answer


def test_non_component_image_inspection_mock_answer_still_works() -> None:
    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Why is this image defective?",
        visible_context={},
        inspection_response={
            "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.component_id is None
    assert response.provider_used == "mock"
    assert response.provider_error_stage is None
    assert response.provider_error_reason is None
    assert response.grounding_status == "grounded"
    assert "inspection result is defective" in response.answer.lower()
    assert "manual review" in response.answer.lower()


def test_mock_provider_routes_through_safety_guard(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_pre_generation_guard(grounding_context):
        calls.append(("pre", grounding_context.component_id))
        return SimpleNamespace(
            blocked=False,
            status="pass",
            sanitized_text=None,
            reasons=(),
            warnings=(),
            limitations=(),
            sanitized_context={},
            safe_to_send=True,
            safe_to_display=True,
        )

    def fake_post_generation_guard(answer_text, *, grounding_context=None, allowed_evidence_values=None):
        calls.append(("post", grounding_context.component_id if grounding_context else None))
        return SimpleNamespace(
            blocked=False,
            status="pass",
            sanitized_text=answer_text,
            reasons=(),
            warnings=(),
            limitations=(),
            sanitized_context={},
            safe_to_send=True,
            safe_to_display=True,
        )

    monkeypatch.setattr(
        provider_router_module,
        "guard_pre_generation_context",
        fake_pre_generation_guard,
    )
    monkeypatch.setattr(
        provider_router_module,
        "guard_post_generation_text",
        fake_post_generation_guard,
    )

    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Why is this image defective?",
        visible_context={},
        inspection_response={
            "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert calls[0] == ("pre", None)
    assert calls[1] == ("post", None)
    assert response.provider_used == "mock"
    assert response.grounding_status == "grounded"


def test_ai_assistant_copy_uses_current_mock_agent_language() -> None:
    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id="ai_assistant",
        section_id="preview_status",
        question="What is the assistant state?",
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    normalized_answer = response.answer.lower()
    assert "mock backend agent" in normalized_answer
    assert "external llm" in normalized_answer
    assert "planned future capability" not in normalized_answer
    assert "planned / not active" not in normalized_answer


def _component_response(
    page_id: str,
    section_id: str,
    component_id: str,
    question: str = "Explain this component.",
):
    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id=page_id,
        section_id=section_id,
        component_id=component_id,
        question=question,
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )
    return router.explain(grounding_context)
