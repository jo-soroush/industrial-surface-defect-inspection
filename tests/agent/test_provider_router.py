"""Tests for the agent provider router and mock fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace

from src.inspection_ai.agent.context_builder import build_grounding_context
import src.inspection_ai.agent.provider_router as provider_router_module
from src.inspection_ai.agent.provider_router import AgentProviderRouter, AgentProviderSettings


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
