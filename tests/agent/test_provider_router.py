"""Tests for the agent provider router and mock fallback behavior."""

from __future__ import annotations

from src.inspection_ai.agent.context_builder import build_grounding_context
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
    assert "human review" in review_response.answer.lower()
    assert deploy_response.provider_used == "mock"
    assert review_response.provider_used == "mock"
