"""API tests for the Agent/RAG MVP endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app.main import app


client = TestClient(app)


def test_agent_health_reports_mock_only_mvp_state() -> None:
    response = client.get("/agent/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "industrial-surface-defect-agent"
    assert payload["agent_ready"] is True
    assert payload["llm_enabled"] is False
    assert payload["default_provider"] == "mock"
    assert payload["available_providers"] == ["mock"]
    assert payload["fallback_available"] is True
    assert payload["grounding_ready"] is True


def test_agent_explain_returns_grounded_mock_answer_for_image_inspection() -> None:
    response = client.post(
        "/agent/explain",
        json={
            "page_id": "image_inspection",
            "section_id": "final_decision",
            "question": "Why is this image defective?",
            "visible_context": {"page_title": "Image Inspection"},
            "inspection_response": {
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
            "include_raw_evidence": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_used"] == "mock"
    assert payload["fallback_used"] is True
    assert payload["grounding_status"] == "grounded"
    assert payload["page_id"] == "image_inspection"
    assert payload["section_id"] == "final_decision"
    assert "manual review" in payload["answer"].lower()
    assert any(item["source"] == "inspection_response.decision.final_decision" for item in payload["evidence_used"])


def test_agent_explain_rejects_unsupported_page_section_pair() -> None:
    response = client.post(
        "/agent/explain",
        json={
            "page_id": "image_inspection",
            "section_id": "not_a_real_section",
            "question": "Explain this.",
            "visible_context": {},
            "inspection_response": {},
            "include_raw_evidence": False,
        },
    )

    assert response.status_code == 422


def test_agent_explain_rejects_missing_page_id() -> None:
    response = client.post(
        "/agent/explain",
        json={
            "section_id": "final_decision",
            "question": "Why is this image defective?",
            "visible_context": {},
            "inspection_response": {},
            "include_raw_evidence": False,
        },
    )

    assert response.status_code == 422


def test_agent_health_stays_mock_first_when_llm_is_requested(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("AGENT_DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,grok,mock")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)

    response = client.get("/agent/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_enabled"] is False
    assert payload["default_provider"] == "mock"
    assert payload["available_providers"] == ["mock"]
    assert payload["fallback_available"] is True
    assert any("intentionally disabled" in warning.lower() for warning in payload["warnings"])
    assert any("gemini" in warning.lower() for warning in payload["warnings"])
    assert any("grok" in warning.lower() for warning in payload["warnings"])
