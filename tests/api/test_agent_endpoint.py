"""API tests for the Agent/RAG MVP endpoints."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.app.main import app
import src.inspection_ai.agent.provider_router as provider_router_module
from src.inspection_ai.agent.gemini_provider import GeminiSdkLoadResult
from src.inspection_ai.agent.provider_contracts import build_provider_response


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
    assert any("gemini readiness:" in warning.lower() for warning in payload["warnings"])


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
    assert payload["fallback_reason"] is not None
    assert "mock" in payload["fallback_reason"].lower()
    assert payload["provider_error_stage"] is None
    assert payload["provider_error_reason"] is None
    assert payload["safety_block_reason"] is None
    assert payload["grounding_status"] == "grounded"
    assert payload["page_id"] == "image_inspection"
    assert payload["section_id"] == "final_decision"
    assert "manual review" in payload["answer"].lower()
    assert any(item["source"] == "inspection_response.decision.final_decision" for item in payload["evidence_used"])
    assert payload["component_id"] is None


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


def test_agent_health_stays_mock_first_with_fake_key_present(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("AGENT_DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,grok,mock")
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-disabled")
    monkeypatch.setenv("GROK_API_KEY", "present-but-disabled")

    response = client.get("/agent/health")

    assert response.status_code == 200
    payload = response.json()
    joined_warnings = " ".join(payload["warnings"]).lower()
    assert payload["llm_enabled"] is False
    assert payload["default_provider"] == "mock"
    assert payload["available_providers"] == ["mock"]
    assert payload["fallback_available"] is True
    assert payload["grounding_ready"] is True
    assert "present-but-disabled" not in joined_warnings
    assert any("intentionally disabled" in warning.lower() for warning in payload["warnings"])
    assert any("gemini" in warning.lower() for warning in payload["warnings"])
    assert any("grok" in warning.lower() for warning in payload["warnings"])


def test_agent_explain_routes_through_gated_fake_gemini_when_all_explicit_gates_pass(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("AGENT_ENABLE_REAL_PROVIDER_RUNTIME", "true")
    monkeypatch.setenv("AGENT_DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,mock")
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-test-only")
    monkeypatch.delenv("GROK_API_KEY", raising=False)

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
                "default_provider": settings.default_provider,
                "sdk_loader_present": sdk_loader is not None,
                "sdk_module_loader_present": sdk_module_loader is not None,
                "allowed_evidence_values": list(allowed_evidence_values or []),
            }
        )
        return SimpleNamespace(
            provider_response=build_provider_response(
                answer="Gemini gated endpoint answer. Manual review still applies.",
                provider_used="gemini",
                fallback_used=False,
                fallback_reason=None,
                provider_error_stage=None,
                provider_error_reason=None,
                safety_block_reason=None,
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

    response = client.post(
        "/agent/explain",
        json={
            "page_id": "image_inspection",
            "section_id": "final_decision",
            "question": "Explain this image inspection result safely.",
            "visible_context": {"page_title": "Image Inspection"},
            "inspection_response": {
                "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
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
    assert len(calls) == 1
    assert calls[0]["provider_name"] == "gemini"
    assert calls[0]["enable_llm"] is True
    assert calls[0]["runtime_gate"] is True
    assert calls[0]["default_provider"] == "gemini"
    assert calls[0]["sdk_loader_present"] is True
    assert calls[0]["sdk_module_loader_present"] is True
    allowed_evidence_values = calls[0]["allowed_evidence_values"]
    assert "defective" in allowed_evidence_values
    assert "manual_check_rule" in allowed_evidence_values
    assert "defect" in allowed_evidence_values
    assert 1 in allowed_evidence_values
    assert "anomaly" in allowed_evidence_values
    assert "/inspect/image" in allowed_evidence_values
    assert "Image Inspection" in allowed_evidence_values
    assert "image_inspection" in allowed_evidence_values
    assert "final_decision" in allowed_evidence_values
    assert payload["provider_used"] == "gemini"
    assert payload["fallback_used"] is False
    assert payload["fallback_reason"] is None
    assert payload["provider_error_stage"] is None
    assert payload["provider_error_reason"] is None
    assert payload["safety_block_reason"] is None
    assert payload["grounding_status"] == "grounded"
    assert "gemini gated endpoint answer" in payload["answer"].lower()
    assert "manual review" in payload["answer"].lower()
    assert "present-but-test-only" not in payload["answer"].lower()


def test_agent_explain_exposes_safe_fallback_reason_for_gated_fake_gemini_fallback(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("AGENT_ENABLE_REAL_PROVIDER_RUNTIME", "true")
    monkeypatch.setenv("AGENT_DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,mock")
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-test-only")
    monkeypatch.delenv("GROK_API_KEY", raising=False)

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
        return SimpleNamespace(
            provider_response=build_provider_response(
                answer="Mock fallback remains the safe path. Manual review still applies.",
                provider_used="mock",
                fallback_used=True,
                fallback_reason="Gemini real provider service unavailable; mock fallback remains the safe path.",
                provider_error_stage="client_invocation",
                provider_error_reason="service_unavailable",
                safety_block_reason=None,
                grounding_status="grounded",
                safety_status="pass",
                limitations=["Manual review still applies."],
                evidence_used=[{"source": "inspection_response.decision.final_decision", "value": "defective"}],
            ),
            status="provider_error",
            safe_to_send=True,
            safe_to_display=True,
            provider_error="Gemini real provider service unavailable.",
            fallback_reason="Gemini real provider service unavailable; mock fallback remains the safe path.",
        )

    monkeypatch.setattr(
        provider_router_module,
        "generate_with_real_gemini_provider",
        fake_generate_with_real_gemini_provider,
    )

    response = client.post(
        "/agent/explain",
        json={
            "page_id": "image_inspection",
            "section_id": "final_decision",
            "question": "Explain this image inspection result safely.",
            "visible_context": {"page_title": "Image Inspection"},
            "inspection_response": {
                "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
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
    assert payload["fallback_reason"] is not None
    assert "service unavailable" in payload["fallback_reason"].lower()
    assert payload["provider_error_stage"] == "client_invocation"
    assert payload["provider_error_reason"] == "service_unavailable"
    assert payload["safety_block_reason"] is None
    assert "present-but-test-only" not in payload["fallback_reason"].lower()
    assert "present-but-test-only" not in payload["answer"].lower()


def test_agent_explain_exposes_sanitized_safety_block_reason_for_gated_fake_gemini_post_generation_block(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENT_ENABLE_LLM", "true")
    monkeypatch.setenv("AGENT_ENABLE_REAL_PROVIDER_RUNTIME", "true")
    monkeypatch.setenv("AGENT_DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,mock")
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-test-only")
    monkeypatch.delenv("GROK_API_KEY", raising=False)

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
        return SimpleNamespace(
            provider_response=build_provider_response(
                answer="Mock fallback remains the safe path. Manual review still applies.",
                provider_used="mock",
                fallback_used=True,
                fallback_reason="Gemini real provider output was blocked by the safety guard; mock fallback remains the safe path.",
                provider_error_stage="post_generation",
                provider_error_reason="safety_blocked",
                safety_block_reason="invented_metric_like_output",
                grounding_status="grounded",
                safety_status="blocked",
                limitations=["Manual review still applies."],
                evidence_used=[{"source": "inspection_response.decision.final_decision", "value": "defective"}],
            ),
            status="blocked",
            safe_to_send=False,
            safe_to_display=False,
            provider_error="Gemini real provider output was blocked by the safety guard.",
            fallback_reason="Gemini real provider output was blocked by the safety guard; mock fallback remains the safe path.",
        )

    monkeypatch.setattr(
        provider_router_module,
        "generate_with_real_gemini_provider",
        fake_generate_with_real_gemini_provider,
    )

    response = client.post(
        "/agent/explain",
        json={
            "page_id": "image_inspection",
            "section_id": "final_decision",
            "question": "Explain this image inspection result safely.",
            "visible_context": {"page_title": "Image Inspection"},
            "inspection_response": {
                "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
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
    assert payload["fallback_reason"] is not None
    assert "safety guard" in payload["fallback_reason"].lower()
    assert payload["provider_error_stage"] == "post_generation"
    assert payload["provider_error_reason"] == "safety_blocked"
    assert payload["safety_block_reason"] == "invented_metric_like_output"
    assert "present-but-test-only" not in payload["fallback_reason"].lower()
    assert "present-but-test-only" not in payload["answer"].lower()


def test_agent_explain_accepts_classification_component_id() -> None:
    response = client.post(
        "/agent/explain",
        json={
            "page_id": "classification",
            "section_id": "detailed_metrics",
            "component_id": "classification_threshold_curve_chart",
            "question": "What does this chart mean?",
            "visible_context": {},
            "inspection_response": {},
            "include_raw_evidence": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["component_id"] == "classification_threshold_curve_chart"
    assert payload["provider_used"] == "mock"
    assert payload["grounding_status"] == "grounded"
    assert any(
        item["source"] == "artifacts/frontend/track_a/threshold_curve_chart_data.json#recommended_threshold"
        for item in payload["evidence_used"]
    )


def test_agent_explain_accepts_detection_component_id() -> None:
    response = client.post(
        "/agent/explain",
        json={
            "page_id": "detection",
            "section_id": "visual_evidence",
            "component_id": "detection_confidence_chart",
            "question": "What does detection confidence mean?",
            "visible_context": {},
            "inspection_response": {},
            "include_raw_evidence": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["component_id"] == "detection_confidence_chart"
    assert any(
        item["source"]
        == "artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json#confidence_bins.count"
        for item in payload["evidence_used"]
    )


def test_agent_explain_rejects_invalid_component_id_safely() -> None:
    response = client.post(
        "/agent/explain",
        json={
            "page_id": "classification",
            "section_id": "detailed_metrics",
            "component_id": "not_a_real_component",
            "question": "Explain this.",
            "visible_context": {},
            "inspection_response": {},
            "include_raw_evidence": False,
        },
    )

    assert response.status_code == 422


def test_agent_explain_existing_image_inspection_request_without_component_id_still_works() -> None:
    response = client.post(
        "/agent/explain",
        json={
            "page_id": "image_inspection",
            "section_id": "final_decision",
            "question": "Explain this inspection result.",
            "visible_context": {"page_title": "Image Inspection"},
            "inspection_response": {
                "decision": {"final_decision": "good", "rule_id": "manual_check_rule"},
                "traceability": {"source_endpoint": "/inspect/image"},
            },
            "include_raw_evidence": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["component_id"] is None
    assert payload["grounding_status"] == "grounded"
    assert any(item["source"] == "inspection_response.decision.final_decision" for item in payload["evidence_used"])


def test_agent_explain_accepts_image_inspection_ai_panel_component_id() -> None:
    response = client.post(
        "/agent/explain",
        json={
            "page_id": "image_inspection",
            "section_id": "final_decision",
            "component_id": "image_inspection_ai_explanation_panel",
            "question": "Explain this inspection result.",
            "visible_context": {"page_title": "Image Inspection"},
            "inspection_response": {
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
                "explanation_context": {"context_version": "image_inspection_explanation_context_v0_1"},
            },
            "include_raw_evidence": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["component_id"] == "image_inspection_ai_explanation_panel"
    assert payload["provider_used"] == "mock"
    assert payload["grounding_status"] == "grounded"
    assert any(item["source"] == "inspection_response#decision" for item in payload["evidence_used"])
