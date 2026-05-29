from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from frontend import streamlit_app as app
from frontend.streamlit_app import (
    _build_agent_explanation_diagnostic_items,
    _agent_explanation_status_caption,
    _annotate_detection_boxes,
    _build_image_inspection_agent_request,
    _build_image_inspection_agent_inspection_response,
    _call_image_inspection_api,
    _call_agent_explain_api,
    _extract_detection_rows,
    _friendly_metric_display,
    _get_api_base_url,
)
from src.inspection_ai.agent.context_builder import build_grounding_context
from src.inspection_ai.agent.safety_guard import guard_pre_generation_context


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeStreamlitContext:
    def __init__(self, recorder: list[tuple[str, object]]) -> None:
        self._recorder = recorder

    def __enter__(self) -> "_FakeStreamlitContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeStreamlit:
    def __init__(self, submitted: bool = False) -> None:
        self.submitted = submitted
        self.session_state: dict[str, object] = {}
        self.calls: list[tuple[str, object]] = []

    def markdown(self, text: str) -> None:
        self.calls.append(("markdown", text))

    def caption(self, text: str) -> None:
        self.calls.append(("caption", text))

    def form(self, name: str, clear_on_submit: bool = False) -> _FakeStreamlitContext:
        self.calls.append(("form", name))
        return _FakeStreamlitContext(self.calls)

    def text_input(self, label: str, key: str | None = None) -> str:
        self.calls.append(("text_input", f"{label}:{key}"))
        if key is not None and key not in self.session_state:
            self.session_state[key] = "Explain this image inspection result safely for manual review."
        return str(self.session_state.get(key, "Explain this image inspection result safely for manual review."))

    def form_submit_button(self, label: str, type: str = "secondary") -> bool:
        self.calls.append(("submit", label))
        return self.submitted

    def warning(self, text: str) -> None:
        self.calls.append(("warning", text))

    def info(self, text: str) -> None:
        self.calls.append(("info", text))

    def write(self, text: str) -> None:
        self.calls.append(("write", text))

    def json(self, payload: object) -> None:
        self.calls.append(("json", payload))

    def expander(self, label: str, expanded: bool = False) -> _FakeStreamlitContext:
        self.calls.append(("expander", f"{label}:{expanded}"))
        return _FakeStreamlitContext(self.calls)

    def columns(self, count: int) -> list[_FakeStreamlitContext]:
        self.calls.append(("columns", count))
        return [_FakeStreamlitContext(self.calls) for _ in range(count)]

    def metric(self, label: str, value: str, help: str | None = None) -> None:
        self.calls.append(("metric", (label, value)))


def test_image_inspection_api_call_uses_unified_endpoint(monkeypatch) -> None:
    captured = {}

    def fake_post(endpoint, files, timeout):
        captured["endpoint"] = endpoint
        captured["files"] = files
        captured["timeout"] = timeout
        return _FakeResponse(
            200,
            {
                "request_id": "request-0001",
                "timestamp_utc": "2026-05-21T12:00:00Z",
                "input": {
                    "filename": "surface.png",
                    "content_type": "image/png",
                    "file_size_bytes": 100,
                    "image_width": 100,
                    "image_height": 100,
                    "image_mode": "RGB",
                    "preprocessing_notes": [],
                },
                "classification": {"status": "success"},
                "detection": {"status": "success", "detections": []},
                "anomaly": {"status": "success"},
                "decision": {"status": "success", "final_decision": "needs_manual_review"},
                "traceability": {"source_endpoint": "/inspect/image"},
                "limitations": [],
                "errors": [],
                "warnings": [],
                "explanation_context": {
                    "status": "available",
                    "context_version": "image_inspection_explanation_context_v0_1",
                    "allowed_sources": [],
                    "summary_inputs": {},
                    "safety_boundaries": [],
                    "forbidden_claims": [],
                },
            },
        )

    monkeypatch.setattr("frontend.streamlit_app.requests.post", fake_post)

    uploaded = SimpleNamespace(
        name="surface.png",
        type="image/png",
        getvalue=lambda: _png_bytes(),
    )
    payload = _call_image_inspection_api("http://localhost:8000", uploaded)

    assert captured["endpoint"] == "http://localhost:8000/inspect/image"
    assert "file" in captured["files"]
    assert captured["timeout"] == 120
    assert payload["decision"]["final_decision"] == "needs_manual_review"
    assert payload["traceability"]["source_endpoint"] == "/inspect/image"


def test_frontend_api_base_url_defaults_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("STREAMLIT_API_BASE_URL", raising=False)

    assert _get_api_base_url() == "http://localhost:8000"


def test_frontend_api_base_url_uses_env_override_and_trims_trailing_slash(monkeypatch) -> None:
    monkeypatch.setenv("STREAMLIT_API_BASE_URL", "http://api:8000/")

    assert _get_api_base_url() == "http://api:8000"


def test_agent_explain_api_call_uses_unified_endpoint(monkeypatch) -> None:
    captured = {}
    request_payload = _build_image_inspection_agent_request(
        question="Explain this inspection result.",
        inspection_response={"request_id": "inspection-0001"},
        visible_context={"final_decision": "Defective"},
        include_raw_evidence=False,
    )

    def fake_post(endpoint, json, timeout):
        captured["endpoint"] = endpoint
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            200,
            {
                "answer": "Mock grounded explanation.",
                "evidence_used": [{"source": "decision.final_decision", "value": "Defective"}],
                "limitations": ["Manual review still applies."],
                "provider_used": "mock",
                "fallback_used": True,
                "grounding_status": "grounded",
                "page_id": "image_inspection",
                "section_id": "final_decision",
            },
        )

    monkeypatch.setattr("frontend.streamlit_app.requests.post", fake_post)

    payload = _call_agent_explain_api("http://api:8000/", request_payload)

    assert captured["endpoint"] == "http://api:8000/agent/explain"
    assert captured["timeout"] == 60
    assert captured["json"]["page_id"] == "image_inspection"
    assert captured["json"]["section_id"] == "final_decision"
    assert captured["json"]["component_id"] == "image_inspection_ai_explanation_panel"
    assert captured["json"]["inspection_response"]["request_id"] == "inspection-0001"
    assert payload["provider_used"] == "mock"
    assert payload["fallback_used"] is True


def test_image_inspection_agent_request_includes_required_fields() -> None:
    request_payload = _build_image_inspection_agent_request(
        question="Explain this inspection result.",
        inspection_response={"request_id": "inspection-0001"},
        visible_context={"final_decision": "Defective"},
        include_raw_evidence=False,
    )

    assert request_payload["page_id"] == "image_inspection"
    assert request_payload["section_id"] == "final_decision"
    assert request_payload["component_id"] == "image_inspection_ai_explanation_panel"
    assert request_payload["question"] == "Explain this inspection result."
    assert request_payload["inspection_response"]["request_id"] == "inspection-0001"
    assert request_payload["visible_context"]["final_decision"] == "Defective"
    assert request_payload["include_raw_evidence"] is False


def test_image_inspection_api_call_rejects_api_errors(monkeypatch) -> None:
    def fake_post(endpoint, files, timeout):
        return _FakeResponse(503, {"detail": "Model dependency unavailable"})

    monkeypatch.setattr("frontend.streamlit_app.requests.post", fake_post)

    uploaded = SimpleNamespace(
        name="surface.png",
        type="image/png",
        getvalue=lambda: _png_bytes(),
    )

    with pytest.raises(RuntimeError, match="Model dependency unavailable"):
        _call_image_inspection_api("http://localhost:8000", uploaded)


def test_detection_overlay_helper_draws_boxes() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    payload = {
        "image_width": 100,
        "image_height": 100,
        "detections": [
            {
                "box_id": 0,
                "class_label": "inclusion",
                "display_label": "Inclusion",
                "confidence": 0.82,
                "bbox_xyxy": [10, 10, 40, 40],
                "is_best_prediction": True,
            }
        ],
    }

    annotated = _annotate_detection_boxes(image, payload)

    assert annotated.size == image.size
    assert annotated.getpixel((10, 10)) != (255, 255, 255)
    assert _extract_detection_rows(payload) == payload["detections"]


def test_image_inspection_compact_labels_are_friendly() -> None:
    assert _friendly_metric_display("classification_detection_agree_v0") == "Class + detection agree"
    assert _friendly_metric_display("autoencoder") == "Autoencoder"
    assert _friendly_metric_display("review_required_weak_evidence") == "Needs review"
    assert _friendly_metric_display("review_required") == "Needs review"


def test_agent_explanation_status_caption_reports_mock_fallback() -> None:
    mock_expected = "Safe mock fallback active · manual review still applies."
    gemini_expected = "Gemini-gated explanation returned a grounded response."
    assert _agent_explanation_status_caption("mock", False) == mock_expected
    assert _agent_explanation_status_caption("gemini", True) == mock_expected
    assert _agent_explanation_status_caption("gemini", False) == gemini_expected


def test_agent_explanation_diagnostic_items_omit_empty_values() -> None:
    items = _build_agent_explanation_diagnostic_items(
        {
            "fallback_reason": "",
            "provider_error_stage": "post_generation",
            "provider_error_reason": "safety_blocked",
            "safety_block_reason": None,
        }
    )

    assert items == [
        ("Provider error stage", "post_generation"),
        ("Provider error reason", "safety_blocked"),
    ]


def test_image_inspection_agent_request_sanitizes_readiness_boundary_text() -> None:
    raw_inspection_response = {
        "request_id": "inspection-0001",
        "decision": {"final_decision": "manual_review_required", "rule_id": "local_gated_runtime_validation_rule"},
        "classification": {
            "predicted_label": "defect",
            "limitations": [
                "Classification output is local model output and not production-ready.",
                "Classification output is not deployment-safe.",
            ],
            "traceability": {"source_endpoint": "local_gated_agent_endpoint_validation"},
        },
        "detection": {
            "predicted_box_count": 1,
            "limitations": [
                "Detection output is local model output and not production-ready.",
                "Detection output is not deployment-safe.",
            ],
            "traceability": {"source_endpoint": "local_gated_agent_endpoint_validation"},
        },
        "anomaly": {
            "predicted_label": "anomaly",
            "limitations": [
                "Anomaly output is local autoencoder reconstruction evidence and not production-ready.",
                "Anomaly output is not deployment-safe.",
            ],
            "traceability": {"source_endpoint": "local_gated_agent_endpoint_validation"},
        },
        "traceability": {
            "source_endpoint": "local_gated_agent_endpoint_validation",
            "classification": {
                "limitations": [
                    "Classification output is local model output and not production-ready.",
                    "Classification output is not deployment-safe.",
                ]
            },
            "detection": {
                "limitations": [
                    "Detection output is local model output and not production-ready.",
                    "Detection output is not deployment-safe.",
                ]
            },
            "anomaly": {
                "limitations": [
                    "Anomaly output is local autoencoder reconstruction evidence and not production-ready.",
                    "Anomaly output is not deployment-safe.",
                ]
            },
        },
        "limitations": [
            "This response does not claim production readiness.",
            "This response does not claim deployment safety.",
        ],
        "warnings": ["No inspection warnings were returned."],
        "errors": [],
        "explanation_context": {
            "safety_boundaries": ["No production-ready claim.", "No deployment-safe claim."],
            "forbidden_claims": ["production-ready", "deployment-safe"],
        },
    }

    sanitized = _build_image_inspection_agent_inspection_response(raw_inspection_response)
    request = _build_image_inspection_agent_request(
        question="Explain the current image inspection result for manual review.",
        inspection_response=raw_inspection_response,
        visible_context={"final_decision": "manual_review_required"},
        include_raw_evidence=False,
    )

    assert request["inspection_response"] == sanitized
    assert "limitations" not in str(request["inspection_response"]).lower()
    assert "limitations" not in sanitized
    assert "warnings" not in sanitized
    assert "errors" not in sanitized
    assert "explanation_context" not in sanitized
    assert "production readiness" not in str(request).lower()
    assert "deployment safety" not in str(request).lower()
    assert "manual_review_required" in str(request)


def test_image_inspection_agent_request_sanitizes_nested_readiness_fields_for_guard() -> None:
    raw_inspection_response = {
        "request_id": "inspection-0001",
        "decision": {"final_decision": "manual_review_required", "rule_id": "local_gated_runtime_validation_rule"},
        "classification": {
            "predicted_label": "defect",
            "limitations": [
                "Classification output is local model output and not production-ready.",
                "Classification output is not deployment-safe.",
            ],
        },
        "detection": {
            "predicted_box_count": 1,
            "limitations": [
                "Detection output is local model output and not production-ready.",
                "Detection output is not deployment-safe.",
            ],
        },
        "anomaly": {
            "predicted_label": "anomaly",
            "limitations": [
                "Anomaly output is local autoencoder reconstruction evidence and not production-ready.",
                "Anomaly output is not deployment-safe.",
            ],
        },
        "traceability": {
            "source_endpoint": "local_gated_agent_endpoint_validation",
            "classification": {
                "limitations": [
                    "Classification output is local model output and not production-ready.",
                    "Classification output is not deployment-safe.",
                ]
            },
            "detection": {
                "limitations": [
                    "Detection output is local model output and not production-ready.",
                    "Detection output is not deployment-safe.",
                ]
            },
            "anomaly": {
                "limitations": [
                    "Anomaly output is local autoencoder reconstruction evidence and not production-ready.",
                    "Anomaly output is not deployment-safe.",
                ]
            },
        },
        "limitations": [
            "This response does not claim production readiness.",
            "This response does not claim deployment safety.",
        ],
        "warnings": ["No inspection warnings were returned."],
        "errors": [],
        "explanation_context": {
            "safety_boundaries": ["No production-ready claim.", "No deployment-safe claim."],
            "forbidden_claims": ["production-ready", "deployment-safe"],
        },
    }

    request = _build_image_inspection_agent_request(
        question="Explain this image inspection result safely for manual review.",
        inspection_response=raw_inspection_response,
        visible_context={"final_decision": "manual_review_required", "decision_level": "review"},
        include_raw_evidence=False,
    )
    grounding_context = build_grounding_context(
        page_id=request["page_id"],
        section_id=request["section_id"],
        component_id=request["component_id"],
        question=request["question"],
        visible_context=request["visible_context"],
        inspection_response=request["inspection_response"],
        include_raw_evidence=False,
    )
    guard_result = guard_pre_generation_context(grounding_context)

    assert guard_result.blocked is False
    assert guard_result.reasons == ()
    assert "limitations" not in str(request["inspection_response"]).lower()
    assert "production-ready" not in str(request["inspection_response"]).lower()
    assert "deployment-safe" not in str(request["inspection_response"]).lower()


def test_image_inspection_question_field_is_the_readiness_trigger_when_it_contains_forbidden_phrase() -> None:
    request = _build_image_inspection_agent_request(
        question="Is this production-ready?",
        inspection_response={
            "request_id": "inspection-0001",
            "decision": {"final_decision": "manual_review_required", "rule_id": "local_gated_runtime_validation_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "local_gated_agent_endpoint_validation"},
        },
        visible_context={"final_decision": "manual_review_required"},
        include_raw_evidence=False,
    )
    grounding_context = build_grounding_context(
        page_id=request["page_id"],
        section_id=request["section_id"],
        component_id=request["component_id"],
        question=request["question"],
        visible_context=request["visible_context"],
        inspection_response=request["inspection_response"],
        include_raw_evidence=False,
    )
    guard_result = guard_pre_generation_context(grounding_context)

    assert guard_result.blocked is True
    assert guard_result.reasons == (
        "The request asks for production/deployment/autonomous certification, which is not allowed.",
    )
    assert grounding_context.question == "Is this production-ready?"
    assert "production-ready" in grounding_context.question


def test_image_inspection_agent_panel_renders_safe_fallback_diagnostics(monkeypatch) -> None:
    fake_st = _FakeStreamlit(submitted=False)
    monkeypatch.setattr("frontend.streamlit_app.st", fake_st)

    fake_st.session_state["image_inspection_agent_source_request_id"] = "inspection-0001"
    fake_st.session_state["image_inspection_agent_explanation"] = {
        "answer": "Use the evidence and keep manual review in place.",
        "evidence_used": [{"source": "decision.final_decision", "value": "manual_review_required"}],
        "limitations": ["Manual review still applies."],
        "provider_used": "mock",
        "fallback_used": True,
        "grounding_status": "grounded",
        "fallback_reason": "Gemini real provider output was blocked by the safety guard; mock fallback remains the safe path.",
        "provider_error_stage": "post_generation",
        "provider_error_reason": "safety_blocked",
        "safety_block_reason": "invented_metric_like_output",
    }

    app._render_image_inspection_agent_panel(
        api_base_url="http://localhost:8000",
        payload={"request_id": "inspection-0001"},
        decision={},
        classification={},
        detection={},
        anomaly={},
        warnings=[],
        limitations=[],
    )

    assert ("caption", "Safe mock fallback active · manual review still applies.") in fake_st.calls
    assert ("metric", ("Provider", "mock")) in fake_st.calls
    assert ("metric", ("Fallback", "Yes")) in fake_st.calls
    assert ("expander", "Response diagnostics:False") in fake_st.calls
    assert ("metric", ("Fallback reason", "Gemini real provider output was blocked by the safety guard; mock fallback remains the safe path.")) in fake_st.calls
    assert ("metric", ("Provider error stage", "post_generation")) in fake_st.calls
    assert ("metric", ("Provider error reason", "safety_blocked")) in fake_st.calls
    assert ("metric", ("Safety block reason", "invented_metric_like_output")) in fake_st.calls
    assert not any("fake api key" in str(call).lower() for call in fake_st.calls)
    assert not any("raw provider" in str(call).lower() for call in fake_st.calls)


def test_image_inspection_agent_panel_sends_sanitized_payload_to_agent_explain(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(endpoint, json, timeout):
        captured["endpoint"] = endpoint
        captured["json"] = json
        return _FakeResponse(
            200,
            {
                "answer": "Gemini gated endpoint answer. Manual review still applies.",
                "evidence_used": [{"source": "inspection_response.decision.final_decision", "value": "manual_review_required"}],
                "limitations": ["Manual review still applies."],
                "provider_used": "gemini",
                "fallback_used": False,
                "grounding_status": "grounded",
                "page_id": "image_inspection",
                "section_id": "final_decision",
            },
        )

    monkeypatch.setattr("frontend.streamlit_app.requests.post", fake_post)

    payload = {
        "request_id": "inspection-0001",
        "classification": {
            "predicted_label": "defect",
            "limitations": [
                "Classification output is local model output and not production-ready.",
                "Classification output is not deployment-safe.",
            ],
        },
        "detection": {
            "predicted_box_count": 1,
            "limitations": [
                "Detection output is local model output and not production-ready.",
                "Detection output is not deployment-safe.",
            ],
        },
        "anomaly": {
            "predicted_label": "anomaly",
            "limitations": [
                "Anomaly output is local autoencoder reconstruction evidence and not production-ready.",
                "Anomaly output is not deployment-safe.",
            ],
        },
        "decision": {"final_decision": "manual_review_required", "rule_id": "local_gated_runtime_validation_rule"},
        "traceability": {
            "source_endpoint": "local_gated_agent_endpoint_validation",
            "classification": {
                "limitations": [
                    "Classification output is local model output and not production-ready.",
                    "Classification output is not deployment-safe.",
                ]
            },
            "detection": {
                "limitations": [
                    "Detection output is local model output and not production-ready.",
                    "Detection output is not deployment-safe.",
                ]
            },
            "anomaly": {
                "limitations": [
                    "Anomaly output is local autoencoder reconstruction evidence and not production-ready.",
                    "Anomaly output is not deployment-safe.",
                ]
            },
        },
        "limitations": [
            "This response does not claim production readiness.",
            "This response does not claim deployment safety.",
        ],
        "warnings": ["No inspection warnings were returned."],
        "errors": [],
        "explanation_context": {
            "safety_boundaries": ["No production-ready claim.", "No deployment-safe claim."],
            "forbidden_claims": ["production-ready", "deployment-safe"],
        },
    }
    visible_context = {"final_decision": "manual_review_required", "decision_level": "review"}

    sanitized_request = _build_image_inspection_agent_request(
        question="Explain the current image inspection result for manual review.",
        inspection_response=payload,
        visible_context=visible_context,
        include_raw_evidence=False,
    )

    response = _call_agent_explain_api("http://api:8000/", sanitized_request)

    assert captured["endpoint"] == "http://api:8000/agent/explain"
    sent_payload = captured["json"]
    assert "production-ready" not in str(sent_payload).lower()
    assert "deployment-safe" not in str(sent_payload).lower()
    assert "limitations" not in sent_payload["inspection_response"]
    assert "limitations" not in str(sent_payload["inspection_response"]).lower()
    assert "warnings" not in sent_payload["inspection_response"]
    assert "errors" not in sent_payload["inspection_response"]
    assert "explanation_context" not in sent_payload["inspection_response"]
    assert response["provider_used"] == "gemini"
    assert response["fallback_used"] is False


def test_image_inspection_agent_panel_resets_stale_question_on_new_request_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(endpoint, json, timeout):
        captured["endpoint"] = endpoint
        captured["json"] = json
        return _FakeResponse(
            200,
            {
                "answer": "Gemini gated endpoint answer. Manual review still applies.",
                "evidence_used": [{"source": "inspection_response.decision.final_decision", "value": "manual_review_required"}],
                "limitations": ["Manual review still applies."],
                "provider_used": "gemini",
                "fallback_used": False,
                "grounding_status": "grounded",
                "page_id": "image_inspection",
                "section_id": "final_decision",
            },
        )

    monkeypatch.setattr("frontend.streamlit_app.requests.post", fake_post)

    fake_st = _FakeStreamlit(submitted=True)
    monkeypatch.setattr("frontend.streamlit_app.st", fake_st)
    fake_st.session_state["image_inspection_agent_source_request_id"] = "old-request"
    fake_st.session_state["image_inspection_agent_question"] = "Is this production-ready?"
    fake_st.session_state["image_inspection_agent_explanation"] = None
    fake_st.session_state["image_inspection_agent_error"] = None

    app._render_image_inspection_agent_panel(
        api_base_url="http://localhost:8000",
        payload={"request_id": "new-request"},
        decision={},
        classification={},
        detection={},
        anomaly={},
        warnings=[],
        limitations=[],
    )

    sent_payload = captured["json"]
    assert sent_payload["question"] == "Explain this image inspection result safely for manual review."
    assert "production-ready" not in sent_payload["question"].lower()
    assert "deployment-safe" not in str(sent_payload).lower()
    assert "limitations" not in sent_payload["inspection_response"]
    assert "limitations" not in str(sent_payload["inspection_response"]).lower()
    assert "warnings" not in sent_payload["inspection_response"]
    assert "errors" not in sent_payload["inspection_response"]
    assert "explanation_context" not in sent_payload["inspection_response"]


def test_frontend_source_no_longer_uses_classification_only_flow() -> None:
    source = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    assert "/inspect/image" in source
    assert "/agent/explain" in source
    assert "classification only" not in source
    assert "unified inspection UI is not yet connected" not in source
    assert "Explain this inspection result" in source
    assert "Response diagnostics" in source
    assert "Safe mock fallback active · manual review still applies." in source
    assert "Gemini-gated explanation returned a grounded response." in source
    assert "Evidence-grounded explanation for the current inspection result." in source
    assert "Uses the /agent/explain response path" in source
    assert "Gemini-gated responses are available only when explicitly enabled." in source
    assert "Safe mock fallback remains available and manual review still applies." in source
    assert "One-shot, evidence-grounded explanation for the current inspection result." in source
    assert "Safe explanation path · fallback available" in source
    assert "MOCK-FIRST · GATED GEMINI AVAILABLE" in source
    assert "Detection box details" in source
    assert "Classification details" in source
    assert "Detection details" in source
    assert "Anomaly details" in source
    assert "Inspection warnings" in source
    assert "st.warning(\"Inspection warnings were returned by the unified inspection response.\")" not in source
    assert "Raw API response" in source
    assert '("Decision", _friendly_status_label(classification.get("decision")))' in source
    assert '("Defect prob.", _format_probability(classification.get("probability_defect")))' in source
    assert '("Review", _friendly_metric_display(detection.get("review_status")))' in source
    assert '("Quality", _friendly_metric_display(anomaly.get("quality_status")))' in source
    assert source.index('with st.expander("Classification details", expanded=False):') > source.index("with model_cols[2]:")
    assert source.index('with st.expander("Detection box details", expanded=False):') > source.index("with model_cols[2]:")
    assert source.index('with st.expander("Anomaly details", expanded=False):') > source.index("with model_cols[2]:")


def _png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
