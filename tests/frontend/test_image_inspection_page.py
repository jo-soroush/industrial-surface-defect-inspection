from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from frontend.streamlit_app import (
    _agent_explanation_status_caption,
    _annotate_detection_boxes,
    _build_image_inspection_agent_request,
    _call_image_inspection_api,
    _call_agent_explain_api,
    _extract_detection_rows,
    _friendly_metric_display,
    _get_api_base_url,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, object]:
        return self._payload


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
    expected = "Mock explanation MVP active · external LLM not connected · no fake AI"
    assert _agent_explanation_status_caption("mock", False) == expected
    assert _agent_explanation_status_caption("gemini", True) == expected


def test_frontend_source_no_longer_uses_classification_only_flow() -> None:
    source = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    assert "/inspect/image" in source
    assert "/agent/explain" in source
    assert "classification only" not in source
    assert "unified inspection UI is not yet connected" not in source
    assert "Explain this inspection result" in source
    assert "Mock explanation MVP active for the current inspection result. The panel uses governed response evidence, model outputs, warnings, limitations, and traceability." in source
    assert "One-shot, evidence-grounded explanation for the current inspection result." in source
    assert "Mock explanation MVP active · external LLM not connected · no fake AI" in source
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
