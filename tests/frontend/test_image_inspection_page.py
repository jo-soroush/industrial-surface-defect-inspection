from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from frontend.streamlit_app import (
    _annotate_detection_boxes,
    _call_image_inspection_api,
    _extract_detection_rows,
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


def test_frontend_source_no_longer_uses_classification_only_flow() -> None:
    source = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    assert "/inspect/image" in source
    assert "classification only" not in source
    assert "unified inspection UI is not yet connected" not in source


def _png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
