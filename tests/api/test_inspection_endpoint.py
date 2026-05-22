"""API tests for the unified /inspect/image endpoint."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.app.main import app
from api.app.schemas.inspection import (
    AnomalyResult,
    ClassificationResult,
    DecisionResult,
    DetectionBox,
    DetectionResult,
    ExplanationContext,
    ImageInspectionResponse,
    InputMetadata,
    TopLevelTraceability,
)
import api.app.routes.inspection as inspection_routes


client = TestClient(app)


def test_inspect_image_returns_full_response(monkeypatch) -> None:
    monkeypatch.setattr(inspection_routes, "inspect_image", lambda **_: _fake_response())

    response = client.post(
        "/inspect/image",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"]["final_decision"] == "needs_manual_review"
    assert payload["classification"]["status"] == "success"
    assert payload["detection"]["status"] == "success"
    assert payload["anomaly"]["status"] == "success"
    assert payload["errors"] == []
    assert payload["warnings"] == []


def test_inspect_image_rejects_missing_upload() -> None:
    response = client.post("/inspect/image")

    assert response.status_code == 400
    assert "Missing image upload" in response.json()["detail"]


def test_inspect_image_rejects_empty_upload() -> None:
    response = client.post(
        "/inspect/image",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert "empty or missing" in response.json()["detail"]


def test_inspect_image_rejects_non_image_upload() -> None:
    response = client.post(
        "/inspect/image",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]


def test_inspect_image_rejects_corrupt_image_payload(monkeypatch) -> None:
    def fail_service(**kwargs):
        raise ValueError("Uploaded file could not be decoded as an image.")

    monkeypatch.setattr(inspection_routes, "inspect_image", fail_service)

    response = client.post(
        "/inspect/image",
        files={"file": ("surface.png", b"not a real png", "image/png")},
    )

    assert response.status_code == 400
    assert "could not be decoded as an image" in response.json()["detail"]


def test_inspect_image_returns_500_on_unexpected_orchestration_error(monkeypatch) -> None:
    def fail_service(**kwargs):
        raise RuntimeError("orchestration bug")

    monkeypatch.setattr(inspection_routes, "inspect_image", fail_service)

    response = client.post(
        "/inspect/image",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unified inspection failed unexpectedly."


def _fake_response() -> ImageInspectionResponse:
    detection_box = DetectionBox(
        box_id=0,
        class_id=2,
        class_label="inclusion",
        display_label="Inclusion",
        confidence=0.82,
        bbox_format="xyxy",
        bbox_xyxy=[10.0, 12.0, 80.0, 90.0],
        score_rank=1,
        is_best_prediction=True,
    )
    return ImageInspectionResponse(
        request_id="request-0001",
        timestamp_utc="2026-05-21T12:00:00Z",
        input=InputMetadata(
            filename="surface.png",
            content_type="image/png",
            file_size_bytes=245760,
            image_width=2048,
            image_height=1000,
            image_mode="RGB",
        ),
        classification=ClassificationResult(
            status="success",
            model_name="resnet18",
            model_version="0.4.0",
            run_id="1bc92561-c5bf-48f2-8246-b8f3d5718ffe",
            threshold=0.65,
            predicted_label="defect",
            predicted_label_id=1,
            probability_good=0.214,
            probability_defect=0.786,
            decision="defect",
        ),
        detection=DetectionResult(
            status="success",
            model_name="yolo",
            model_version="0.2.0",
            run_id="yolo_train_v0_2_0",
            confidence_threshold=0.25,
            iou_threshold=0.7,
            image_width=2048,
            image_height=1000,
            predicted_box_count=1,
            defect_count=1,
            detections=[detection_box],
            best_detection=detection_box,
            review_status="review_required",
        ),
        anomaly=AnomalyResult(
            status="success",
            model_name="autoencoder",
            model_version="0.1.0",
            run_id="b8ca43f5-0d53-4a42-ab37-b5fca9544a36",
            anomaly_score=0.219,
            reconstruction_loss=0.219,
            threshold=0.2043069839477539,
            predicted_label="anomaly",
            decision="anomaly",
            quality_status="review_required_weak_evidence",
        ),
        decision=DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="mixed_signals",
            primary_signal="detection",
            supporting_signals=[
                "classification predicted defect",
                "detection returned one box",
                "anomaly predicted anomaly",
            ],
            recommended_action="Send image for manual review.",
            rule_id="example_review_when_any_model_flags_defect_v0",
            rule_summary="Require manual review when model outputs flag possible defect.",
        ),
        traceability=TopLevelTraceability(
            contract_version="image_inspection_response_v0_1",
            api_version="inspect_image_api_v0_1",
            source_endpoint="/inspect/image",
            classification={"run_id": "1bc92561-c5bf-48f2-8246-b8f3d5718ffe"},
            detection={"run_id": "yolo_train_v0_2_0"},
            anomaly={"run_id": "b8ca43f5-0d53-4a42-ab37-b5fca9544a36"},
            decision={"rule_id": "example_review_when_any_model_flags_defect_v0"},
            frontend_evidence_sources=[
                "artifacts/models/metrics/anomaly_pr_curve__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
            ],
        ),
        limitations=[
            "This response is a governed inspection aggregation of model outputs.",
            "This response does not claim production readiness.",
            "This response does not claim deployment safety.",
        ],
        warnings=[],
        errors=[],
        explanation_context=ExplanationContext(
            status="available",
            context_version="image_inspection_explanation_context_v0_1",
            allowed_sources=[
                "classification",
                "detection",
                "anomaly",
                "decision",
                "traceability",
                "limitations",
            ],
            summary_inputs={"final_decision": "needs_manual_review"},
            safety_boundaries=[
                "No production-ready claim.",
                "No deployment-safe claim.",
            ],
            forbidden_claims=[
                "production-ready",
                "deployment-safe",
            ],
        ),
    )


def _png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(120, 130, 140))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
