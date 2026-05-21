"""Smoke tests for unified image inspection schemas."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def test_image_inspection_response_schema_accepts_minimal_success_payload() -> None:
    """Validate the future unified inspection response shape."""
    detection_box = DetectionBox(
        box_id=0,
        class_id=2,
        class_label="inclusion",
        display_label="Inclusion",
        confidence=0.802,
        bbox_format="xyxy",
        bbox_xyxy=[1840.27, 0.0, 1987.1, 1000.0],
        score_rank=1,
        is_best_prediction=True,
    )

    response = ImageInspectionResponse(
        request_id="test-request-0001",
        timestamp_utc="2026-05-21T12:00:00Z",
        input=InputMetadata(
            filename="example_surface.png",
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
            quality_status="weak_governed_evidence_requires_review",
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
            api_version="UNKNOWN",
            source_endpoint="/inspect/image",
            classification={"run_id": "1bc92561-c5bf-48f2-8246-b8f3d5718ffe"},
            detection={"run_id": "yolo_train_v0_2_0"},
            anomaly={"run_id": "b8ca43f5-0d53-4a42-ab37-b5fca9544a36"},
            decision={"rule_id": "example_review_when_any_model_flags_defect_v0"},
        ),
        limitations=[
            "This response is not production-ready.",
            "This response is not deployment-safe.",
        ],
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

    dumped = response.model_dump()

    for key in (
        "request_id",
        "timestamp_utc",
        "input",
        "classification",
        "detection",
        "anomaly",
        "decision",
        "traceability",
        "limitations",
        "errors",
        "warnings",
        "explanation_context",
    ):
        assert key in dumped

    assert response.decision.final_decision == "needs_manual_review"
    assert response.detection.detections[0].bbox_format == "xyxy"
    assert response.classification.production_ready is False
    assert response.classification.deployment_safe is False
    assert response.detection.production_ready is False
    assert response.detection.deployment_safe is False
    assert response.anomaly.production_ready is False
    assert response.anomaly.deployment_safe is False
