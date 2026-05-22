"""Unit tests for the deterministic smart decision layer."""

from __future__ import annotations

from src.inspection_ai.decision.inspection_decision import build_inspection_decision
from src.inspection_ai.contracts.inspection import (
    AnomalyResult,
    ClassificationResult,
    DetectionBox,
    DetectionResult,
)


RULES_CONFIG_PATH = "configs/decision/thresholds.yaml"
SOURCE_ENDPOINT = "/inspect/image"


def test_all_unavailable_missing_returns_inconclusive() -> None:
    result = build_inspection_decision()

    assert result.status == "unavailable"
    assert result.final_decision == "inconclusive"
    assert result.decision_level == "inconclusive"
    assert result.model_agreement_status == "no_signal"
    assert result.rule_id == "all_signals_unavailable"
    assert "classification=unavailable" in result.supporting_signals
    assert "detection=unavailable" in result.supporting_signals
    assert "anomaly=unavailable" in result.supporting_signals


def test_classification_good_detection_no_boxes_anomaly_normal_returns_good() -> None:
    result = build_inspection_decision(
        classification=_classification_result(predicted_label="good"),
        detection=_detection_result(predicted_box_count=0, defect_count=0, review_status="no_detection"),
        anomaly=_anomaly_result(predicted_label="normal", quality_status="review_required"),
    )

    assert result.status == "success"
    assert result.final_decision == "good"
    assert result.decision_level == "auto_review_safe"
    assert result.model_agreement_status == "all_available_signals_agree"
    assert result.primary_signal == "classification"
    assert result.rule_id == "good_all_signals_agree_v0"
    assert result.recommended_action == "Accept as likely good, with safety limitations."


def test_classification_defect_detection_boxes_returns_defective() -> None:
    result = build_inspection_decision(
        classification=_classification_result(predicted_label="defect"),
        detection=_detection_result(predicted_box_count=2, defect_count=2, review_status="review_required"),
        anomaly=_anomaly_result(predicted_label="normal", quality_status="review_required"),
    )

    assert result.status == "success"
    assert result.final_decision == "defective"
    assert result.decision_level == "evidence_supported"
    assert result.model_agreement_status == "classification_detection_agree"
    assert result.primary_signal == "detection"
    assert result.rule_id == "classification_detection_agree_v0"


def test_detection_boxes_and_classification_good_returns_manual_review() -> None:
    result = build_inspection_decision(
        classification=_classification_result(predicted_label="good"),
        detection=_detection_result(predicted_box_count=1, defect_count=1, review_status="review_required"),
        anomaly=_anomaly_result(predicted_label="normal", quality_status="review_required"),
    )

    assert result.final_decision == "needs_manual_review"
    assert result.decision_level == "review"
    assert result.model_agreement_status == "conflict"
    assert result.primary_signal == "detection"
    assert "Classification predicted good" in (result.conflict_reason or "")


def test_classification_defect_without_boxes_returns_manual_review() -> None:
    result = build_inspection_decision(
        classification=_classification_result(predicted_label="defect"),
        detection=_detection_result(predicted_box_count=0, defect_count=0, review_status="no_detection"),
        anomaly=_anomaly_result(predicted_label="normal", quality_status="review_required"),
    )

    assert result.final_decision == "needs_manual_review"
    assert result.decision_level == "review"
    assert result.model_agreement_status == "partial_or_conflict"
    assert result.primary_signal == "classification"
    assert "no localized boxes" in (result.conflict_reason or "")


def test_anomaly_only_weak_evidence_returns_manual_review() -> None:
    result = build_inspection_decision(
        anomaly=_anomaly_result(
            predicted_label="anomaly",
            quality_status="review_required_weak_evidence",
            anomaly_score=0.219,
            reconstruction_loss=0.219,
        )
    )

    assert result.final_decision == "needs_manual_review"
    assert result.decision_level == "review"
    assert result.model_agreement_status == "supporting_only"
    assert result.primary_signal == "anomaly"
    assert "weak/review-only" in (result.recommended_action or "")


def test_good_classification_no_boxes_weak_anomaly_returns_manual_review() -> None:
    result = build_inspection_decision(
        classification=_classification_result(predicted_label="good"),
        detection=_detection_result(predicted_box_count=0, defect_count=0, review_status="no_detection"),
        anomaly=_anomaly_result(predicted_label="anomaly", quality_status="review_required_weak_evidence"),
    )

    assert result.final_decision == "needs_manual_review"
    assert result.model_agreement_status == "conflict"
    assert result.primary_signal == "anomaly"
    assert "weak anomaly evidence" in (result.conflict_reason or "").lower()


def test_detection_only_boxes_returns_defective() -> None:
    result = build_inspection_decision(
        detection=_detection_result(predicted_box_count=1, defect_count=1, review_status="review_required"),
    )

    assert result.final_decision == "defective"
    assert result.decision_level == "evidence_supported"
    assert result.model_agreement_status == "partial_signal"
    assert result.primary_signal == "detection"


def test_classification_only_good_returns_manual_review() -> None:
    result = build_inspection_decision(
        classification=_classification_result(predicted_label="good"),
    )

    assert result.final_decision == "needs_manual_review"
    assert result.decision_level == "review"
    assert result.model_agreement_status == "partial_signal"
    assert result.primary_signal == "classification"


def test_all_failed_statuses_returns_inconclusive() -> None:
    result = build_inspection_decision(
        classification={"status": "failed"},
        detection={"status": "failed"},
        anomaly={"status": "failed"},
    )

    assert result.status == "unavailable"
    assert result.final_decision == "inconclusive"
    assert result.decision_level == "inconclusive"
    assert result.model_agreement_status == "no_signal"


def test_traceability_contains_rules_config_path_and_source_endpoint() -> None:
    result = build_inspection_decision(
        classification=_classification_result(predicted_label="good"),
        detection=_detection_result(predicted_box_count=0, defect_count=0, review_status="no_detection"),
        anomaly=_anomaly_result(predicted_label="normal", quality_status="review_required"),
        rules_config_path=RULES_CONFIG_PATH,
        source_endpoint=SOURCE_ENDPOINT,
    )

    assert result.traceability.rules_config_path == RULES_CONFIG_PATH
    assert result.traceability.source_endpoint == SOURCE_ENDPOINT
    assert result.traceability.rule_source.endswith("build_inspection_decision")


def test_limitations_include_conservative_safety_language() -> None:
    result = build_inspection_decision(
        classification=_classification_result(predicted_label="good"),
        detection=_detection_result(predicted_box_count=0, defect_count=0, review_status="no_detection"),
        anomaly=_anomaly_result(predicted_label="normal", quality_status="review_required_weak_evidence"),
    )

    joined = " | ".join(result.limitations)
    assert "deterministic rule-based aggregation" in joined
    assert "does not claim production readiness" in joined
    assert "does not claim deployment safety" in joined
    assert "weak" in joined


def _classification_result(*, predicted_label: str) -> ClassificationResult:
    return ClassificationResult(
        status="success",
        model_name="resnet18",
        model_version="0.4.0",
        run_id="1bc92561-c5bf-48f2-8246-b8f3d5718ffe",
        threshold=0.65,
        predicted_label=predicted_label,
        predicted_label_id=0 if predicted_label == "good" else 1,
        probability_good=0.82 if predicted_label == "good" else 0.18,
        probability_defect=0.18 if predicted_label == "good" else 0.82,
        decision=predicted_label,
        production_ready=False,
        deployment_safe=False,
    )


def _detection_result(
    *,
    predicted_box_count: int,
    defect_count: int,
    review_status: str,
) -> DetectionResult:
    boxes = []
    best_detection = None
    if predicted_box_count > 0:
        box = DetectionBox(
            box_id=0,
            class_id=2,
            class_label="inclusion",
            display_label="Inclusion",
            confidence=0.8,
            bbox_format="xyxy",
            bbox_xyxy=[10.0, 12.0, 80.0, 90.0],
            score_rank=1,
            is_best_prediction=True,
        )
        boxes = [box]
        best_detection = box
    return DetectionResult(
        status="success",
        model_name="yolo",
        model_version="0.2.0",
        run_id="yolo_train_v0_2_0",
        confidence_threshold=0.25,
        iou_threshold=0.7,
        image_width=2048,
        image_height=1000,
        predicted_box_count=predicted_box_count,
        defect_count=defect_count,
        detections=boxes,
        best_detection=best_detection,
        review_status=review_status,
        production_ready=False,
        deployment_safe=False,
    )


def _anomaly_result(
    *,
    predicted_label: str,
    quality_status: str,
    anomaly_score: float = 0.06873100237623937,
    reconstruction_loss: float = 0.06873100237623937,
) -> AnomalyResult:
    return AnomalyResult(
        status="success",
        model_name="autoencoder",
        model_version="0.1.0",
        run_id="b8ca43f5-0d53-4a42-ab37-b5fca9544a36",
        anomaly_score=anomaly_score,
        reconstruction_loss=reconstruction_loss,
        threshold=0.2043069839477539,
        predicted_label=predicted_label,
        decision=predicted_label,
        quality_status=quality_status,
        production_ready=False,
        deployment_safe=False,
        optional_reconstruction_artifacts=None,
    )
