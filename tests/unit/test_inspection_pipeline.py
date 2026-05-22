"""Unit tests for the unified image inspection orchestration service."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from api.app.schemas.inspection import (
    AnomalyResult,
    ClassificationResult,
    DetectionBox,
    DetectionResult,
)
from src.inspection_ai.inference.track_a_classifier import TrackAPredictionResult
from src.inspection_ai.inspection_pipeline import inspect_image


def test_inspect_image_all_success_returns_good_decision() -> None:
    response = inspect_image(
        image_bytes=_png_bytes(),
        filename="surface.png",
        content_type="image/png",
        request_id="request-0001",
        classifier=FakeClassifier(predicted_label="good"),
        detector=FakeDetector(box_count=0),
        anomaly_detector=FakeAnomalyDetector(predicted_label="normal"),
    )

    assert response.request_id == "request-0001"
    assert response.classification.status == "success"
    assert response.detection.status == "success"
    assert response.anomaly.status == "success"
    assert response.decision.final_decision == "good"
    assert response.decision.status == "success"
    assert response.errors == []
    assert response.warnings == []
    assert response.classification.production_ready is False
    assert response.detection.production_ready is False
    assert response.anomaly.production_ready is False
    assert response.traceability.frontend_evidence_sources
    assert "anomaly_pr_curve" in response.traceability.frontend_evidence_sources[2]


def test_inspect_image_classification_failure_with_detection_boxes_stays_defective() -> None:
    response = inspect_image(
        image_bytes=_png_bytes(),
        filename="surface.png",
        content_type="image/png",
        classifier=FailingClassifier("Track A classifier failed."),
        detector=FakeDetector(box_count=1),
        anomaly_detector=FakeAnomalyDetector(predicted_label="normal"),
    )

    assert response.classification.status == "failed"
    assert response.detection.status == "success"
    assert response.decision.final_decision == "defective"
    assert response.decision.decision_level == "evidence_supported"
    assert any(error.component == "classification" for error in response.errors)
    assert response.warnings
    assert response.classification.predicted_label is None


def test_inspect_image_detection_failure_with_good_classification_returns_manual_review() -> None:
    response = inspect_image(
        image_bytes=_png_bytes(),
        filename="surface.png",
        content_type="image/png",
        classifier=FakeClassifier(predicted_label="good"),
        detector=FailingDetector("YOLO detection failed."),
        anomaly_detector=FakeAnomalyDetector(predicted_label="normal"),
    )

    assert response.detection.status == "failed"
    assert response.classification.status == "success"
    assert response.decision.final_decision == "needs_manual_review"
    assert response.decision.decision_level == "review"
    assert any(error.component == "detection" for error in response.errors)
    assert response.warnings


def test_inspect_image_anomaly_failure_with_good_classification_and_no_boxes_remains_good() -> None:
    response = inspect_image(
        image_bytes=_png_bytes(),
        filename="surface.png",
        content_type="image/png",
        classifier=FakeClassifier(predicted_label="good"),
        detector=FakeDetector(box_count=0),
        anomaly_detector=FailingDetector("Anomaly detection failed."),
    )

    assert response.anomaly.status == "failed"
    assert response.decision.final_decision == "good"
    assert response.decision.model_agreement_status == "all_available_signals_agree"
    assert any(error.component == "anomaly" for error in response.errors)
    assert response.warnings


def test_inspect_image_all_failures_returns_inconclusive_with_three_errors() -> None:
    response = inspect_image(
        image_bytes=_png_bytes(),
        filename="surface.png",
        content_type="image/png",
        classifier=FailingClassifier("Track A classifier failed."),
        detector=FailingDetector("YOLO detection failed."),
        anomaly_detector=FailingDetector("Anomaly detection failed."),
    )

    assert response.classification.status == "failed"
    assert response.detection.status == "failed"
    assert response.anomaly.status == "failed"
    assert response.decision.final_decision == "inconclusive"
    assert response.decision.status == "unavailable"
    assert len(response.errors) == 3
    assert {error.component for error in response.errors} == {"classification", "detection", "anomaly"}
    assert response.warnings


def test_inspect_image_weak_anomaly_evidence_is_flagged_for_review() -> None:
    response = inspect_image(
        image_bytes=_png_bytes(),
        filename="surface.png",
        content_type="image/png",
        classifier=FakeClassifier(predicted_label="good"),
        detector=FakeDetector(box_count=0),
        anomaly_detector=FakeAnomalyDetector(
            predicted_label="anomaly",
            quality_status="review_required_weak_evidence",
        ),
    )

    assert response.decision.final_decision == "needs_manual_review"
    assert any("weak" in warning.lower() for warning in response.warnings)
    assert any("weak" in limitation.lower() for limitation in response.limitations)
    assert response.explanation_context.status == "available"
    assert response.explanation_context.summary_inputs["final_decision"] == "needs_manual_review"


def test_inspect_image_rejects_empty_payload() -> None:
    try:
        inspect_image(image_bytes=b"", filename="surface.png", content_type="image/png")
    except ValueError as exc:
        assert "empty or missing" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError for empty payload.")


class FakeClassifier:
    def __init__(self, *, predicted_label: str) -> None:
        self.predicted_label = predicted_label
        self.model_name = "resnet18"
        self.model_version = "0.4.0"
        self.run_id = "1bc92561-c5bf-48f2-8246-b8f3d5718ffe"
        self.threshold = 0.65

    def predict(self, image_path):
        predicted_label_id = 0 if self.predicted_label == "good" else 1
        return TrackAPredictionResult(
            model_name=self.model_name,
            model_version=self.model_version,
            run_id=self.run_id,
            threshold=self.threshold,
            predicted_label=self.predicted_label,
            predicted_label_id=predicted_label_id,
            probability_good=0.82 if self.predicted_label == "good" else 0.18,
            probability_defect=0.18 if self.predicted_label == "good" else 0.82,
            decision=self.predicted_label,
            production_ready=False,
            deployment_safe=False,
        )


class FailingClassifier:
    def __init__(self, message: str) -> None:
        self.message = message
        self.model_name = "resnet18"
        self.model_version = "0.4.0"
        self.run_id = "1bc92561-c5bf-48f2-8246-b8f3d5718ffe"
        self.threshold = 0.65

    def predict(self, image_path):
        raise RuntimeError(self.message)


class FakeDetector:
    def __init__(self, *, box_count: int) -> None:
        self.box_count = box_count
        self.model_name = "yolo"
        self.model_version = "0.2.0"
        self.run_id = "yolo_train_v0_2_0"
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.7

    def predict(self, image: Image.Image) -> DetectionResult:
        detections: list[DetectionBox] = []
        best_detection = None
        if self.box_count > 0:
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
            detections = [box]
            best_detection = box
        return DetectionResult(
            status="success",
            model_name=self.model_name,
            model_version=self.model_version,
            run_id=self.run_id,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            image_width=image.width,
            image_height=image.height,
            predicted_box_count=self.box_count,
            defect_count=self.box_count,
            detections=detections,
            best_detection=best_detection,
            review_status="review_required" if self.box_count else "no_detection",
            production_ready=False,
            deployment_safe=False,
        )


class FailingDetector:
    def __init__(self, message: str) -> None:
        self.message = message
        self.model_name = "yolo"
        self.model_version = "0.2.0"
        self.run_id = "yolo_train_v0_2_0"
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.7

    def predict(self, image: Image.Image) -> DetectionResult:
        raise RuntimeError(self.message)


class FakeAnomalyDetector:
    def __init__(
        self,
        *,
        predicted_label: str,
        quality_status: str = "review_required",
    ) -> None:
        self.predicted_label = predicted_label
        self.quality_status = quality_status
        self.model_name = "autoencoder"
        self.model_version = "0.1.0"
        self.run_id = "b8ca43f5-0d53-4a42-ab37-b5fca9544a36"
        self.threshold = 0.2043069839477539

    def predict(self, image: Image.Image) -> AnomalyResult:
        score = 0.219 if self.predicted_label == "anomaly" else 0.06873100237623937
        return AnomalyResult(
            status="success",
            model_name=self.model_name,
            model_version=self.model_version,
            run_id=self.run_id,
            anomaly_score=score,
            reconstruction_loss=score,
            threshold=self.threshold,
            predicted_label=self.predicted_label,
            decision=self.predicted_label,
            quality_status=self.quality_status,
            production_ready=False,
            deployment_safe=False,
            optional_reconstruction_artifacts=None,
        )


def _png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(120, 130, 140))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
