"""API tests for the live YOLO detection endpoint."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.app.main import app
from api.app.schemas.inspection import DetectionBox, DetectionResult
import api.app.routes.predict as predict_routes


client = TestClient(app)


def test_predict_detection_returns_detection_result(monkeypatch) -> None:
    """Valid image uploads return the detection contract without real YOLO inference."""

    class FakeDetector:
        def predict(self, image: Image.Image) -> DetectionResult:
            detection = DetectionBox(
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
            return DetectionResult(
                status="success",
                model_name="yolo",
                model_version="0.2.0",
                run_id="yolo_train_v0_2_0",
                confidence_threshold=0.25,
                iou_threshold=0.7,
                image_width=image.width,
                image_height=image.height,
                predicted_box_count=1,
                defect_count=1,
                detections=[detection],
                best_detection=detection,
                review_status="review_required",
                production_ready=False,
                deployment_safe=False,
                limitations=[
                    "Detection output is local model output and not production-ready.",
                    "Detection output is not deployment-safe.",
                ],
            )

    monkeypatch.setattr(predict_routes, "get_yolo_detector", lambda: FakeDetector())

    response = client.post(
        "/predict/detection",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["predicted_box_count"] == 1
    assert payload["detections"][0]["bbox_format"] == "xyxy"
    assert payload["best_detection"]["class_label"] == "inclusion"
    assert payload["production_ready"] is False
    assert payload["deployment_safe"] is False


def test_predict_detection_rejects_non_image_upload() -> None:
    response = client.post(
        "/predict/detection",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]


def test_predict_detection_rejects_missing_upload() -> None:
    response = client.post("/predict/detection")

    assert response.status_code == 400
    assert "Missing image upload" in response.json()["detail"]
    assert "field 'file'" in response.json()["detail"]


def test_predict_detection_rejects_empty_upload() -> None:
    response = client.post(
        "/predict/detection",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert "empty or missing" in response.json()["detail"]


def test_predict_detection_rejects_corrupt_image_payload() -> None:
    response = client.post(
        "/predict/detection",
        files={"file": ("surface.png", b"not a real png", "image/png")},
    )

    assert response.status_code == 400
    assert "could not be decoded as an image" in response.json()["detail"]


def test_predict_detection_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setattr(predict_routes, "MAX_UPLOAD_BYTES", 8)

    response = client.post(
        "/predict/detection",
        files={"file": ("surface.png", b"012345678", "image/png")},
    )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_predict_detection_returns_clean_detector_failure(monkeypatch) -> None:
    """Detector dependency failures return clean service errors without real inference."""

    class FailingDetector:
        def predict(self, image: Image.Image) -> DetectionResult:
            raise RuntimeError("YOLO live detection requires the 'ultralytics' package.")

    monkeypatch.setattr(predict_routes, "get_yolo_detector", lambda: FailingDetector())

    response = client.post(
        "/predict/detection",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "YOLO detection failed" in response.json()["detail"]
    assert "ultralytics" in response.json()["detail"]


def test_predict_detection_returns_clean_file_not_found_failure(monkeypatch) -> None:
    class FailingDetector:
        def predict(self, image: Image.Image) -> DetectionResult:
            raise FileNotFoundError("missing best.pt")

    monkeypatch.setattr(predict_routes, "get_yolo_detector", lambda: FailingDetector())

    response = client.post(
        "/predict/detection",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "YOLO detection dependency is unavailable" in response.json()["detail"]


def test_predict_detection_returns_clean_invalid_output_failure(monkeypatch) -> None:
    class FailingDetector:
        def predict(self, image: Image.Image) -> DetectionResult:
            raise ValueError("bbox_xyxy must contain exactly four values")

    monkeypatch.setattr(predict_routes, "get_yolo_detector", lambda: FailingDetector())

    response = client.post(
        "/predict/detection",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert "YOLO detection returned invalid output" in response.json()["detail"]


def test_predict_detection_returns_generic_unexpected_failure(monkeypatch) -> None:
    class FailingDetector:
        def predict(self, image: Image.Image) -> DetectionResult:
            raise Exception("internal implementation detail")

    monkeypatch.setattr(predict_routes, "get_yolo_detector", lambda: FailingDetector())

    response = client.post(
        "/predict/detection",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "YOLO detection failed unexpectedly."
    assert "internal implementation detail" not in response.text


def _png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(120, 130, 140))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
