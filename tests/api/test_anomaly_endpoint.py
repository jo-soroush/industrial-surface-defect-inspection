"""API tests for the live surface anomaly detection endpoint."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.app.main import app
from api.app.schemas.inspection import AnomalyResult
import api.app.routes.predict as predict_routes


client = TestClient(app)


def test_predict_anomaly_returns_anomaly_result(monkeypatch) -> None:
    """Valid image uploads return the anomaly contract without real checkpoint inference."""

    class FakeDetector:
        def predict(self, image: Image.Image) -> AnomalyResult:
            return AnomalyResult(
                status="success",
                model_name="autoencoder",
                model_version="0.1.0",
                run_id="b8ca43f5-0d53-4a42-ab37-b5fca9544a36",
                anomaly_score=0.25,
                reconstruction_loss=0.25,
                threshold=0.2043069839477539,
                predicted_label="anomaly",
                decision="anomaly",
                quality_status="review_required_weak_evidence",
                production_ready=False,
                deployment_safe=False,
                limitations=[
                    "Anomaly output is local autoencoder reconstruction evidence and not production-ready.",
                    "Anomaly output is not deployment-safe.",
                ],
                optional_reconstruction_artifacts=None,
            )

    monkeypatch.setattr(predict_routes, "get_anomaly_detector", lambda: FakeDetector())

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["model_name"] == "autoencoder"
    assert payload["anomaly_score"] == 0.25
    assert payload["reconstruction_loss"] == 0.25
    assert payload["predicted_label"] == "anomaly"
    assert payload["decision"] == "anomaly"
    assert payload["production_ready"] is False
    assert payload["deployment_safe"] is False
    assert payload["optional_reconstruction_artifacts"] is None


def test_predict_anomaly_rejects_non_image_upload() -> None:
    response = client.post(
        "/predict/anomaly",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]


def test_predict_anomaly_rejects_missing_upload() -> None:
    response = client.post("/predict/anomaly")

    assert response.status_code == 400
    assert "Missing image upload" in response.json()["detail"]
    assert "field 'file'" in response.json()["detail"]


def test_predict_anomaly_rejects_empty_upload() -> None:
    response = client.post(
        "/predict/anomaly",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert "empty or missing" in response.json()["detail"]


def test_predict_anomaly_rejects_corrupt_image_payload() -> None:
    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", b"not a real png", "image/png")},
    )

    assert response.status_code == 400
    assert "could not be decoded as an image" in response.json()["detail"]


def test_predict_anomaly_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setattr(predict_routes, "MAX_UPLOAD_BYTES", 8)

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", b"012345678", "image/png")},
    )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_predict_anomaly_returns_clean_file_not_found_failure(monkeypatch) -> None:
    class FailingDetector:
        def predict(self, image: Image.Image) -> AnomalyResult:
            raise FileNotFoundError("missing autoencoder checkpoint")

    monkeypatch.setattr(predict_routes, "get_anomaly_detector", lambda: FailingDetector())

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "Anomaly detection dependency is unavailable" in response.json()["detail"]


def test_predict_anomaly_returns_dependency_failure_when_factory_missing_file(
    monkeypatch,
) -> None:
    def fail_factory():
        raise FileNotFoundError("missing anomaly evaluation")

    monkeypatch.setattr(predict_routes, "get_anomaly_detector", fail_factory)

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "Anomaly detection dependency is unavailable" in response.json()["detail"]


def test_predict_anomaly_returns_dependency_failure_when_factory_config_invalid(
    monkeypatch,
) -> None:
    def fail_factory():
        raise ValueError("Unsupported anomaly score_definition: 'unknown'")

    monkeypatch.setattr(predict_routes, "get_anomaly_detector", fail_factory)

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "Anomaly detection dependency is unavailable" in detail
    assert "invalid output" not in detail


def test_predict_anomaly_returns_dependency_failure_when_factory_runtime_fails(
    monkeypatch,
) -> None:
    def fail_factory():
        raise RuntimeError("Anomaly model construction failed: invalid config")

    monkeypatch.setattr(predict_routes, "get_anomaly_detector", fail_factory)

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "Anomaly detection dependency is unavailable" in response.json()["detail"]


def test_predict_anomaly_returns_clean_invalid_output_failure(monkeypatch) -> None:
    class FailingDetector:
        def predict(self, image: Image.Image) -> AnomalyResult:
            raise ValueError("threshold must be numeric")

    monkeypatch.setattr(predict_routes, "get_anomaly_detector", lambda: FailingDetector())

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert "Anomaly detection returned invalid output" in response.json()["detail"]


def test_predict_anomaly_returns_dependency_failure_for_checkpoint_runtime(monkeypatch) -> None:
    class FailingDetector:
        def predict(self, image: Image.Image) -> AnomalyResult:
            raise RuntimeError("Anomaly checkpoint loading failed: invalid state_dict")

    monkeypatch.setattr(predict_routes, "get_anomaly_detector", lambda: FailingDetector())

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "Anomaly detection dependency is unavailable" in response.json()["detail"]


def test_predict_anomaly_returns_generic_unexpected_failure(monkeypatch) -> None:
    class FailingDetector:
        def predict(self, image: Image.Image) -> AnomalyResult:
            raise Exception("internal implementation detail")

    monkeypatch.setattr(predict_routes, "get_anomaly_detector", lambda: FailingDetector())

    response = client.post(
        "/predict/anomaly",
        files={"file": ("surface.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Anomaly detection failed unexpectedly."
    assert "internal implementation detail" not in response.text


def _png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(120, 130, 140))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
