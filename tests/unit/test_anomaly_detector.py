"""Unit tests for the single-image anomaly detection helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

import src.inspection_ai.inference.anomaly_detector as anomaly_detector_module
from src.inspection_ai.inference.anomaly_detector import (
    AnomalyDetector,
    CHECKPOINT_PATH,
    EVALUATION_PATH,
    EXPECTED_RUN_ID,
    PREPROCESSING_CONFIG_PATH,
    build_anomaly_result,
)


GOVERNED_THRESHOLD = 0.2043069839477539


def test_build_anomaly_result_maps_score_below_threshold_to_normal() -> None:
    result = build_anomaly_result(
        anomaly_score=0.1,
        reconstruction_loss=0.1,
        threshold=GOVERNED_THRESHOLD,
    )

    assert result.status == "success"
    assert result.predicted_label == "normal"
    assert result.decision == "normal"
    assert result.anomaly_score == 0.1
    assert result.reconstruction_loss == 0.1
    assert result.threshold == GOVERNED_THRESHOLD
    assert result.production_ready is False
    assert result.deployment_safe is False
    assert result.optional_reconstruction_artifacts is None


def test_build_anomaly_result_maps_score_above_threshold_to_anomaly() -> None:
    result = build_anomaly_result(
        anomaly_score=0.25,
        reconstruction_loss=0.25,
        threshold=GOVERNED_THRESHOLD,
        quality_status="review_required_weak_evidence",
    )

    assert result.predicted_label == "anomaly"
    assert result.decision == "anomaly"
    assert result.quality_status == "review_required_weak_evidence"


def test_build_anomaly_result_includes_traceability() -> None:
    result = build_anomaly_result(
        anomaly_score=0.25,
        reconstruction_loss=0.25,
        threshold=GOVERNED_THRESHOLD,
    )

    assert result.run_id == EXPECTED_RUN_ID
    assert result.traceability.checkpoint_path == CHECKPOINT_PATH.relative_to(
        Path.cwd()
    ).as_posix()
    assert result.traceability.evaluation_path == EVALUATION_PATH.relative_to(
        Path.cwd()
    ).as_posix()
    assert result.traceability.model_config_path == "configs/models/autoencoder.yaml"


def test_build_anomaly_result_rejects_non_numeric_score() -> None:
    with pytest.raises(ValueError, match="anomaly_score must be numeric"):
        build_anomaly_result(
            anomaly_score="0.1",  # type: ignore[arg-type]
            reconstruction_loss=0.1,
            threshold=GOVERNED_THRESHOLD,
        )


def test_anomaly_detector_loads_governed_metadata_without_model_loading() -> None:
    detector = AnomalyDetector()

    assert detector.checkpoint_path == CHECKPOINT_PATH
    assert detector.evaluation_path == EVALUATION_PATH
    assert detector.threshold == GOVERNED_THRESHOLD
    assert detector.score_definition == "mean_squared_reconstruction_error_per_image"
    assert detector.quality_status == "review_required_weak_evidence"
    assert detector._model is None


def test_anomaly_detector_rejects_invalid_image_input_without_model_loading() -> None:
    detector = AnomalyDetector()

    with pytest.raises(TypeError, match="PIL.Image.Image"):
        detector.predict("not an image")  # type: ignore[arg-type]

    assert detector._model is None


def test_anomaly_detector_uses_shared_pil_preprocessing_before_model_load(monkeypatch) -> None:
    calls = []

    def fake_preprocess(image, config):
        calls.append((image, config))
        return torch.zeros((3, 224, 224), dtype=torch.float32)

    class FakeModel:
        def __call__(self, batch):
            return batch

    detector = AnomalyDetector()
    monkeypatch.setattr(anomaly_detector_module, "preprocess_pil_image", fake_preprocess)
    monkeypatch.setattr(detector, "_load_model", lambda: FakeModel())

    image = Image.new("RGB", (12, 8), color=(100, 120, 140))
    result = detector.predict(image)

    assert calls == [(image, detector.preprocessing_config)]
    assert result.status == "success"
    assert result.anomaly_score == 0.0
    assert result.reconstruction_loss == 0.0
    assert result.predicted_label == "normal"
    assert result.threshold == GOVERNED_THRESHOLD


def test_expected_governed_files_exist() -> None:
    assert CHECKPOINT_PATH.is_file()
    assert EVALUATION_PATH.is_file()
    assert PREPROCESSING_CONFIG_PATH.is_file()
