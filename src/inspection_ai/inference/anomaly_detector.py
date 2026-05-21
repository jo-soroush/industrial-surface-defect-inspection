"""Reusable single-image anomaly detection helper for image inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from ..contracts.inspection import AnomalyResult, AnomalyTraceability
from ..evaluation.anomaly_evaluation import (
    compute_reconstruction_scores,
    generate_predictions,
)
from ..models.autoencoder import AutoencoderModel
from ..preprocessing.image_to_tensor import preprocess_pil_image
from ..training.checkpointing import extract_model_state_dict, load_checkpoint_payload


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_RUN_ID = "b8ca43f5-0d53-4a42-ab37-b5fca9544a36"
DEFAULT_MODEL_NAME = "autoencoder"
DEFAULT_MODEL_VERSION = "0.1.0"
DEFAULT_DEVICE = "cpu"

RUN_CONFIG_PATH = REPO_ROOT / "configs/runs/autoencoder_train_v0_1_0.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "configs/models/autoencoder.yaml"
PREPROCESSING_CONFIG_PATH = REPO_ROOT / "configs/data/preprocessing_mvtec.yaml"
EVALUATION_PATH = (
    REPO_ROOT
    / "artifacts/models/metrics/"
    "anomaly_detection_evaluation__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
)
CHECKPOINT_PATH = (
    REPO_ROOT
    / "artifacts/models/checkpoints/"
    "model_checkpoint__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.pt"
)

SUPPORTED_SCORE_DEFINITION = "mean_squared_reconstruction_error_per_image"


class AnomalyDetector:
    """Run governed autoencoder anomaly scoring on one image."""

    def __init__(
        self,
        *,
        checkpoint_path: Path = CHECKPOINT_PATH,
        evaluation_path: Path = EVALUATION_PATH,
        device: str | torch.device = DEFAULT_DEVICE,
    ) -> None:
        self.device = _resolve_device(device)
        self.checkpoint_path = checkpoint_path
        self.evaluation_path = evaluation_path
        self.run_config = _load_yaml_file(RUN_CONFIG_PATH, "anomaly run config")
        self.model_config = _load_yaml_file(MODEL_CONFIG_PATH, "anomaly model config")
        self.preprocessing_config = _load_yaml_file(
            PREPROCESSING_CONFIG_PATH, "anomaly preprocessing config"
        )
        self.evaluation = _load_json_file(evaluation_path, "anomaly evaluation")

        self.model_name = _require_string(
            self.model_config.get("model_name"), "model_config.model_name"
        )
        self.model_version = str(
            _require_string_or_number(
                self.model_config.get("model_version"), "model_config.model_version"
            )
        )
        self.run_id = _require_string(self.evaluation.get("run_id"), "evaluation.run_id")
        self.threshold = _require_number(self.evaluation.get("threshold"), "evaluation.threshold")
        self.score_definition = _require_string(
            self.evaluation.get("score_definition"), "evaluation.score_definition"
        )
        if self.score_definition != SUPPORTED_SCORE_DEFINITION:
            raise ValueError(
                "Unsupported anomaly score_definition: "
                f"{self.score_definition!r}."
            )

        self.quality_status = _quality_status_from_evaluation(self.evaluation)
        self._model: torch.nn.Module | None = None

    def predict(self, image: Image.Image) -> AnomalyResult:
        """Run single-image anomaly scoring for a PIL image."""
        if not isinstance(image, Image.Image):
            raise TypeError("AnomalyDetector.predict requires a PIL.Image.Image input.")

        image_tensor = preprocess_pil_image(image, self.preprocessing_config)
        batch = image_tensor.unsqueeze(0).to(self.device)
        model = self._load_model()

        with torch.no_grad():
            reconstruction = model(batch)
        if not isinstance(reconstruction, torch.Tensor):
            raise RuntimeError("Anomaly model did not return a tensor.")
        if reconstruction.shape != batch.shape:
            raise RuntimeError(
                "Anomaly model output shape must match input batch shape."
            )

        scores = compute_reconstruction_scores(batch, reconstruction)
        if len(scores) != 1:
            raise RuntimeError("Single-image anomaly inference must produce one score.")
        reconstruction_loss = float(scores[0])

        return build_anomaly_result(
            anomaly_score=reconstruction_loss,
            reconstruction_loss=reconstruction_loss,
            threshold=self.threshold,
            model_name=self.model_name,
            model_version=self.model_version,
            run_id=self.run_id,
            quality_status=self.quality_status,
            checkpoint_path=self.checkpoint_path,
            evaluation_path=self.evaluation_path,
        )

    def _load_model(self) -> torch.nn.Module:
        """Build and load the autoencoder lazily."""
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Anomaly checkpoint not found: {_repo_relative(self.checkpoint_path)}"
            )

        try:
            model = AutoencoderModel(self.model_config)
        except Exception as exc:  # pragma: no cover - configuration boundary
            raise RuntimeError(f"Anomaly model construction failed: {exc}") from exc

        try:
            payload = load_checkpoint_payload(self.checkpoint_path)
            state_dict = extract_model_state_dict(payload)
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint does not contain a valid state_dict.")
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - checkpoint boundary
            raise RuntimeError(f"Anomaly checkpoint loading failed: {exc}") from exc

        model.to(self.device)
        model.eval()
        self._model = model
        return self._model


def build_anomaly_result(
    *,
    anomaly_score: float,
    reconstruction_loss: float,
    threshold: float,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    run_id: str = EXPECTED_RUN_ID,
    quality_status: str = "review_required",
    checkpoint_path: Path = CHECKPOINT_PATH,
    evaluation_path: Path = EVALUATION_PATH,
) -> AnomalyResult:
    """Build a contract-compatible anomaly result from numeric model outputs."""
    score = _require_number(anomaly_score, "anomaly_score")
    loss = _require_number(reconstruction_loss, "reconstruction_loss")
    threshold_value = _require_number(threshold, "threshold")
    prediction = generate_predictions([score], threshold_value)[0]
    predicted_label = "anomaly" if prediction == 1 else "normal"

    return AnomalyResult(
        status="success",
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        anomaly_score=score,
        reconstruction_loss=loss,
        threshold=threshold_value,
        predicted_label=predicted_label,
        decision=predicted_label,
        quality_status=quality_status,
        production_ready=False,
        deployment_safe=False,
        limitations=[
            "Anomaly output is local autoencoder reconstruction evidence and not production-ready.",
            "Anomaly output is not deployment-safe.",
            "Current governed evaluation shows weak anomaly recall and requires review.",
        ],
        traceability=AnomalyTraceability(
            checkpoint_path=_repo_relative(checkpoint_path),
            run_config_path=_repo_relative(RUN_CONFIG_PATH),
            model_config_path=_repo_relative(MODEL_CONFIG_PATH),
            evaluation_path=_repo_relative(evaluation_path),
        ),
        optional_reconstruction_artifacts=None,
    )


def _quality_status_from_evaluation(evaluation: dict[str, Any]) -> str:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        return "review_required"
    roc_auc = metrics.get("roc_auc")
    recall = metrics.get("recall")
    f1 = metrics.get("f1")
    if (
        isinstance(roc_auc, int | float)
        and isinstance(recall, int | float)
        and isinstance(f1, int | float)
        and (float(roc_auc) < 0.5 or float(recall) < 0.1 or float(f1) < 0.1)
    ):
        return "review_required_weak_evidence"
    return "review_required"


def _load_yaml_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {_repo_relative(path)}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} YAML is invalid: {_repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} YAML must contain an object: {_repo_relative(path)}")
    return payload


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {_repo_relative(path)}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON is invalid: {_repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object: {_repo_relative(path)}")
    return payload


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string or torch.device.")
    return torch.device(device)


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _require_string_or_number(value: Any, field: str) -> str | int | float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"{field} must be a string or number.")
    if isinstance(value, str) and not value:
        raise ValueError(f"{field} must not be empty.")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric.")
    return float(value)


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
