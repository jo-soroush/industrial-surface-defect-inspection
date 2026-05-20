"""Reusable Track A classification inference helper.

This module provides a small, local-only inference boundary for the governed
Track A ResNet18 candidate. It loads the selected checkpoint, applies the
existing preprocessing contract, and returns a structured prediction result
without introducing an API endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from ..models.resnet18 import ResNet18Model
from ..preprocessing.image_to_tensor import load_and_preprocess_image
from ..training.checkpointing import (
    extract_model_state_dict,
    load_checkpoint_payload,
    resolve_model_checkpoint_path,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_MODEL_NAME = "resnet18"
EXPECTED_MODEL_VERSION = "0.4.0"
EXPECTED_RUN_ID = "1bc92561-c5bf-48f2-8246-b8f3d5718ffe"
EXPECTED_RUN_CONFIG_ID = "resnet18_train_v0_4_0"
EXPECTED_CONFIG_ID = "resnet18_config"
EXPECTED_THRESHOLD = 0.65
EXPECTED_CLASS_TO_INDEX = {"good": 0, "defect": 1}

RUN_CONFIG_PATH = REPO_ROOT / "configs" / "runs" / f"{EXPECTED_RUN_CONFIG_ID}.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "models" / f"{EXPECTED_MODEL_NAME}.yaml"
PREPROCESSING_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "preprocessing_mvtec.yaml"
CLASS_MAPPING_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "class_mapping_mvtec_binary.yaml"
QUALITY_DECISION_PATH = (
    REPO_ROOT
    / "artifacts"
    / "models"
    / "analysis"
    / f"track_a_resnet18_v0_4_0_quality_decision__{EXPECTED_RUN_ID}.json"
)


@dataclass(frozen=True)
class TrackAPredictionResult:
    """Structured Track A prediction output."""

    model_name: str
    model_version: str
    run_id: str
    threshold: float
    predicted_label: str
    predicted_label_id: int
    probability_good: float
    probability_defect: float
    decision: str
    production_ready: bool = False
    deployment_safe: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the prediction result as a plain dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "threshold": self.threshold,
            "predicted_label": self.predicted_label,
            "predicted_label_id": self.predicted_label_id,
            "probability_good": self.probability_good,
            "probability_defect": self.probability_defect,
            "decision": self.decision,
            "production_ready": self.production_ready,
            "deployment_safe": self.deployment_safe,
        }


class TrackAClassifier:
    """Load and run the governed Track A ResNet18 classifier."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = _resolve_device(device)

        self.run_config = _load_yaml_file(RUN_CONFIG_PATH, "Track A run config")
        self.model_config = _load_yaml_file(MODEL_CONFIG_PATH, "Track A model config")
        self.preprocessing_config = _load_yaml_file(
            PREPROCESSING_CONFIG_PATH, "Track A preprocessing config"
        )
        self.class_mapping_config = _load_yaml_file(
            CLASS_MAPPING_CONFIG_PATH, "Track A class mapping config"
        )
        self.quality_decision = _load_json_file(
            QUALITY_DECISION_PATH, "Track A quality decision"
        )

        self.model_name = _require_string(
            self.run_config.get("model_identity", {}).get("model_name"),
            "model_identity.model_name",
        )
        self.model_version = _require_string(
            self.run_config.get("model_identity", {}).get("model_version"),
            "model_identity.model_version",
        )
        self.run_id = _require_string(
            self.quality_decision.get("run_id"), "quality_decision.run_id"
        )
        self.threshold = _require_number(
            self.quality_decision.get("recommended_threshold"),
            "quality_decision.recommended_threshold",
        )

        self.index_to_class = _build_index_to_class(self.class_mapping_config)
        self.class_to_index = _build_class_to_index(self.class_mapping_config)
        self.checkpoint_path = REPO_ROOT / resolve_model_checkpoint_path(self.run_id)
        self.model = self._load_model()

    def predict(self, image_path: str | Path) -> TrackAPredictionResult:
        """Run the Track A classifier on one local image path."""
        validated_path = _validate_image_path(image_path)

        try:
            image_tensor = load_and_preprocess_image(
                str(validated_path), self.preprocessing_config
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"Track A preprocessing failed for {validated_path}: {exc}"
            ) from exc

        if not isinstance(image_tensor, torch.Tensor):
            raise RuntimeError("Track A preprocessing did not return a tensor.")

        batch = image_tensor.unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                logits = self.model(batch)
        except Exception as exc:  # pragma: no cover - narrow inference boundary
            raise RuntimeError(f"Track A inference failed for {validated_path}: {exc}") from exc

        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Track A model did not return a tensor.")
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
            raise RuntimeError("Track A model must return logits with shape [1, 2].")

        probabilities = torch.softmax(logits, dim=1)[0]
        good_index = self.class_to_index["good"]
        defect_index = self.class_to_index["defect"]
        probability_good = float(probabilities[good_index].item())
        probability_defect = float(probabilities[defect_index].item())

        predicted_label_id = defect_index if probability_defect >= self.threshold else good_index
        predicted_label = self.index_to_class[predicted_label_id]
        decision = predicted_label

        return TrackAPredictionResult(
            model_name=self.model_name,
            model_version=self.model_version,
            run_id=self.run_id,
            threshold=float(self.threshold),
            predicted_label=predicted_label,
            predicted_label_id=predicted_label_id,
            probability_good=probability_good,
            probability_defect=probability_defect,
            decision=decision,
            production_ready=False,
            deployment_safe=False,
        )

    def _load_model(self) -> torch.nn.Module:
        """Build the governed model and load its checkpoint weights."""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Track A checkpoint not found: {self.checkpoint_path}")

        model_config = self._build_inference_model_config()
        try:
            model = ResNet18Model(model_config)
        except Exception as exc:  # pragma: no cover - configuration boundary
            raise RuntimeError(f"Track A model construction failed: {exc}") from exc

        if not hasattr(model, "load_state_dict"):
            raise RuntimeError("Track A model does not support load_state_dict().")

        try:
            payload = load_checkpoint_payload(self.checkpoint_path)
            state_dict = extract_model_state_dict(payload)
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint does not contain a valid state_dict.")
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - checkpoint boundary
            raise RuntimeError(f"Track A checkpoint loading failed: {exc}") from exc

        model.to(self.device)
        model.eval()
        return model

    def _build_inference_model_config(self) -> dict[str, Any]:
        """Build a local-only model config that avoids any external weight fetches."""
        model_identity = dict(self.run_config.get("model_identity", {}))
        model_identity["pretrained_policy"] = {"pretrained": False}
        model_identity.setdefault("model_config_id", EXPECTED_CONFIG_ID)
        model_identity.setdefault("model_name", EXPECTED_MODEL_NAME)
        model_identity.setdefault("model_type", EXPECTED_MODEL_NAME)
        model_identity.setdefault("model_version", EXPECTED_MODEL_VERSION)

        inference_config = dict(self.model_config)
        inference_config["model_identity"] = model_identity
        return inference_config


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string or torch.device.")
    return torch.device(device)


def _validate_image_path(image_path: str | Path) -> Path:
    if isinstance(image_path, Path):
        path = image_path
    elif isinstance(image_path, str) and image_path:
        path = Path(image_path)
    else:
        raise ValueError("image_path must be a non-empty string or pathlib.Path.")

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return path


def _load_yaml_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} YAML is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} YAML must contain an object: {path}")
    return payload


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object: {path}")
    return payload


def _build_index_to_class(class_mapping: dict[str, Any]) -> dict[int, str]:
    raw_mapping = class_mapping.get("index_to_class")
    if not isinstance(raw_mapping, dict):
        raise ValueError("Class mapping is missing index_to_class.")

    index_to_class: dict[int, str] = {}
    for key, value in raw_mapping.items():
        if isinstance(key, str) and key.isdigit():
            index = int(key)
        elif isinstance(key, int):
            index = key
        else:
            raise ValueError("Class mapping index_to_class keys must be integers.")
        if not isinstance(value, str) or not value:
            raise ValueError("Class mapping index_to_class values must be strings.")
        index_to_class[index] = value

    if 0 not in index_to_class or 1 not in index_to_class:
        raise ValueError("Class mapping must define classes 0 and 1.")
    return index_to_class


def _build_class_to_index(class_mapping: dict[str, Any]) -> dict[str, int]:
    raw_mapping = class_mapping.get("class_to_index")
    if not isinstance(raw_mapping, dict):
        raise ValueError("Class mapping is missing class_to_index.")

    class_to_index: dict[str, int] = {}
    for key, value in raw_mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Class mapping class_to_index keys must be strings.")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Class mapping class_to_index values must be integers.")
        class_to_index[key] = value

    if "good" not in class_to_index or "defect" not in class_to_index:
        raise ValueError("Class mapping must define good and defect indices.")
    return class_to_index


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric.")
    return float(value)
