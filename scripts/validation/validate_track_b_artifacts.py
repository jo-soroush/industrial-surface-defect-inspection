"""Validate canonical Track B anomaly detection artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Real
from pathlib import Path
import sys
from typing import Any


TRACK_ID = "track_b"
TASK_TYPE = "anomaly_detection"
DATASET_ID = "mvtec_anomaly"
MODEL_TYPE = "autoencoder"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate canonical Track B anomaly detection artifacts."
    )
    parser.add_argument("--training-result", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--inventory", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    training_result_path = Path(args.training_result)
    evaluation_path = Path(args.evaluation)
    inventory_path = Path(args.inventory)

    training_result = _validate_training_result(training_result_path)
    evaluation = _validate_evaluation(evaluation_path, training_result)
    _validate_inventory(inventory_path, training_result, evaluation)

    print("track_b_artifact_contract=pass")
    return 0


def _validate_training_result(path: Path) -> dict[str, Any]:
    payload = _load_json_file(path, "Track B TrainingResult")
    identity = _require_dict(payload.get("identity"), "training_result.identity")
    metadata = _require_dict(payload.get("metadata"), "training_result.metadata")
    artifacts = _require_dict(payload.get("artifacts"), "training_result.artifacts")
    metrics = _require_dict(payload.get("metrics"), "training_result.metrics")
    learning_curves = _require_dict(
        payload.get("learning_curves"), "training_result.learning_curves"
    )

    if identity.get("task_type") != TASK_TYPE:
        raise ValueError("TrainingResult task_type must be anomaly_detection.")
    if identity.get("model_type") != MODEL_TYPE:
        raise ValueError("TrainingResult model_type must be autoencoder.")
    if metadata.get("dataset_id") != DATASET_ID:
        raise ValueError("TrainingResult dataset_id must be mvtec_anomaly.")
    if metadata.get("model_name") != MODEL_TYPE:
        raise ValueError("TrainingResult model_name must be autoencoder.")
    _require_string(identity.get("run_id"), "training_result.identity.run_id")
    _require_string(identity.get("run_config_id"), "training_result.identity.run_config_id")
    _require_string(metadata.get("split_manifest_path"), "metadata.split_manifest_path")
    _require_non_negative_int(metadata.get("train_sample_count"), "train_sample_count")
    _require_non_negative_int(metadata.get("validation_sample_count"), "validation_sample_count")
    _require_non_negative_int(metadata.get("test_sample_count"), "test_sample_count")
    _require_numeric(metrics.get("reconstruction_loss"), "metrics.reconstruction_loss")
    if "train_loss" not in learning_curves:
        raise ValueError("TrainingResult learning_curves must include train_loss.")
    if "val_loss" not in learning_curves:
        raise ValueError("TrainingResult learning_curves must include val_loss.")

    model_artifact = _require_dict(
        artifacts.get("model_artifact"), "training_result.artifacts.model_artifact"
    )
    if model_artifact.get("type") != "pytorch_state_dict":
        raise ValueError("TrainingResult model_artifact.type must be pytorch_state_dict.")
    checkpoint_path = Path(
        _require_string(model_artifact.get("path"), "model_artifact.path")
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    return payload


def _validate_evaluation(
    path: Path, training_result: dict[str, Any]
) -> dict[str, Any]:
    payload = _load_json_file(path, "Track B anomaly evaluation")
    identity = _require_dict(training_result.get("identity"), "training_result.identity")
    metadata = _require_dict(training_result.get("metadata"), "training_result.metadata")
    artifacts = _require_dict(training_result.get("artifacts"), "training_result.artifacts")

    if payload.get("artifact_type") != "anomaly_detection_evaluation":
        raise ValueError("Evaluation artifact_type must be anomaly_detection_evaluation.")
    if payload.get("task_type") != TASK_TYPE:
        raise ValueError("Evaluation task_type must be anomaly_detection.")
    if payload.get("track_id") != TRACK_ID:
        raise ValueError("Evaluation track_id must be track_b.")
    if payload.get("run_id") != identity.get("run_id"):
        raise ValueError("Evaluation run_id must match TrainingResult.")
    if payload.get("model_id") != metadata.get("model_name"):
        raise ValueError("Evaluation model_id must match TrainingResult model_name.")
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError("Evaluation model_type must be autoencoder.")
    if payload.get("dataset_id") != metadata.get("dataset_id"):
        raise ValueError("Evaluation dataset_id must match TrainingResult dataset_id.")
    if payload.get("config_id") != identity.get("run_config_id"):
        raise ValueError("Evaluation config_id must match TrainingResult run_config_id.")

    checkpoint_path = _require_dict(artifacts.get("model_artifact"), "model_artifact").get("path")
    if payload.get("source_model_checkpoint") != checkpoint_path:
        raise ValueError("Evaluation source_model_checkpoint must match TrainingResult.")

    metrics = _require_dict(payload.get("metrics"), "evaluation.metrics")
    for field in ("roc_auc", "precision", "recall", "f1"):
        _require_numeric(metrics.get(field), f"evaluation.metrics.{field}")

    counts = _require_dict(payload.get("counts"), "evaluation.counts")
    train_count = _require_positive_int(counts.get("train_score_count"), "train_score_count")
    test_count = _require_positive_int(counts.get("test_score_count"), "test_score_count")
    normal_count = _require_non_negative_int(counts.get("normal_test_count"), "normal_test_count")
    anomaly_count = _require_non_negative_int(counts.get("anomaly_test_count"), "anomaly_test_count")
    predicted_normal = _require_non_negative_int(
        counts.get("predicted_normal_count"), "predicted_normal_count"
    )
    predicted_anomaly = _require_non_negative_int(
        counts.get("predicted_anomaly_count"), "predicted_anomaly_count"
    )
    correct_count = _require_non_negative_int(counts.get("correct_count"), "correct_count")
    incorrect_count = _require_non_negative_int(
        counts.get("incorrect_count"), "incorrect_count"
    )
    if train_count != metadata.get("train_sample_count"):
        raise ValueError("Evaluation train_score_count must match TrainingResult.")
    if test_count != metadata.get("test_sample_count"):
        raise ValueError("Evaluation test_score_count must match TrainingResult.")
    if normal_count + anomaly_count != test_count:
        raise ValueError("Evaluation normal/anomaly counts must sum to test count.")
    if predicted_normal + predicted_anomaly != test_count:
        raise ValueError("Evaluation prediction counts must sum to test count.")
    if correct_count + incorrect_count != test_count:
        raise ValueError("Evaluation correct/incorrect counts must sum to test count.")

    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != test_count:
        raise ValueError("Evaluation samples length must match test_score_count.")
    _validate_sample(samples[0], "samples[0]")

    return payload


def _validate_inventory(
    path: Path,
    training_result: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    payload = _load_json_file(path, "Track B inventory")
    identity = _require_dict(training_result.get("identity"), "training_result.identity")
    metadata = _require_dict(training_result.get("metadata"), "training_result.metadata")

    if payload.get("artifact_type") != "track_b_artifact_inventory":
        raise ValueError("Inventory artifact_type must be track_b_artifact_inventory.")
    if payload.get("task_type") != TASK_TYPE:
        raise ValueError("Inventory task_type must be anomaly_detection.")
    if payload.get("track_id") != TRACK_ID:
        raise ValueError("Inventory track_id must be track_b.")
    if payload.get("run_id") != identity.get("run_id"):
        raise ValueError("Inventory run_id must match TrainingResult.")
    if payload.get("canonical_run_id") != identity.get("run_id"):
        raise ValueError("Inventory canonical_run_id must match TrainingResult.")
    if payload.get("model_id") != metadata.get("model_name"):
        raise ValueError("Inventory model_id must match TrainingResult model_name.")
    if payload.get("dataset_id") != metadata.get("dataset_id"):
        raise ValueError("Inventory dataset_id must match TrainingResult dataset_id.")
    if payload.get("config_id") != identity.get("run_config_id"):
        raise ValueError("Inventory config_id must match TrainingResult run_config_id.")

    artifacts = _require_dict(payload.get("artifacts"), "inventory.artifacts")
    _validate_inventory_artifact(artifacts, "training_result", "TrainingResult")
    _validate_inventory_artifact(artifacts, "model_checkpoint", "model_checkpoint")
    _validate_inventory_artifact(
        artifacts, "anomaly_evaluation", "anomaly_detection_evaluation"
    )

    linkage = _require_dict(payload.get("linkage"), "inventory.linkage")
    if linkage.get("evaluation_source_training_result") != evaluation.get("source_training_result"):
        raise ValueError("Inventory evaluation source TrainingResult linkage is invalid.")
    if linkage.get("evaluation_source_model_checkpoint") != evaluation.get("source_model_checkpoint"):
        raise ValueError("Inventory evaluation source checkpoint linkage is invalid.")


def _validate_inventory_artifact(
    artifacts: dict[str, Any], name: str, expected_artifact_type: str
) -> None:
    entry = _require_dict(artifacts.get(name), f"inventory.artifacts.{name}")
    if entry.get("exists") is not True:
        raise ValueError(f"Inventory artifact {name} must have exists=true.")
    if entry.get("artifact_type") != expected_artifact_type:
        raise ValueError(
            f"Inventory artifact {name} artifact_type must be {expected_artifact_type}."
        )
    path = Path(_require_string(entry.get("path"), f"inventory.artifacts.{name}.path"))
    if not path.is_file():
        raise FileNotFoundError(f"Inventory artifact {name} not found: {path}")
    expected_sha = _require_string(
        entry.get("sha256"), f"inventory.artifacts.{name}.sha256"
    )
    if _sha256(path) != expected_sha:
        raise ValueError(f"Inventory artifact {name} checksum does not match file.")


def _validate_sample(sample: Any, field_name: str) -> None:
    sample_dict = _require_dict(sample, field_name)
    for key in (
        "sample_id",
        "image_path",
        "true_label",
        "true_label_id",
        "defect_type",
        "anomaly_score",
        "predicted_label",
        "predicted_label_id",
        "correct",
    ):
        if key not in sample_dict:
            raise ValueError(f"{field_name}.{key} is required.")
    _require_numeric(sample_dict.get("anomaly_score"), f"{field_name}.anomaly_score")
    if not isinstance(sample_dict.get("correct"), bool):
        raise ValueError(f"{field_name}.correct must be boolean.")


def _load_json_file(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact_name} JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON must contain an object.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_numeric(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    value = _require_non_negative_int(value, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
