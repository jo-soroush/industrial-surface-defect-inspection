"""Validation boundary for structured TrainingResult payloads."""

from __future__ import annotations

from datetime import datetime
from numbers import Real
from typing import Any

from inspection_ai.training.training_result import TrainingResult


_REQUIRED_TOP_LEVEL_SECTIONS = (
    "identity",
    "metrics",
    "learning_curves",
    "artifacts",
    "metadata",
)

_REQUIRED_IDENTITY_FIELDS = (
    "run_id",
    "run_config_id",
    "task_type",
    "model_type",
    "is_experiment",
    "created_at",
)

_REQUIRED_METADATA_FIELDS = (
    "dataset_id",
    "model_type",
    "model_name",
    "model_version",
    "track_id",
    "dataset_version",
    "preprocessing_version",
    "epochs",
    "device",
    "completed_at",
    "duration_seconds",
)

_REQUIRED_LEARNING_CURVE_FIELDS = (
    "train_loss",
    "val_loss",
)

_REQUIRED_METRICS_BY_TASK_TYPE = {
    "classification": ("accuracy", "f1"),
    "anomaly_detection": ("reconstruction_loss",),
    "object_detection": ("mAP",),
}


def validate_training_result(result: TrainingResult) -> None:
    """Validate that a TrainingResult has the required structured payload."""
    if not hasattr(result, "to_dict"):
        raise ValueError("Training result must provide a to_dict() method.")

    payload = result.to_dict()
    if not isinstance(payload, dict):
        raise ValueError("Training result to_dict() must return a dictionary.")

    for section in _REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in payload:
            raise ValueError(f"Training result is missing section: {section}.")

    identity = _require_section(payload, "identity", dict)
    metrics = _require_section(payload, "metrics", dict)
    learning_curves = _require_section(payload, "learning_curves", dict)
    artifacts = _require_section(payload, "artifacts", dict)
    metadata = _require_section(payload, "metadata", dict)

    _validate_artifacts(artifacts)

    for field in _REQUIRED_LEARNING_CURVE_FIELDS:
        if field not in learning_curves:
            raise ValueError(
                f"Training result learning_curves is missing field: {field}."
            )

    for field in _REQUIRED_IDENTITY_FIELDS:
        if field not in identity:
            raise ValueError(f"Training result identity is missing field: {field}.")

    task_type = identity["task_type"]
    if task_type not in _REQUIRED_METRICS_BY_TASK_TYPE:
        raise ValueError(f"Unsupported training result task_type: {task_type}.")

    for metric_name in _REQUIRED_METRICS_BY_TASK_TYPE[task_type]:
        if metric_name not in metrics:
            raise ValueError(
                "Training result metrics for task_type "
                f"{task_type} are missing required metric: {metric_name}."
            )

    for field in _REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            raise ValueError(f"Training result metadata is missing field: {field}.")

    _validate_timing_metadata(metadata)

    if not isinstance(metrics, dict):
        raise ValueError("Training result metrics must be a dictionary.")
    if not isinstance(learning_curves, dict):
        raise ValueError("Training result learning_curves must be a dictionary.")
    if not isinstance(artifacts, dict):
        raise ValueError("Training result artifacts must be a dictionary.")
    if not isinstance(metadata, dict):
        raise ValueError("Training result metadata must be a dictionary.")
    if not isinstance(identity, dict):
        raise ValueError("Training result identity must be a dictionary.")


def _validate_artifacts(artifacts: dict[Any, Any]) -> None:
    for name, value in artifacts.items():
        if not isinstance(name, str):
            raise ValueError(f"Training result artifact name is invalid: {name}.")

        if isinstance(value, str):
            continue

        if isinstance(value, dict) and isinstance(value.get("path"), str):
            continue

        raise ValueError(
            "Training result artifact "
            f"{name} must be a path string or a dictionary with a string path field."
        )


def _validate_timing_metadata(metadata: dict[str, Any]) -> None:
    completed_at = metadata["completed_at"]
    if not isinstance(completed_at, str):
        raise ValueError("Training result metadata completed_at must be a string.")
    if not completed_at.endswith("Z"):
        raise ValueError("Training result metadata completed_at must end with Z.")
    try:
        datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "Training result metadata completed_at must be ISO 8601 format."
        ) from exc

    duration_seconds = metadata["duration_seconds"]
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, Real):
        raise ValueError(
            "Training result metadata duration_seconds must be a number."
        )
    if duration_seconds < 0:
        raise ValueError(
            "Training result metadata duration_seconds must be greater than or equal to 0."
        )


def _require_section(
    payload: dict[str, Any], section: str, expected_type: type
) -> Any:
    value = payload[section]
    if not isinstance(value, expected_type):
        raise ValueError(f"Training result section {section} must be a dictionary.")
    return value
