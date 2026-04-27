"""Validation boundary for structured TrainingResult payloads."""

from __future__ import annotations

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


def _require_section(
    payload: dict[str, Any], section: str, expected_type: type
) -> Any:
    value = payload[section]
    if not isinstance(value, expected_type):
        raise ValueError(f"Training result section {section} must be a dictionary.")
    return value
