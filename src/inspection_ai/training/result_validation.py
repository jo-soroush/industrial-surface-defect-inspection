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
    "training_config_id",
    "seed",
    "device_policy",
    "split_manifest_path",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "forward_contract_checked",
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

_TRACK_A_SUPERVISED_DATASET_ID = "mvtec_classification_supervised"
_TRACK_A_SUPERVISED_RUN_CONFIG_ID = "mlp_train_supervised_v0_1_0"

_REQUIRED_TRACK_A_SUPERVISED_CLASSIFICATION_METRICS = (
    "train_accuracy",
    "train_f1",
    "val_loss",
    "val_accuracy",
    "val_f1",
)


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

    if _is_track_a_supervised_classification_result(identity, metadata):
        _validate_track_a_supervised_classification_metrics(metrics)

    _validate_timing_metadata(metadata)
    _validate_config_reproducibility_metadata(metadata)
    _validate_split_count_metadata(metadata)
    _validate_forward_contract_metadata(metadata)

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


def _is_track_a_supervised_classification_result(
    identity: dict[str, Any], metadata: dict[str, Any]
) -> bool:
    return (
        identity.get("task_type") == "classification"
        and (
            metadata.get("dataset_id") == _TRACK_A_SUPERVISED_DATASET_ID
            or identity.get("run_config_id") == _TRACK_A_SUPERVISED_RUN_CONFIG_ID
        )
    )


def _validate_track_a_supervised_classification_metrics(
    metrics: dict[str, Any]
) -> None:
    for metric_name in _REQUIRED_TRACK_A_SUPERVISED_CLASSIFICATION_METRICS:
        if metric_name not in metrics:
            raise ValueError(
                "Track A supervised classification metrics are missing required "
                f"metric: {metric_name}."
            )

    for metric_name in ("train_accuracy", "train_f1", "val_accuracy", "val_f1"):
        _validate_unit_interval_metric(metrics, metric_name)

    _validate_non_negative_numeric_metric(metrics, "val_loss")


def _validate_unit_interval_metric(metrics: dict[str, Any], field: str) -> None:
    value = metrics[field]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Training result metric {field} must be numeric.")
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"Training result metric {field} must be between 0.0 and 1.0."
        )


def _validate_non_negative_numeric_metric(
    metrics: dict[str, Any], field: str
) -> None:
    value = metrics[field]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Training result metric {field} must be numeric.")
    if value < 0.0:
        raise ValueError(
            f"Training result metric {field} must be greater than or equal to 0.0."
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


def _validate_config_reproducibility_metadata(metadata: dict[str, Any]) -> None:
    training_config_id = metadata["training_config_id"]
    if not isinstance(training_config_id, str) or not training_config_id:
        raise ValueError(
            "Training result metadata training_config_id must be a non-empty string."
        )

    seed = metadata["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Training result metadata seed must be an integer.")
    if seed < 0:
        raise ValueError(
            "Training result metadata seed must be greater than or equal to 0."
        )

    device_policy = metadata["device_policy"]
    if not isinstance(device_policy, str) or not device_policy:
        raise ValueError(
            "Training result metadata device_policy must be a non-empty string."
        )

    split_manifest_path = metadata["split_manifest_path"]
    if not isinstance(split_manifest_path, str) or not split_manifest_path:
        raise ValueError(
            "Training result metadata split_manifest_path must be a non-empty string."
        )


def _validate_split_count_metadata(metadata: dict[str, Any]) -> None:
    for field in (
        "train_sample_count",
        "validation_sample_count",
        "test_sample_count",
    ):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Training result metadata {field} must be an integer.")
        if value < 0:
            raise ValueError(
                f"Training result metadata {field} must be greater than or equal to 0."
            )


def _validate_forward_contract_metadata(metadata: dict[str, Any]) -> None:
    checked = metadata["forward_contract_checked"]
    if not isinstance(checked, bool):
        raise ValueError(
            "Training result metadata forward_contract_checked must be a boolean."
        )

    optional_fields_present = any(
        field in metadata
        for field in (
            "forward_contract_name",
            "forward_contract_batch_size",
            "forward_contract_output_dimension",
        )
    )
    if checked or optional_fields_present:
        _validate_non_empty_string_metadata(metadata, "forward_contract_name")
        _validate_non_negative_integer_metadata(
            metadata, "forward_contract_batch_size"
        )
        _validate_non_negative_integer_metadata(
            metadata, "forward_contract_output_dimension"
        )


def _validate_non_empty_string_metadata(
    metadata: dict[str, Any], field: str
) -> None:
    value = metadata.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Training result metadata {field} must be a non-empty string."
        )


def _validate_non_negative_integer_metadata(
    metadata: dict[str, Any], field: str
) -> None:
    value = metadata.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Training result metadata {field} must be an integer.")
    if value < 0:
        raise ValueError(
            f"Training result metadata {field} must be greater than or equal to 0."
        )


def _require_section(
    payload: dict[str, Any], section: str, expected_type: type
) -> Any:
    value = payload[section]
    if not isinstance(value, expected_type):
        raise ValueError(f"Training result section {section} must be a dictionary.")
    return value
