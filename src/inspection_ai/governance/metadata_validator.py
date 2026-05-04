"""Schema validation for governed run and artifact metadata."""

from __future__ import annotations

from numbers import Real
from typing import Any


ALLOWED_TASK_TYPES = {"classification", "anomaly_detection", "object_detection"}
ALLOWED_RUN_STATUSES = {"success", "failed", "stopped"}

SHARED_REQUIRED_STRING_FIELDS = (
    "artifact_id",
    "run_id",
    "model_id",
    "model_name",
    "model_type",
    "model_version",
    "dataset_id",
    "dataset_version",
    "task_type",
    "track_id",
    "config_id",
    "training_config_path",
    "split_manifest_id",
    "preprocessing_version",
    "artifact_path",
    "artifact_hash",
    "training_start_time",
    "training_end_time",
    "device_used",
    "framework_name",
    "framework_version",
    "final_metrics_path",
    "training_log_path",
    "run_status",
)

CLASSIFICATION_REQUIRED_STRING_FIELDS = (
    "label_mapping_version",
    "learning_curve_path",
    "confusion_matrix_path",
)
CLASSIFICATION_REQUIRED_NUMERIC_FIELDS = (
    "final_train_loss",
    "final_validation_loss",
    "final_validation_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)

ANOMALY_REQUIRED_STRING_FIELDS = (
    "anomaly_score_definition",
    "threshold_reference",
)
ANOMALY_REQUIRED_NUMERIC_FIELDS = ("final_reconstruction_loss",)

DETECTION_REQUIRED_STRING_FIELDS = ("data_config_path",)
DETECTION_REQUIRED_NUMERIC_FIELDS = (
    "confidence_threshold_used",
    "iou_threshold_used",
    "mAP_50",
    "mAP_50_95",
    "precision",
    "recall",
)

FAILED_RUN_REQUIRED_STRING_FIELDS = ("failure_stage", "failure_reason")


def validate_metadata(metadata: dict[str, Any]) -> bool:
    """Return True when metadata satisfies the governed schema."""
    try:
        validate_metadata_or_raise(metadata)
    except ValueError:
        return False
    return True


def validate_metadata_or_raise(metadata: dict[str, Any]) -> None:
    """Validate governed metadata and raise ValueError for schema violations.

    This function performs schema and type validation only. It intentionally
    does not check whether referenced paths exist on disk.
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")

    for field_name in SHARED_REQUIRED_STRING_FIELDS:
        _require_non_empty_string(metadata.get(field_name), field_name)

    task_type = metadata["task_type"]
    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(f"task_type must be one of: {sorted(ALLOWED_TASK_TYPES)}.")

    run_status = metadata["run_status"]
    if run_status not in ALLOWED_RUN_STATUSES:
        raise ValueError(f"run_status must be one of: {sorted(ALLOWED_RUN_STATUSES)}.")

    _require_non_negative_number(
        metadata.get("training_duration_seconds"),
        "training_duration_seconds",
    )
    _require_non_negative_int(metadata.get("parameter_count"), "parameter_count")
    _require_non_negative_int(
        metadata.get("trainable_parameter_count"),
        "trainable_parameter_count",
    )
    _validate_checkpoint_path(metadata)

    if task_type == "classification":
        _validate_classification_metadata(metadata)
    elif task_type == "anomaly_detection":
        _validate_anomaly_metadata(metadata)
    elif task_type == "object_detection":
        _validate_detection_metadata(metadata)

    if run_status == "failed":
        _validate_failed_run_metadata(metadata)


def _validate_classification_metadata(metadata: dict[str, Any]) -> None:
    for field_name in CLASSIFICATION_REQUIRED_STRING_FIELDS:
        _require_non_empty_string(metadata.get(field_name), field_name)
    for field_name in CLASSIFICATION_REQUIRED_NUMERIC_FIELDS:
        _require_number(metadata.get(field_name), field_name)


def _validate_anomaly_metadata(metadata: dict[str, Any]) -> None:
    for field_name in ANOMALY_REQUIRED_STRING_FIELDS:
        _require_non_empty_string(metadata.get(field_name), field_name)
    for field_name in ANOMALY_REQUIRED_NUMERIC_FIELDS:
        _require_number(metadata.get(field_name), field_name)

    if "validation_reconstruction_loss" in metadata:
        value = metadata["validation_reconstruction_loss"]
        if value is not None:
            _require_number(value, "validation_reconstruction_loss")


def _validate_detection_metadata(metadata: dict[str, Any]) -> None:
    for field_name in DETECTION_REQUIRED_STRING_FIELDS:
        _require_non_empty_string(metadata.get(field_name), field_name)
    for field_name in DETECTION_REQUIRED_NUMERIC_FIELDS:
        _require_number(metadata.get(field_name), field_name)


def _validate_failed_run_metadata(metadata: dict[str, Any]) -> None:
    for field_name in FAILED_RUN_REQUIRED_STRING_FIELDS:
        _require_non_empty_string(metadata.get(field_name), field_name)

    artifact_created = metadata.get("artifact_created")
    if not isinstance(artifact_created, bool):
        raise ValueError("artifact_created must be a boolean for failed runs.")


def _validate_checkpoint_path(metadata: dict[str, Any]) -> None:
    if "checkpoint_path" not in metadata:
        raise ValueError("checkpoint_path is required.")
    checkpoint_path = metadata["checkpoint_path"]
    if checkpoint_path is not None:
        _require_non_empty_string(checkpoint_path, "checkpoint_path")


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _require_non_negative_number(value: Any, field_name: str) -> float:
    number = _require_number(value, field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be >= 0.")
    return number


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value
