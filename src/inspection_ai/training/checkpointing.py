"""Checkpointing boundary for governed training artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def resolve_model_checkpoint_path(run_id: str) -> Path:
    """Return the governed model checkpoint path for a training run."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string.")
    return Path("artifacts/models/checkpoints") / f"model_checkpoint__{run_id}.pt"


def resolve_checkpoint_path(run_id: str, checkpoint_name: str) -> Path:
    """Return the governed checkpoint path for a future save operation."""
    return Path("artifacts/models/checkpoints") / run_id / checkpoint_name


def save_checkpoint(model_state: Any, target_path: Path) -> None:
    """Persist a model checkpoint payload without overwriting existing artifacts."""
    if not isinstance(target_path, Path):
        raise TypeError("target_path must be a pathlib.Path.")
    if target_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, target_path)


def load_checkpoint_payload(path: Path) -> Any:
    """Load a governed checkpoint payload from disk."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path.")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def extract_model_state_dict(checkpoint_payload: Any) -> Any:
    """Return the model state_dict from either a raw or structured checkpoint."""
    if isinstance(checkpoint_payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint_payload.get(key)
            if isinstance(candidate, dict):
                return candidate
    return checkpoint_payload


def build_structured_checkpoint_payload(
    *,
    model_state_dict: Any,
    result: Any,
    checkpoint_kind: str,
    validation_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Build a governed structured checkpoint payload for Track A classification."""
    if not isinstance(checkpoint_kind, str) or not checkpoint_kind:
        raise ValueError("checkpoint_kind must be a non-empty string.")
    if checkpoint_kind not in {"final", "best"}:
        raise ValueError("checkpoint_kind must be either 'final' or 'best'.")
    if not isinstance(validation_evaluation, dict):
        raise ValueError("validation_evaluation must be a dictionary.")

    identity = getattr(result, "identity", None)
    metadata = getattr(result, "metadata", None)
    if not isinstance(identity, dict):
        raise ValueError("TrainingResult identity must be a dictionary.")
    if not isinstance(metadata, dict):
        raise ValueError("TrainingResult metadata must be a dictionary.")

    run_id = _require_string(identity.get("run_id"), "result.identity.run_id")
    run_config_id = _require_string(
        identity.get("run_config_id"), "result.identity.run_config_id"
    )
    model_type = _require_string(identity.get("model_type"), "result.identity.model_type")
    model_version = _require_string(
        metadata.get("model_version"), "result.metadata.model_version"
    )
    validation_path = _require_string(
        metadata.get("validation_evaluation_path"),
        "result.metadata.validation_evaluation_path",
    )
    if validation_evaluation.get("artifact_type") != "classification_validation_evaluation":
        raise ValueError(
            "validation_evaluation.artifact_type must be classification_validation_evaluation."
        )
    if validation_evaluation.get("run_id") != run_id:
        raise ValueError("validation_evaluation.run_id must match result.identity.run_id.")

    epoch_value = metadata.get("real_training_epoch_count", metadata.get("epochs"))
    if isinstance(epoch_value, bool) or not isinstance(epoch_value, int) or epoch_value < 1:
        raise ValueError("Checkpoint payload requires a positive integer epoch.")

    validation_confusion_matrix = _require_binary_confusion_matrix(validation_evaluation)
    validation_metrics = {
        "total_samples": _require_positive_int(
            validation_evaluation.get("total_samples"),
            "validation_evaluation.total_samples",
        ),
        "confusion_matrix": validation_confusion_matrix,
        "per_class": _require_dict(validation_evaluation.get("per_class"), "per_class"),
        "macro_metrics": _require_dict(
            validation_evaluation.get("macro_metrics"), "macro_metrics"
        ),
    }

    return {
        "checkpoint_format": "inspection_ai_track_a_structured_checkpoint_v1",
        "checkpoint_kind": checkpoint_kind,
        "run_id": run_id,
        "run_config_id": run_config_id,
        "model_type": model_type,
        "model_version": model_version,
        "epoch": epoch_value,
        "validation_evaluation_path": validation_path,
        "validation_confusion_matrix": validation_confusion_matrix,
        "validation_metrics": validation_metrics,
        "validation_evaluation": validation_evaluation,
        "model_state_dict": model_state_dict,
    }


class CheckpointPolicy:
    """Placeholder interface for future checkpoint policy handling."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def should_save(self) -> bool:
        """Return a placeholder checkpoint policy decision."""
        raise NotImplementedError("CheckpointPolicy.should_save is not implemented yet.")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dictionary.")
    return value


def _require_binary_confusion_matrix(validation_evaluation: dict[str, Any]) -> list[list[int]]:
    confusion_matrix = validation_evaluation.get("confusion_matrix")
    if not isinstance(confusion_matrix, list) or len(confusion_matrix) != 2:
        raise ValueError("validation_evaluation.confusion_matrix must be a 2x2 list.")

    validated_matrix: list[list[int]] = []
    for row in confusion_matrix:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("validation_evaluation.confusion_matrix must be a 2x2 list.")
        validated_row: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "validation_evaluation.confusion_matrix values must be non-negative integers."
                )
            validated_row.append(value)
        validated_matrix.append(validated_row)

    return validated_matrix
