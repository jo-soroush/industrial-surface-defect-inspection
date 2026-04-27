"""Training loop boundary skeleton for Phase 3 model development.

This module defines the governed source location for future training-loop
execution. In Phase 3 it establishes the interface boundary for orchestrating
epoch-level training behavior without implementing optimization, loss handling,
or model-specific runtime logic yet.
"""

from __future__ import annotations

from typing import Any

from inspection_ai.training.training_result import TrainingResult


class TrainingLoop:
    """Placeholder interface for future training loop orchestration."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, model: Any, data_loader: Any) -> TrainingResult:
        """Run a training loop placeholder."""
        result = TrainingResult(self.config)
        _add_placeholder_outputs(result, self.config)
        _add_data_loader_metadata(result, data_loader)
        return result


def run_training_loop(
    config: dict[str, Any], model: Any, data_loader: Any
) -> TrainingResult:
    """Functional placeholder for future training loop execution."""
    result = TrainingResult(config)
    _add_placeholder_outputs(result, config)
    _add_data_loader_metadata(result, data_loader)
    return result


def _add_placeholder_outputs(
    result: TrainingResult, config: dict[str, Any]
) -> None:
    task_type = config["identity"]["task_type"]

    if task_type == "classification":
        result.add_metric("accuracy", 0.5)
        result.add_metric("f1", 0.5)
    elif task_type == "anomaly_detection":
        result.add_metric("reconstruction_loss", 0.1)
    elif task_type == "object_detection":
        result.add_metric("mAP", 0.3)

    result.add_learning_point("train_loss", [1.0, 0.8, 0.6])
    result.add_learning_point("val_loss", [1.1, 0.9, 0.7])

    training_runtime = config.get("training_runtime", {})
    result.add_metadata("epochs", training_runtime.get("epochs"))
    result.add_metadata("device", training_runtime.get("device"))


def _add_data_loader_metadata(result: TrainingResult, data_loader: Any) -> None:
    if not isinstance(data_loader, dict):
        raise ValueError("data_loader must be a dictionary.")

    dataset_id = data_loader.get("dataset_id")
    task_type_from_loader = data_loader.get("task_type")

    result.add_metadata("dataset_id", dataset_id)
    result.add_metadata("task_type_from_loader", task_type_from_loader)
