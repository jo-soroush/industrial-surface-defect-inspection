"""Structured training result contract for future governed training runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class TrainingResult:
    """Container for training outputs shared across evaluation and operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        task_type = self._extract_task_type(config)
        model_type = self._extract_model_type(config)
        run_config_id = self._extract_run_config_id(config)
        is_experiment = self._extract_is_experiment(config)
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        self.identity = {
            "run_id": str(uuid4()),
            "task_type": task_type,
            "model_type": model_type,
            "run_config_id": run_config_id,
            "is_experiment": is_experiment,
            "created_at": created_at,
        }
        self.metrics: dict[str, Any] = {}
        self.learning_curves: dict[str, Any] = {}
        self.artifacts: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """Return the full structured training result."""
        return {
            "identity": self.identity,
            "metrics": self.metrics,
            "learning_curves": self.learning_curves,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }

    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric value to the result contract."""
        self.metrics[name] = value

    def add_learning_point(self, name: str, value: Any) -> None:
        """Add a learning-curve point or series to the result contract."""
        self.learning_curves[name] = value

    def add_artifact(self, name: str, path: Any) -> None:
        """Add an artifact reference to the result contract."""
        self.artifacts[name] = path

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the result contract."""
        self.metadata[key] = value

    @staticmethod
    def _extract_task_type(config: dict[str, Any]) -> str:
        identity = config.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("Training config is missing required identity section.")

        task_type = identity.get("task_type")
        if not isinstance(task_type, str):
            raise ValueError("Training config is missing required identity.task_type.")

        return task_type

    @staticmethod
    def _extract_model_type(config: dict[str, Any]) -> str:
        model_identity = config.get("model_identity")
        if not isinstance(model_identity, dict):
            raise ValueError(
                "Training config is missing required model_identity section."
            )

        model_type = model_identity.get("model_type")
        if not isinstance(model_type, str):
            raise ValueError(
                "Training config is missing required model_identity.model_type."
            )

        return model_type

    @staticmethod
    def _extract_run_config_id(config: dict[str, Any]) -> str:
        identity = config.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("Training config is missing required identity section.")

        run_config_id = identity.get("run_config_id")
        if not isinstance(run_config_id, str):
            raise ValueError(
                "Training config is missing required identity.run_config_id."
            )

        return run_config_id

    @staticmethod
    def _extract_is_experiment(config: dict[str, Any]) -> bool:
        identity = config.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("Training config is missing required identity section.")

        is_experiment = identity.get("is_experiment", True)
        if not isinstance(is_experiment, bool):
            raise ValueError(
                "Training config field identity.is_experiment must be a boolean."
            )

        return is_experiment
