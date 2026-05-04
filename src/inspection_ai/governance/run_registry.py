"""Run registry utilities for governed model lifecycle tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ALLOWED_RUN_STATUSES = {"running", "success", "failed", "stopped"}


class RunRegistry:
    """Load, validate, update, and save a YAML run registry.

    The default registry structure is ``{"runs": []}``. Registry changes are
    kept in memory until callers explicitly persist them with ``save_registry``.
    Failed and stopped runs are preserved like any other run status.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self._registry: dict[str, Any] | None = None

    def load_registry(self) -> dict[str, Any]:
        """Load an existing run registry YAML or return an empty registry."""
        if not self.registry_path.exists():
            self._registry = {"runs": []}
            return self._registry

        if not self.registry_path.is_file():
            raise ValueError(f"Run registry path is not a file: {self.registry_path}")

        try:
            with self.registry_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Run registry YAML is invalid: {self.registry_path}") from exc

        if payload is None:
            payload = {"runs": []}
        if not isinstance(payload, dict):
            raise ValueError("Run registry must contain a YAML object.")

        self._validate_registry(payload)
        self._registry = payload
        return self._registry

    def save_registry(self, registry: dict[str, Any]) -> None:
        """Validate and save a registry dictionary to the configured YAML path."""
        self._validate_registry(registry)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                registry,
                handle,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )
        self._registry = registry

    def register_run(self, run_metadata: dict[str, Any]) -> dict[str, Any]:
        """Register a run in memory and return the updated registry.

        Registering the same ``run_id`` with identical metadata is idempotent.
        Registering the same ``run_id`` with different metadata raises a clear
        error.
        """
        self.validate_run_entry(run_metadata)
        registry = self._registry if self._registry is not None else self.load_registry()
        runs = registry["runs"]
        run_id = run_metadata["run_id"]

        for existing_entry in runs:
            if existing_entry.get("run_id") != run_id:
                continue
            if existing_entry == run_metadata:
                return registry
            raise ValueError(
                "Run registry already contains run_id with different metadata: "
                f"{run_id}"
            )

        runs.append(dict(run_metadata))
        return registry

    def update_run_status(
        self,
        run_id: str,
        status: str,
        ended_at: str | None = None,
    ) -> dict[str, Any]:
        """Update the status of an existing run and return the updated registry."""
        _require_non_empty_string(run_id, "run_id")
        _validate_run_status(status)
        if ended_at is not None:
            _require_non_empty_string(ended_at, "ended_at")

        registry = self._registry if self._registry is not None else self.load_registry()
        for run_entry in registry["runs"]:
            if run_entry.get("run_id") != run_id:
                continue
            run_entry["run_status"] = status
            if ended_at is not None:
                run_entry["ended_at"] = ended_at
            self.validate_run_entry(run_entry)
            return registry

        raise ValueError(f"Run registry does not contain run_id: {run_id}")

    def validate_run_entry(self, run_metadata: dict[str, Any]) -> None:
        """Validate one run metadata entry."""
        if not isinstance(run_metadata, dict):
            raise ValueError("Run metadata must be a dictionary.")

        for field_name in (
            "run_id",
            "model_name",
            "model_version",
            "dataset_id",
            "dataset_version",
            "task_type",
            "config_id",
        ):
            _require_non_empty_string(run_metadata.get(field_name), field_name)

        _validate_run_status(run_metadata.get("run_status"))

        artifact_created = run_metadata.get("artifact_created")
        if not isinstance(artifact_created, bool):
            raise ValueError("artifact_created must be a boolean.")

    def _validate_registry(self, registry: dict[str, Any]) -> None:
        if not isinstance(registry, dict):
            raise ValueError("Run registry must be a dictionary.")

        runs = registry.get("runs")
        if not isinstance(runs, list):
            raise ValueError("Run registry field 'runs' must be a list.")

        seen: dict[str, dict[str, Any]] = {}
        for index, run_entry in enumerate(runs):
            try:
                self.validate_run_entry(run_entry)
            except ValueError as exc:
                raise ValueError(f"Invalid run registry entry at index {index}.") from exc

            run_id = run_entry["run_id"]
            previous_entry = seen.get(run_id)
            if previous_entry is None:
                seen[run_id] = run_entry
            elif previous_entry != run_entry:
                raise ValueError(
                    "Run registry contains duplicate run_id with different "
                    f"metadata: {run_id}"
                )


def _validate_run_status(status: Any) -> None:
    if status not in ALLOWED_RUN_STATUSES:
        raise ValueError(
            f"run_status must be one of: {sorted(ALLOWED_RUN_STATUSES)}."
        )


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value
