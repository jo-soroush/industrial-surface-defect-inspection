"""Artifact registry utilities for governed model artifact tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ALLOWED_ARTIFACT_STATUSES = {"active", "archived", "failed", "deprecated"}


class ArtifactRegistry:
    """Load, validate, update, and save a YAML artifact registry.

    The default registry structure is ``{"artifacts": []}``. Registry changes
    remain in memory until callers explicitly persist them with
    ``save_registry``.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self._registry: dict[str, Any] | None = None

    def load_registry(self) -> dict[str, Any]:
        """Load an existing registry YAML or return an empty registry."""
        if not self.registry_path.exists():
            self._registry = {"artifacts": []}
            return self._registry

        if not self.registry_path.is_file():
            raise ValueError(f"Artifact registry path is not a file: {self.registry_path}")

        try:
            with self.registry_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Artifact registry YAML is invalid: {self.registry_path}"
            ) from exc

        if payload is None:
            payload = {"artifacts": []}
        if not isinstance(payload, dict):
            raise ValueError("Artifact registry must contain a YAML object.")

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

    def register_artifact(self, artifact_metadata: dict[str, Any]) -> dict[str, Any]:
        """Register an artifact in memory and return the updated registry.

        Registering the same ``artifact_id`` with identical metadata is
        idempotent. Registering the same ``artifact_id`` with different
        metadata raises a clear error.
        """
        self.validate_artifact_entry(artifact_metadata)
        registry = self._registry if self._registry is not None else self.load_registry()
        artifacts = registry["artifacts"]
        artifact_id = artifact_metadata["artifact_id"]

        for existing_entry in artifacts:
            if existing_entry.get("artifact_id") != artifact_id:
                continue
            if existing_entry == artifact_metadata:
                return registry
            raise ValueError(
                "Artifact registry already contains artifact_id with different "
                f"metadata: {artifact_id}"
            )

        artifacts.append(dict(artifact_metadata))
        return registry

    def validate_artifact_entry(self, artifact_metadata: dict[str, Any]) -> None:
        """Validate one artifact metadata entry."""
        if not isinstance(artifact_metadata, dict):
            raise ValueError("Artifact metadata must be a dictionary.")

        _require_non_empty_string(artifact_metadata.get("artifact_id"), "artifact_id")
        _require_non_empty_string(
            artifact_metadata.get("artifact_path"), "artifact_path"
        )
        _require_non_empty_string(
            artifact_metadata.get("artifact_hash"), "artifact_hash"
        )

        status = artifact_metadata.get("status")
        if status not in ALLOWED_ARTIFACT_STATUSES:
            raise ValueError(
                "Artifact status must be one of: "
                f"{sorted(ALLOWED_ARTIFACT_STATUSES)}."
            )

        size_value = artifact_metadata.get("artifact_size_bytes")
        if size_value is not None and (
            isinstance(size_value, bool)
            or not isinstance(size_value, int)
            or size_value < 0
        ):
            raise ValueError("artifact_size_bytes must be a non-negative integer.")

    def _validate_registry(self, registry: dict[str, Any]) -> None:
        if not isinstance(registry, dict):
            raise ValueError("Artifact registry must be a dictionary.")

        artifacts = registry.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("Artifact registry field 'artifacts' must be a list.")

        seen: dict[str, dict[str, Any]] = {}
        for index, artifact_entry in enumerate(artifacts):
            try:
                self.validate_artifact_entry(artifact_entry)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid artifact registry entry at index {index}."
                ) from exc

            artifact_id = artifact_entry["artifact_id"]
            previous_entry = seen.get(artifact_id)
            if previous_entry is None:
                seen[artifact_id] = artifact_entry
            elif previous_entry != artifact_entry:
                raise ValueError(
                    "Artifact registry contains duplicate artifact_id with "
                    f"different metadata: {artifact_id}"
                )


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value
