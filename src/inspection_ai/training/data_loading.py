"""Training data-loader construction boundary.

This module owns future construction of training data loaders. Notebooks must
not become the canonical implementation for data-loading behavior. Framework
dataset construction, image loading, and batching are intentionally deferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def build_data_loaders(config: dict[str, Any]) -> dict[str, Any]:
    """Return governed split entries for a training config."""
    dataset_binding = config.get("dataset_binding")
    if dataset_binding is None:
        raise ValueError("Training config is missing required dataset_binding section.")
    if not isinstance(dataset_binding, dict):
        raise ValueError("Training config section dataset_binding must be a dictionary.")

    dataset_id = dataset_binding.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise ValueError(
            "Training config is missing required dataset_binding.dataset_id."
        )

    dataset_version = dataset_binding.get("dataset_version")
    if not isinstance(dataset_version, str):
        raise ValueError(
            "Training config is missing required dataset_binding.dataset_version."
        )

    split_manifest_path = dataset_binding.get("split_manifest_path")
    if not isinstance(split_manifest_path, str) or not split_manifest_path:
        raise ValueError(
            "Training config is missing required dataset_binding.split_manifest_path."
        )

    identity = config.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Training config is missing required identity section.")

    task_type = identity.get("task_type")
    if not isinstance(task_type, str):
        raise ValueError("Training config is missing required identity.task_type.")

    manifest = _load_split_manifest(split_manifest_path)
    _validate_manifest_identity(manifest, dataset_id, dataset_version)
    train_entries = _require_split_entries(manifest, "train")
    validation_entries = _require_split_entries(manifest, "validation")
    test_entries = _require_split_entries(manifest, "test")

    return {
        "task_type": task_type,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "split_manifest_path": split_manifest_path,
        "train": train_entries,
        "validation": validation_entries,
        "test": test_entries,
    }


def _load_split_manifest(split_manifest_path: str) -> dict[str, Any]:
    manifest_path = Path(split_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest file not found: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Split manifest YAML is invalid: {manifest_path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must parse to a dictionary.")

    return manifest


def _validate_manifest_identity(
    manifest: dict[str, Any], dataset_id: str, dataset_version: str
) -> None:
    manifest_dataset_id = manifest.get("dataset_id")
    if manifest_dataset_id != dataset_id:
        raise ValueError(
            "Split manifest dataset_id does not match training config dataset_id."
        )

    manifest_dataset_version = manifest.get("dataset_version")
    if manifest_dataset_version != dataset_version:
        raise ValueError(
            "Split manifest dataset_version does not match training config dataset_version."
        )


def _require_split_entries(
    manifest: dict[str, Any], split_name: str
) -> list[dict[str, Any]]:
    field_name = f"{split_name}_entries"
    entries = manifest.get(field_name)
    if not isinstance(entries, list):
        raise ValueError(f"Split manifest {field_name} must be a list.")

    for index, entry in enumerate(entries):
        _validate_split_entry(entry, split_name, field_name, index)

    return entries


def _validate_split_entry(
    entry: Any, split_name: str, field_name: str, index: int
) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"Split manifest {field_name}[{index}] must be a dictionary.")

    for field in ("path", "category", "split", "label"):
        value = entry.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Split manifest {field_name}[{index}].{field} must be a non-empty string."
            )

    if entry["split"] != split_name:
        raise ValueError(
            f"Split manifest {field_name}[{index}].split must match {split_name}."
        )
