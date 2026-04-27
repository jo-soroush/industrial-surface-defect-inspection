"""Training data-loader construction boundary.

This module owns future construction of training data loaders. Notebooks must
not become the canonical implementation for data-loading behavior. Real dataset
access, image loading, and framework-specific loaders are intentionally
deferred.
"""

from __future__ import annotations

from typing import Any


def build_data_loaders(config: dict[str, Any]) -> dict[str, Any]:
    """Return placeholder data-loader slots for a governed training config."""
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

    identity = config.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Training config is missing required identity section.")

    task_type = identity.get("task_type")
    if not isinstance(task_type, str):
        raise ValueError("Training config is missing required identity.task_type.")

    return {
        "task_type": task_type,
        "dataset_id": dataset_id,
        "train": None,
        "validation": None,
        "test": None,
    }
