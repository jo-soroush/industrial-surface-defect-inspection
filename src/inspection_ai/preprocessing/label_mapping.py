"""Config-driven label mapping utilities for preprocessing."""

from __future__ import annotations

from typing import Any


def map_label(label: str, config: dict[str, Any]) -> int:
    """Map one raw manifest label to a governed integer class index."""
    if not isinstance(label, str) or not label:
        raise ValueError("Label must be a non-empty string.")

    class_to_index = _validate_class_to_index(config)
    if label == "good":
        return class_to_index["good"]

    return class_to_index["defect"]


def _validate_class_to_index(config: dict[str, Any]) -> dict[str, int]:
    if not isinstance(config, dict):
        raise ValueError("Class mapping config must be a dictionary.")

    class_to_index = config.get("class_to_index")
    if not isinstance(class_to_index, dict):
        raise ValueError("Class mapping config must contain class_to_index.")

    good_index = class_to_index.get("good")
    defect_index = class_to_index.get("defect")
    if isinstance(good_index, bool) or not isinstance(good_index, int):
        raise ValueError("class_to_index.good must be an integer.")
    if isinstance(defect_index, bool) or not isinstance(defect_index, int):
        raise ValueError("class_to_index.defect must be an integer.")

    return {
        "good": good_index,
        "defect": defect_index,
    }
