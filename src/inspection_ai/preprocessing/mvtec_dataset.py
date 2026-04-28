"""Governed MVTec dataset wrappers for preprocessing."""

from __future__ import annotations

from typing import Any

import torch

from inspection_ai.preprocessing.image_to_tensor import load_and_preprocess_image
from inspection_ai.preprocessing.label_mapping import map_label


class MVTecBinaryClassificationDataset(torch.utils.data.Dataset):
    """Thin Dataset wrapper for governed MVTec binary classification entries."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        preprocessing_config: dict[str, Any],
        class_mapping_config: dict[str, Any],
    ) -> None:
        if not isinstance(entries, list):
            raise ValueError("Dataset entries must be a list.")

        self.entries = entries
        self.preprocessing_config = preprocessing_config
        self.class_mapping_config = class_mapping_config

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Dataset index must be an integer.")
        if index < 0 or index >= len(self.entries):
            raise IndexError("Dataset index is out of range.")

        entry = self.entries[index]
        if not isinstance(entry, dict):
            raise ValueError("Dataset entry must be a dictionary.")

        path = _require_entry_string(entry, "path")
        raw_label = _require_entry_string(entry, "label")

        image = load_and_preprocess_image(path, self.preprocessing_config)
        label = map_label(raw_label, self.class_mapping_config)

        return {
            "image": image,
            "label": label,
            "path": path,
            "category": entry.get("category"),
            "raw_label": raw_label,
            "split": entry.get("split"),
        }


def _require_entry_string(entry: dict[str, Any], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Dataset entry field {field} must be a non-empty string.")

    return value
