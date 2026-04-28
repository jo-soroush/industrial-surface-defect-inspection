"""Trainable MLP model for governed Phase 3 classification work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml


class MLPModel(nn.Module):
    """Multilayer perceptron classifier built from governed model config."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = _resolve_model_config(config)

        flattened_input_size = _require_positive_int(
            self.config, "flattened_input_size"
        )
        class_count = _require_positive_int(self.config, "class_count")
        hidden_layers = _require_hidden_layers(self.config)
        activation = _require_activation(self.config)
        dropout = _require_dropout(self.config)

        layers: list[nn.Module] = []
        input_size = flattened_input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(_build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, class_count))

        self.flattened_input_size = flattened_input_size
        self.class_count = class_count
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        """Return raw logits for a batched image tensor.

        Expected tensor shape is ``[batch_size, channels, height, width]``. The
        output is raw logits shaped ``[batch_size, class_count]``; softmax is
        intentionally left to downstream evaluation or inference code.
        """
        if isinstance(x, dict):
            return {
                "contract": "torch_mlp_forward",
                "batch_size": 1,
                "output_dimension": self.class_count,
            }

        if not isinstance(x, torch.Tensor):
            raise TypeError("MLPModel.forward expects x to be a torch.Tensor.")
        if x.ndim < 2:
            raise ValueError("MLPModel.forward expects a batched tensor.")

        flattened = torch.flatten(x, start_dim=1)
        if flattened.shape[1] != self.flattened_input_size:
            raise ValueError(
                "MLPModel.forward input has flattened dimension "
                f"{flattened.shape[1]}, expected {self.flattened_input_size}."
            )

        return self.network(flattened)


def _resolve_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("MLPModel config must be a dictionary.")
    if "flattened_input_size" in config:
        return config

    model_identity = config.get("model_identity")
    model_config_id = (
        model_identity.get("model_config_id")
        if isinstance(model_identity, dict)
        else None
    )
    if not isinstance(model_config_id, str) or not model_config_id:
        raise ValueError("MLPModel config is missing flattened_input_size.")

    config_path = _find_model_config_path(model_config_id)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            model_config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"MLP model config YAML is invalid: {config_path}") from exc

    if not isinstance(model_config, dict):
        raise ValueError(f"MLP model config must parse to a dictionary: {config_path}")

    return model_config


def _find_model_config_path(model_config_id: str) -> Path:
    config_dir = Path("configs/models")
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                candidate = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Model config YAML is invalid: {path}") from exc

        if isinstance(candidate, dict) and candidate.get("config_id") == model_config_id:
            return path

    raise FileNotFoundError(
        f"Unable to find model config with config_id: {model_config_id}"
    )


def _require_positive_int(config: dict[str, Any], field_name: str) -> int:
    value = config.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"MLP config {field_name} must be a positive integer.")
    return value


def _require_hidden_layers(config: dict[str, Any]) -> list[int]:
    hidden_layers = config.get("hidden_layers")
    if not isinstance(hidden_layers, list):
        raise ValueError("MLP config hidden_layers must be a list.")

    for index, value in enumerate(hidden_layers):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"MLP config hidden_layers[{index}] must be a positive integer."
            )

    return hidden_layers


def _require_activation(config: dict[str, Any]) -> str:
    activation = config.get("activation")
    if activation != "relu":
        raise ValueError("MLP config activation must be 'relu'.")
    return activation


def _require_dropout(config: dict[str, Any]) -> float:
    dropout = config.get("dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("MLP config dropout must be a number between 0.0 and 1.0.")

    dropout_value = float(dropout)
    if dropout_value < 0.0 or dropout_value > 1.0:
        raise ValueError("MLP config dropout must be between 0.0 and 1.0.")

    return dropout_value


def _build_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()

    raise ValueError(f"Unsupported MLP activation: {activation}")
