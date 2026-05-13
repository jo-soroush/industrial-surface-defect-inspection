"""Trainable ResNet-18 classifier for governed Track A classification work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models as torchvision_models
import yaml


class ResNet18Model(nn.Module):
    """Offline-safe ResNet-18 binary classifier returning raw logits."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.class_count = _resolve_class_count(config)
        weights = _resolve_resnet18_weights(config)
        self.network = torchvision_models.resnet18(weights=weights)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, self.class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input tensors shaped [B, 3, H, W]."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("ResNet18Model.forward expects x to be a torch.Tensor.")
        if x.ndim != 4:
            raise ValueError("ResNet18Model.forward expects a 4D torch.Tensor.")
        if x.shape[1] != 3:
            raise ValueError("ResNet18Model.forward expects 3 input channels.")

        logits = self.network(x)
        if logits.ndim != 2 or logits.shape[1] != self.class_count:
            raise ValueError(
                "ResNet18Model must output logits with shape "
                f"[B, {self.class_count}]."
            )
        return logits


def _resolve_class_count(config: dict[str, Any]) -> int:
    class_count = config.get("class_count")
    if _is_positive_int(class_count):
        return int(class_count)

    model_identity = config.get("model_identity")
    if isinstance(model_identity, dict):
        nested_class_count = model_identity.get("class_count")
        if _is_positive_int(nested_class_count):
            return int(nested_class_count)

        model_config_id = model_identity.get("model_config_id")
        if isinstance(model_config_id, str) and model_config_id:
            model_config = _load_model_config(model_config_id)
            model_config_class_count = model_config.get("class_count")
            if _is_positive_int(model_config_class_count):
                return int(model_config_class_count)

    raise ValueError(
        "ResNet18Model requires a positive integer class_count from the governed "
        "config or model config."
    )


def _resolve_resnet18_weights(config: dict[str, Any]) -> Any:
    pretrained_policy = config.get("pretrained_policy")
    if not isinstance(pretrained_policy, dict):
        model_identity = config.get("model_identity")
        if isinstance(model_identity, dict):
            pretrained_policy = model_identity.get("pretrained_policy")

    if (
        not isinstance(pretrained_policy, dict)
        or pretrained_policy.get("pretrained") is not True
    ):
        return None

    weights_name = pretrained_policy.get("torchvision_weights")
    if not isinstance(weights_name, str) or not weights_name:
        raise ValueError(
            "ResNet18 pretrained_policy.pretrained=true requires "
            "pretrained_policy.torchvision_weights to be one of: IMAGENET1K_V1, DEFAULT."
        )

    resnet18_weights = getattr(torchvision_models, "ResNet18_Weights", None)
    if resnet18_weights is None:
        raise ValueError(
            "Pretrained ResNet18 weights are unavailable in the installed torchvision "
            "version."
        )

    try:
        return getattr(resnet18_weights, weights_name)
    except AttributeError as exc:
        raise ValueError(
            "Unsupported torchvision_weights for ResNet18 pretrained_policy: "
            f"{weights_name!r}. Supported values: IMAGENET1K_V1, DEFAULT."
        ) from exc


def _load_model_config(model_config_id: str) -> dict[str, Any]:
    config_dir = Path("configs/models")
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                candidate = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Model config YAML is invalid: {path}") from exc
        if isinstance(candidate, dict) and candidate.get("config_id") == model_config_id:
            return candidate
    raise FileNotFoundError(
        f"Unable to find governed model config with config_id: {model_config_id}"
    )


def _is_positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0
