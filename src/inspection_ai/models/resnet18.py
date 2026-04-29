"""Trainable ResNet-18 classifier for governed Track A classification work."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import resnet18


class ResNet18Model(nn.Module):
    """Offline-safe ResNet-18 binary classifier returning raw logits."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.network = resnet18(weights=None)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input tensors shaped [B, 3, H, W]."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("ResNet18Model.forward expects x to be a torch.Tensor.")
        if x.ndim != 4:
            raise ValueError("ResNet18Model.forward expects a 4D torch.Tensor.")
        if x.shape[1] != 3:
            raise ValueError("ResNet18Model.forward expects 3 input channels.")

        logits = self.network(x)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("ResNet18Model must output logits with shape [B, 2].")
        return logits
