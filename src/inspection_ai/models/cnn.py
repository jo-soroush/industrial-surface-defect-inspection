"""Minimal trainable CNN classifier for governed classification work."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CNNModel(nn.Module):
    """Lightweight binary CNN classifier returning raw logits."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input tensors shaped [B, 3, H, W]."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("CNNModel.forward expects a 4D torch.Tensor.")

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
