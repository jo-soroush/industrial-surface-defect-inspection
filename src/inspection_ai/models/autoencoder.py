"""Convolutional autoencoder model for Track B anomaly detection."""

from __future__ import annotations

from typing import Any

import torch


class AutoencoderModel(torch.nn.Module):
    """Simple convolutional autoencoder that reconstructs RGB MVTec images."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an image reconstruction with the same shape as the input."""
        encoded = self.encoder(x)
        return self.decoder(encoded)
