"""Minimal ResNet-18 model skeleton for Phase 3 classification work.

This module reserves the canonical source location for a ResNet-18-based
classification model. The current Phase 3 step defines only an import-safe
interface and does not include network layers, pretrained loading, or training
behavior.
"""

from __future__ import annotations

from typing import Any


class ResNet18Model:
    """Placeholder ResNet-18 classification model."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def forward(self, x: Any) -> Any:
        """Run a forward pass placeholder."""
        raise NotImplementedError("ResNet18Model.forward is not implemented yet.")
