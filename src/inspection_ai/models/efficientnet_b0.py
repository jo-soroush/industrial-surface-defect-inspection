"""Minimal EfficientNet-B0 model skeleton for Phase 3 classification work.

This module defines the governed placeholder for an EfficientNet-B0
classification model. It provides the interface expected by future Phase 3
training code while deliberately omitting architecture and runtime behavior.
"""

from __future__ import annotations

from typing import Any


class EfficientNetB0Model:
    """Placeholder EfficientNet-B0 classification model."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def forward(self, x: Any) -> Any:
        """Run a forward pass placeholder."""
        raise NotImplementedError(
            "EfficientNetB0Model.forward is not implemented yet."
        )
