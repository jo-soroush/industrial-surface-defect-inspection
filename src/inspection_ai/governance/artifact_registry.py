"""Artifact registry skeleton for Phase 3 governance.

This module defines the minimal source boundary for artifact tracking within
the model lifecycle. In Phase 3 it serves as the governed location for loading
artifact registry state and registering new model artifacts without embedding
that responsibility in training or evaluation code.

This module will eventually handle artifact registry loading, artifact
registration, and artifact metadata tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ArtifactRegistry:
    """Minimal placeholder interface for artifact registry operations."""

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)

    def load_registry(self) -> dict[str, Any]:
        """Return a placeholder registry structure."""
        return {"artifacts": []}

    def register_artifact(self, artifact_metadata: dict[str, Any]) -> None:
        """Placeholder for future artifact registration logic."""
        pass
