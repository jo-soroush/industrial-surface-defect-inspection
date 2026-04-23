"""Run registry skeleton for Phase 3 governance.

This module defines the minimal source boundary for governed run tracking
within the model lifecycle. In Phase 3 it serves as the canonical location for
creating and updating run records instead of scattering run-state handling
across training scripts or artifact code.

This module will eventually handle run entry creation, run status updates, and
run-level traceability metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RunRegistry:
    """Minimal placeholder interface for run registry operations."""

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)

    def create_run_entry(self, run_metadata: dict[str, Any]) -> dict[str, Any]:
        """Return a placeholder run entry."""
        return run_metadata

    def update_run_status(self, run_id: str, status: str) -> None:
        """Placeholder for future run status update logic."""
        pass
