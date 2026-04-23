"""Checkpointing boundary skeleton for Phase 3 training governance.

This module defines the governed source location for future checkpoint path
resolution and checkpoint-save policy handling. In the current step it
establishes import-safe interfaces only and intentionally avoids any file-write
behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_checkpoint_path(run_id: str, checkpoint_name: str) -> Path:
    """Return the governed checkpoint path for a future save operation."""
    return Path("artifacts/models/checkpoints") / run_id / checkpoint_name


def save_checkpoint(model_state: Any, target_path: Path) -> None:
    """Placeholder for future checkpoint persistence logic."""
    raise NotImplementedError("save_checkpoint is not implemented yet.")


class CheckpointPolicy:
    """Placeholder interface for future checkpoint policy handling."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def should_save(self) -> bool:
        """Return a placeholder checkpoint policy decision."""
        raise NotImplementedError("CheckpointPolicy.should_save is not implemented yet.")
