"""Checkpointing boundary for governed training artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def resolve_model_checkpoint_path(run_id: str) -> Path:
    """Return the governed model checkpoint path for a training run."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string.")
    return Path("artifacts/models/checkpoints") / f"model_checkpoint__{run_id}.pt"


def resolve_checkpoint_path(run_id: str, checkpoint_name: str) -> Path:
    """Return the governed checkpoint path for a future save operation."""
    return Path("artifacts/models/checkpoints") / run_id / checkpoint_name


def save_checkpoint(model_state: Any, target_path: Path) -> None:
    """Persist a model checkpoint without overwriting existing artifacts."""
    if not isinstance(target_path, Path):
        raise TypeError("target_path must be a pathlib.Path.")
    if target_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, target_path)


class CheckpointPolicy:
    """Placeholder interface for future checkpoint policy handling."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def should_save(self) -> bool:
        """Return a placeholder checkpoint policy decision."""
        raise NotImplementedError("CheckpointPolicy.should_save is not implemented yet.")
