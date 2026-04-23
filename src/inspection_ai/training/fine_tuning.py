"""Fine-tuning boundary skeleton for Phase 3 model development.

This module defines the governed source location for future staged fine-tuning
control. It is intended to manage training-stage transitions and fine-tuning
policies later, while intentionally omitting any real training behavior in the
current step.
"""

from __future__ import annotations

from typing import Any


class FineTuningManager:
    """Placeholder interface for staged fine-tuning control."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, model: Any) -> None:
        """Run a fine-tuning placeholder."""
        raise NotImplementedError("FineTuningManager.run is not implemented yet.")


def run_fine_tuning(config: dict[str, Any], model: Any) -> None:
    """Functional placeholder for future fine-tuning execution."""
    raise NotImplementedError("run_fine_tuning is not implemented yet.")
