"""Persistence boundary for structured TrainingResult payloads.

This module owns saving TrainingResult objects as governed JSON payloads.
Registry integration is intentionally deferred to a later implementation step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspection_ai.training.training_result import TrainingResult


def persist_training_result(
    result: TrainingResult, output_dir: str | Path
) -> Path:
    """Persist a TrainingResult as JSON and return the saved path."""
    if not hasattr(result, "to_dict"):
        raise ValueError("Training result must provide a to_dict() method.")

    run_id = result.identity.get("run_id")
    if not run_id:
        raise ValueError("Training result is missing required identity.run_id.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_payload: dict[str, Any] = result.to_dict()
    result_path = output_path / f"training_result__{run_id}.json"

    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)

    return result_path
