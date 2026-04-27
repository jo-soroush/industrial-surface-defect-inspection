"""Governed writer for completed training run registry entries.

This module tracks run-level records only. Model artifact registration is a
separate governance concern handled by the artifact registry boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from inspection_ai.training.training_result import TrainingResult


def append_run_registry_entry(
    result: TrainingResult,
    result_path: str | Path,
    registry_path: str | Path = "artifacts/models/registry/run_registry.yaml",
) -> None:
    """Append a completed training run entry to the governed run registry."""
    run_id = result.identity.get("run_id")
    if not run_id:
        raise ValueError("Training result is missing required identity.run_id.")

    if not result_path:
        raise ValueError("result_path is required for run registry entries.")

    registry_file = Path(registry_path)
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    if not registry_file.exists():
        registry_data: dict[str, Any] = {"runs": []}
    else:
        with registry_file.open("r", encoding="utf-8") as handle:
            loaded_registry = yaml.safe_load(handle)

        if loaded_registry is None:
            registry_data = {"runs": []}
        elif isinstance(loaded_registry, dict):
            registry_data = loaded_registry
        else:
            raise ValueError("Run registry must contain a YAML dictionary.")

    runs = registry_data.setdefault("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Run registry field 'runs' must be a list.")

    runs.append(
        {
            "run_id": run_id,
            "task_type": result.identity["task_type"],
            "model_type": result.identity["model_type"],
            "status": "completed",
            "result_path": str(result_path),
        }
    )

    with registry_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            registry_data,
            handle,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        )
