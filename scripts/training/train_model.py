"""Governed training entrypoint skeleton for Phase 3.

This script defines the CLI boundary for future model-training execution. In
Phase 3, its role is to provide a clean, governed entrypoint that will
eventually load training configuration and dispatch the appropriate training
flow without embedding training logic directly in the script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

ALLOWED_TASK_TYPES = {
    "classification",
    "anomaly_detection",
    "object_detection",
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a governed training config file."""
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Training config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Training config YAML is invalid: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Unable to read training config: {path}") from exc

    if config is None:
        raise ValueError(f"Training config is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Training config must parse to a dictionary at this stage: {path}"
        )

    return config


def handle_classification(config: dict[str, Any]) -> None:
    """Placeholder handler for classification training dispatch."""
    raise NotImplementedError("Classification training is not implemented yet.")


def handle_anomaly_detection(config: dict[str, Any]) -> None:
    """Placeholder handler for anomaly-detection training dispatch."""
    raise NotImplementedError("Anomaly-detection training is not implemented yet.")


def handle_object_detection(config: dict[str, Any]) -> None:
    """Placeholder handler for object-detection training dispatch."""
    raise NotImplementedError("Object-detection training is not implemented yet.")


def dispatch_training(config: dict[str, Any]) -> None:
    """Validate task type and route to the governed training placeholder."""
    task_type = config.get("task_type")

    if task_type is None:
        raise ValueError("Training config is missing required field: task_type")

    if not isinstance(task_type, str):
        raise ValueError("Training config field task_type must be a string")

    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(
            "Training config has unsupported task_type: "
            f"{task_type}. Allowed values: {sorted(ALLOWED_TASK_TYPES)}"
        )

    if task_type == "classification":
        handle_classification(config)
        return

    if task_type == "anomaly_detection":
        handle_anomaly_detection(config)
        return

    if task_type == "object_detection":
        handle_object_detection(config)
        return

    raise RuntimeError(f"Unhandled task_type dispatch path: {task_type}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the governed training entrypoint."""
    parser = argparse.ArgumentParser(description="Governed model training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to the training config.")
    return parser


def main() -> int:
    """Parse CLI arguments and invoke the training dispatch placeholder."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    dispatch_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
