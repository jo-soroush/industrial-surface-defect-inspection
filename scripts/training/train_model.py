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

from inspection_ai.models.factory import create_model
from inspection_ai.training.result_persistence import persist_training_result
from inspection_ai.training.train_loop import run_training_loop

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


def handle_classification(config: dict[str, Any]) -> object:
    """Placeholder handler for classification training dispatch."""
    model = create_model(config)
    result = run_training_loop(config=config, model=model, data_loader=None)
    return result


def handle_anomaly_detection(config: dict[str, Any]) -> object:
    """Placeholder handler for anomaly-detection training dispatch."""
    model = create_model(config)
    result = run_training_loop(config=config, model=model, data_loader=None)
    return result


def handle_object_detection(config: dict[str, Any]) -> object:
    """Placeholder handler for object-detection training dispatch."""
    model = create_model(config)
    result = run_training_loop(config=config, model=model, data_loader=None)
    return result


def extract_task_type(config: dict[str, Any]) -> str:
    """Extract and validate task_type from the resolved run config identity block."""
    identity = config.get("identity")

    if identity is None:
        raise ValueError("Training config is missing required section: identity")

    if not isinstance(identity, dict):
        raise ValueError("Training config section identity must be a dictionary")

    task_type = identity.get("task_type")

    if task_type is None:
        raise ValueError(
            "Training config is missing required field: identity.task_type"
        )

    if not isinstance(task_type, str):
        raise ValueError("Training config field identity.task_type must be a string")

    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(
            "Training config has unsupported identity.task_type: "
            f"{task_type}. Allowed values: {sorted(ALLOWED_TASK_TYPES)}"
        )

    return task_type


def dispatch_training(config: dict[str, Any]) -> object:
    """Validate task type and route to the governed training placeholder."""
    task_type = extract_task_type(config)

    if task_type == "classification":
        return handle_classification(config)

    if task_type == "anomaly_detection":
        return handle_anomaly_detection(config)

    if task_type == "object_detection":
        return handle_object_detection(config)

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
    result = dispatch_training(config)
    result_path = persist_training_result(
        result=result,
        output_dir=Path("artifacts/models/analysis/training_results"),
    )
    print(f"Training result created: {type(result).__name__}")
    print(f"Training result saved: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
