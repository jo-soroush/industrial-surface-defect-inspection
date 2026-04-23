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


def load_config(config_path: str) -> dict[str, Any]:
    """Return minimal placeholder config metadata for a requested path."""
    return {"config_path": str(Path(config_path))}


def dispatch_training(config: dict[str, Any]) -> None:
    """Placeholder for future governed training dispatch."""
    raise NotImplementedError("dispatch_training is not implemented yet.")


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
