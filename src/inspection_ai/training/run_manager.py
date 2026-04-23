"""Run-management boundary skeleton for Phase 3 training governance.

This module defines the governed source location for creating and tracking
training-run lifecycle events. It will eventually coordinate run identity,
startup, and finalization behavior without embedding that responsibility in
model or loop code.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4


def create_run_id() -> str:
    """Return a lightweight placeholder run identifier."""
    return str(uuid4())


def start_run(config: dict[str, Any]) -> dict[str, Any]:
    """Return placeholder run-start metadata."""
    return {"run_id": create_run_id(), "status": "started", "config": config}


def finalize_run(run_id: str, status: str) -> dict[str, str]:
    """Return placeholder run-finalization metadata."""
    return {"run_id": run_id, "status": status}
