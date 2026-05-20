"""Helpers for the first frontend scaffold.

This module intentionally exposes only basic path constants and a safe JSON
loader. Full dashboard data wiring will be added in a later phase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
TRACK_A_BUNDLE_DIR = ROOT_DIR / "artifacts" / "frontend" / "track_a"
TRACK_B_BUNDLE_DIR = ROOT_DIR / "artifacts" / "frontend" / "track_b"
DETECTION_BUNDLE_DIR = ROOT_DIR / "artifacts" / "frontend" / "detection" / "yolo_train_v0_2_0"


def load_json_file(path: str | Path) -> dict[str, Any] | list[Any] | None:
    """Load a JSON file safely.

    Returns None when the file does not exist or cannot be parsed.
    """
    json_path = Path(path)
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
