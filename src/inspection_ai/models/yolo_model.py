"""Minimal YOLO model boundary for Phase 3 object-detection work.

This module defines the governed wrapper boundary for a YOLO-style detection
model. It keeps ultralytics imports lazy so the project can import the wrapper
without requiring the backend package at module import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _load_ultralytics_yolo() -> Any:
    """Return the ultralytics YOLO backend class, or fail with guidance."""
    try:
        ultralytics = import_module("ultralytics")
    except ImportError as exc:
        raise RuntimeError(
            "YOLOModel requires the 'ultralytics' package. Install the backend "
            "dependency before enabling detection training or prediction."
        ) from exc

    yolo_cls = getattr(ultralytics, "YOLO", None)
    if yolo_cls is None:
        raise RuntimeError(
            "The installed 'ultralytics' package does not expose a YOLO class."
        )
    return yolo_cls


class YOLOModel:
    """Placeholder YOLO object-detection model wrapper."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.backend_package = config.get("backend_package", "ultralytics")
        self.backend_status = config.get(
            "backend_status", "dependency_declared_lazy_loaded"
        )

    def _load_backend(self) -> Any:
        """Lazily resolve the configured YOLO backend."""
        if self.backend_package != "ultralytics":
            raise RuntimeError(
                f"Unsupported YOLO backend package: {self.backend_package}"
            )
        return _load_ultralytics_yolo()

    def train(self) -> None:
        """Run training placeholder."""
        raise NotImplementedError(
            "YOLOModel.train is not implemented yet. The ultralytics backend "
            "boundary is present, but governed detection training is not built."
        )

    def predict(self, x: Any | None = None) -> Any:
        """Run prediction placeholder."""
        raise NotImplementedError(
            "YOLOModel.predict is not implemented yet. The ultralytics backend "
            "boundary is present, but governed detection prediction is not built."
        )
