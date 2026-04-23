"""Governed evaluation metrics boundary for Phase 3 and Phase 4 handoff.

This module defines the source boundary for evaluation-metrics handling within
the governed ML system. It will eventually provide a consistent metrics payload
layer for classification, anomaly detection, and object detection so that model
evaluation remains reviewable, reproducible, and separated from training and
artifact-writing concerns.
"""

from __future__ import annotations

from typing import Any


def build_metrics_payload(task_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal governed metrics payload wrapper."""
    return {"task_type": task_type, "metrics": metrics}


def compute_classification_metrics(predictions: Any, targets: Any) -> dict[str, Any]:
    """Placeholder for future classification metrics computation."""
    raise NotImplementedError(
        "compute_classification_metrics is not implemented yet."
    )


def compute_anomaly_metrics(scores: Any, targets: Any) -> dict[str, Any]:
    """Placeholder for future anomaly-detection metrics computation."""
    raise NotImplementedError("compute_anomaly_metrics is not implemented yet.")


def compute_detection_metrics(predictions: Any, targets: Any) -> dict[str, Any]:
    """Placeholder for future object-detection metrics computation."""
    raise NotImplementedError("compute_detection_metrics is not implemented yet.")
