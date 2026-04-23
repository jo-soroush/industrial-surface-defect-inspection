"""Governed model comparison boundary for Phase 4 and Phase 5 workflows.

This module defines the source boundary for comparing model candidates using
governed metrics and metadata payloads. It will eventually consume evaluation
outputs and lifecycle metadata to support reviewable model comparison and
selection workflows while remaining separate from training and business
decision logic.
"""

from __future__ import annotations

from typing import Any


def build_comparison_table(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a minimal placeholder comparison table structure."""
    return list(candidates)


def aggregate_model_metrics(
    metrics_payloads: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return a minimal placeholder aggregated metrics structure."""
    return {"candidates": list(metrics_payloads)}


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Placeholder for future governed candidate-selection logic."""
    raise NotImplementedError("select_best_candidate is not implemented yet.")
