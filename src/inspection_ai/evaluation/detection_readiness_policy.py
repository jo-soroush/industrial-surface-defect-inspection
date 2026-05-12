"""Governed readiness policy for Detection/YOLO evaluation metrics."""

from __future__ import annotations

from typing import Any


REQUIRED_METRICS = ("precision", "recall", "mAP50", "mAP50_95")

DEFAULT_BASELINE_METRICS = {
    "precision": 0.00477,
    "recall": 0.54003,
    "mAP50": 0.04518,
    "mAP50_95": 0.01651,
}

DEFAULT_THRESHOLDS = {
    "not_ready_max_mAP50": 0.10,
    "not_ready_max_mAP50_95": 0.03,
    "extremely_low_precision": 0.05,
    "meaningful_mAP50_delta": 0.05,
    "meaningful_mAP50_95_delta": 0.02,
    "review_required_mAP50": 0.25,
    "model_ready_candidate_mAP50": 0.50,
    "model_ready_candidate_mAP50_95": 0.25,
    "model_ready_candidate_precision": 0.40,
    "model_ready_candidate_recall": 0.40,
    "production_ready_mAP50": 0.70,
    "production_ready_mAP50_95": 0.45,
    "production_ready_precision": 0.60,
    "production_ready_recall": 0.60,
}


def evaluate_detection_readiness(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
    evidence_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a governed Detection/YOLO readiness decision.

    The thresholds are internal project gates for staged review. They are not
    universal industrial safety guarantees and do not replace dataset, visual,
    class-level, or deployment review.
    """

    evidence_flags = evidence_flags or {}
    thresholds = dict(DEFAULT_THRESHOLDS)
    normalized_metrics, metric_errors = _normalize_metrics(metrics)
    baseline_comparison = _compare_to_baseline(
        normalized_metrics,
        baseline_metrics or DEFAULT_BASELINE_METRICS,
    )
    blocking_reasons: list[str] = list(metric_errors)
    improvement_signals: list[str] = []
    notes = [
        "Readiness is based on governed validation metrics and required evidence flags.",
        "Production readiness requires stronger evidence than validation metrics alone.",
    ]

    if not metric_errors:
        precision = normalized_metrics["precision"]
        recall = normalized_metrics["recall"]
        map50 = normalized_metrics["mAP50"]
        map50_95 = normalized_metrics["mAP50_95"]

        if map50 <= thresholds["not_ready_max_mAP50"]:
            blocking_reasons.append("mAP50 remains below the minimum not-ready gate.")
        if map50_95 <= thresholds["not_ready_max_mAP50_95"]:
            blocking_reasons.append("mAP50_95 remains below the minimum not-ready gate.")
        if precision < thresholds["extremely_low_precision"]:
            blocking_reasons.append("precision is extremely low.")

        if baseline_comparison.get("meaningful_mAP50_improvement"):
            improvement_signals.append("mAP50 improved meaningfully over baseline.")
        if baseline_comparison.get("meaningful_mAP50_95_improvement"):
            improvement_signals.append("mAP50_95 improved meaningfully over baseline.")
        if precision >= thresholds["model_ready_candidate_precision"]:
            improvement_signals.append("precision is above the model-ready candidate floor.")
        if recall >= thresholds["model_ready_candidate_recall"]:
            improvement_signals.append("recall is above the model-ready candidate floor.")

    if blocking_reasons:
        status = "not_ready"
        level = "low_initial_baseline"
        reason = "Detection metrics do not clear the minimum governed readiness gates."
    else:
        precision = normalized_metrics["precision"]
        recall = normalized_metrics["recall"]
        map50 = normalized_metrics["mAP50"]
        map50_95 = normalized_metrics["mAP50_95"]
        meaningful_improvement = (
            baseline_comparison.get("meaningful_mAP50_improvement") is True
            and baseline_comparison.get("meaningful_mAP50_95_improvement") is True
        )
        balanced_candidate = (
            map50 >= thresholds["model_ready_candidate_mAP50"]
            and map50_95 >= thresholds["model_ready_candidate_mAP50_95"]
            and precision >= thresholds["model_ready_candidate_precision"]
            and recall >= thresholds["model_ready_candidate_recall"]
        )
        production_metric_candidate = (
            map50 >= thresholds["production_ready_mAP50"]
            and map50_95 >= thresholds["production_ready_mAP50_95"]
            and precision >= thresholds["production_ready_precision"]
            and recall >= thresholds["production_ready_recall"]
        )
        production_evidence_complete = all(
            evidence_flags.get(flag) is True
            for flag in (
                "test_evaluation_exists",
                "class_level_metrics_exist",
                "visual_review_completed",
                "audit_approved",
            )
        )

        if production_metric_candidate and production_evidence_complete:
            status = "production_ready"
            level = "production_ready"
            reason = "Production metric and evidence gates are satisfied."
        elif balanced_candidate:
            status = "model_ready_candidate"
            level = "model_ready_candidate"
            reason = "Validation metrics satisfy model-ready candidate thresholds; test and visual review are still required."
        elif map50 >= thresholds["review_required_mAP50"] or precision >= thresholds["model_ready_candidate_precision"] or recall >= thresholds["model_ready_candidate_recall"]:
            status = "review_required"
            level = "review_required"
            reason = "Some metrics are promising, but the balance or evidence is not sufficient for model-ready status."
        elif meaningful_improvement:
            status = "improved_baseline"
            level = "improved_baseline"
            reason = "Metrics improve meaningfully over the baseline but remain below model-ready thresholds."
        else:
            status = "not_ready"
            level = "low_initial_baseline"
            reason = "Metrics clear the minimum floor but do not show enough improvement over baseline."

    return {
        "production_readiness": status,
        "readiness_status": status,
        "readiness_level": level,
        "performance_level": level,
        "decision_reason": reason,
        "summary": reason,
        "recommendation_note": _recommendation_for_status(status),
        "blocking_reasons": blocking_reasons,
        "improvement_signals": improvement_signals,
        "metric_thresholds": thresholds,
        "baseline_comparison": baseline_comparison,
        "notes": notes,
    }


def _normalize_metrics(metrics: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    normalized: dict[str, float] = {}
    errors: list[str] = []
    for metric_name in REQUIRED_METRICS:
        value = metrics.get(metric_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{metric_name} is missing or non-numeric.")
            continue
        normalized[metric_name] = float(value)
    return normalized, errors


def _compare_to_baseline(
    metrics: dict[str, float],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for metric_name in REQUIRED_METRICS:
        baseline_value = baseline_metrics.get(metric_name)
        current_value = metrics.get(metric_name)
        if isinstance(baseline_value, bool) or not isinstance(baseline_value, (int, float)):
            comparison[metric_name] = {"baseline": None, "current": current_value, "delta": None}
            continue
        if current_value is None:
            comparison[metric_name] = {"baseline": float(baseline_value), "current": None, "delta": None}
            continue
        comparison[metric_name] = {
            "baseline": float(baseline_value),
            "current": current_value,
            "delta": current_value - float(baseline_value),
        }

    map50_delta = comparison.get("mAP50", {}).get("delta")
    map50_95_delta = comparison.get("mAP50_95", {}).get("delta")
    comparison["meaningful_mAP50_improvement"] = (
        isinstance(map50_delta, (int, float))
        and map50_delta >= DEFAULT_THRESHOLDS["meaningful_mAP50_delta"]
    )
    comparison["meaningful_mAP50_95_improvement"] = (
        isinstance(map50_95_delta, (int, float))
        and map50_95_delta >= DEFAULT_THRESHOLDS["meaningful_mAP50_95_delta"]
    )
    return comparison


def _recommendation_for_status(status: str) -> str:
    if status == "production_ready":
        return "Production-ready status requires registry and audit approval before deployment."
    if status == "model_ready_candidate":
        return "Proceed to test evaluation, class-level review, visual review, and re-audit before any production claim."
    if status == "review_required":
        return "Review metric balance, class-level results, and qualitative outputs before promotion."
    if status == "improved_baseline":
        return "Use as an improved governed baseline; continue training and evaluation before model-ready claims."
    return "Do not mark Detection model-ready; improve training, validation metrics, and supporting evidence."
