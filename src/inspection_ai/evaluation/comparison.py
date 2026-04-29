"""Governed model comparison boundary for Phase 4 and Phase 5 workflows.

This module defines the source boundary for comparing model candidates using
governed metrics and metadata payloads. It will eventually consume evaluation
outputs and lifecycle metadata to support reviewable model comparison and
selection workflows while remaining separate from training and business
decision logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRACK_A_SUPERVISED_DATASET_ID = "mvtec_classification_supervised"
TRACK_A_MINIMUM_DEFECT_RECALL = 0.50
TRACK_A_MACRO_F1_NEAR_TIE_THRESHOLD = 0.02
TRACK_A_SELECTION_METRIC = "macro_f1"
TRACK_A_RISK_SIGNAL = "defect_recall"


def build_track_a_comparison(
    mlp_training_result_path: str,
    cnn_training_result_path: str,
) -> dict[str, Any]:
    """Build a side-effect-free Track A comparison from explicit artifacts."""
    mlp_result = _load_training_result(mlp_training_result_path)
    cnn_result = _load_training_result(cnn_training_result_path)

    mlp_candidate = _build_track_a_candidate(mlp_result, mlp_training_result_path)
    cnn_candidate = _build_track_a_candidate(cnn_result, cnn_training_result_path)

    dataset_id = mlp_candidate["dataset_id"]
    if dataset_id != cnn_candidate["dataset_id"]:
        raise ValueError("Track A comparison requires matching dataset_id values.")

    candidates = [mlp_candidate, cnn_candidate]
    recommended_candidate = _apply_track_a_recommendation(candidates)

    return {
        "comparison_id": (
            "track_a_supervised_classification__"
            f"{mlp_candidate['run_id']}__{cnn_candidate['run_id']}"
        ),
        "dataset_id": dataset_id,
        "task_type": "classification",
        "selection_metric": TRACK_A_SELECTION_METRIC,
        "decision_policy": _build_track_a_decision_policy(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "recommended_candidate": recommended_candidate,
    }


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


def _load_training_result(path_value: str) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("TrainingResult path must be a non-empty string.")

    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"TrainingResult JSON not found: {path_value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"TrainingResult JSON is invalid: {path_value}") from exc

    if not isinstance(payload, dict):
        raise ValueError("TrainingResult JSON must contain an object.")

    return payload


def _build_track_a_candidate(
    training_result: dict[str, Any],
    training_result_path: str,
) -> dict[str, Any]:
    identity = _require_section(training_result, "identity")
    metrics = _require_section(training_result, "metrics")
    metadata = _require_section(training_result, "metadata")

    dataset_id = metadata.get("dataset_id")
    if dataset_id != TRACK_A_SUPERVISED_DATASET_ID:
        raise ValueError(
            "Track A comparison requires dataset_id "
            f"{TRACK_A_SUPERVISED_DATASET_ID}."
        )

    evaluation = _load_validation_evaluation_artifact(metadata)
    macro_metrics = _require_dict(evaluation.get("macro_metrics"), "macro_metrics")
    confusion_matrix = _validate_confusion_matrix(evaluation)
    true_negatives, false_positives = confusion_matrix[0]
    false_negatives, true_positives = confusion_matrix[1]
    defect_recall = _safe_ratio(true_positives, true_positives + false_negatives)
    defect_precision = _safe_ratio(true_positives, true_positives + false_positives)
    warnings = _build_candidate_warnings(
        defect_recall=defect_recall,
        false_negatives=false_negatives,
        true_positives=true_positives,
    )
    macro_f1 = _require_number(macro_metrics.get("f1"), "macro_f1")

    candidate = {
        "model_name": _require_string(metadata.get("model_name"), "model_name"),
        "model_type": _require_string(identity.get("model_type"), "model_type"),
        "run_config_id": _require_string(
            identity.get("run_config_id"), "run_config_id"
        ),
        "dataset_id": dataset_id,
        "run_id": _require_string(identity.get("run_id"), "run_id"),
        "train_accuracy": _require_number(metrics.get("train_accuracy"), "train_accuracy"),
        "train_f1": _require_number(metrics.get("train_f1"), "train_f1"),
        "val_loss": _require_number(metrics.get("val_loss"), "val_loss"),
        "val_accuracy": _require_number(metrics.get("val_accuracy"), "val_accuracy"),
        "val_f1": _require_number(metrics.get("val_f1"), "val_f1"),
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "defect_recall": defect_recall,
        "defect_precision": defect_precision,
        "warnings": warnings,
        "explanation": _build_candidate_explanation(macro_f1, defect_recall),
        "training_result_path": training_result_path,
        "validation_evaluation_path": _require_string(
            metadata.get("validation_evaluation_path"),
            "validation_evaluation_path",
        ),
    }

    if evaluation.get("run_id") != candidate["run_id"]:
        raise ValueError("Evaluation artifact run_id must match TrainingResult run_id.")
    if evaluation.get("dataset_id") != dataset_id:
        raise ValueError(
            "Evaluation artifact dataset_id must match TrainingResult dataset_id."
        )

    return candidate


def _load_validation_evaluation_artifact(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    artifact_path_value = _require_string(
        metadata.get("validation_evaluation_path"),
        "validation_evaluation_path",
    )
    artifact_path = Path(artifact_path_value)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Validation evaluation artifact not found: {artifact_path_value}"
        )

    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Validation evaluation artifact JSON is invalid: {artifact_path_value}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Validation evaluation artifact must contain an object.")

    return payload


def _validate_confusion_matrix(evaluation: dict[str, Any]) -> list[list[int]]:
    confusion_matrix = evaluation.get("confusion_matrix")
    if not isinstance(confusion_matrix, list) or len(confusion_matrix) != 2:
        raise ValueError("Validation confusion_matrix must be a 2x2 list.")

    validated_matrix = []
    for row in confusion_matrix:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("Validation confusion_matrix must be a 2x2 list.")
        validated_row = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Validation confusion_matrix values must be integers.")
            if value < 0:
                raise ValueError(
                    "Validation confusion_matrix values must be non-negative."
                )
            validated_row.append(value)
        validated_matrix.append(validated_row)

    total_samples = evaluation.get("total_samples")
    if isinstance(total_samples, bool) or not isinstance(total_samples, int):
        raise ValueError("Validation evaluation total_samples must be an integer.")
    if sum(sum(row) for row in validated_matrix) != total_samples:
        raise ValueError("Validation confusion_matrix sum must equal total_samples.")

    return validated_matrix


def _apply_track_a_recommendation(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    first, second = candidates
    first_macro_f1 = first["macro_f1"]
    second_macro_f1 = second["macro_f1"]

    if first_macro_f1 == second_macro_f1:
        reason = "Candidates are tied on macro_f1; manual review is required."
        for candidate in candidates:
            candidate["recommendation_status"] = "review_required"
            candidate["recommendation_reason_short"] = reason
        return {
            "recommendation_status": "review_required",
            "recommendation_reason_short": reason,
            "decision_explanation": reason,
        }

    selected = first if first_macro_f1 > second_macro_f1 else second
    for candidate in candidates:
        if candidate is selected:
            candidate["recommendation_status"] = "selected"
            candidate["recommendation_reason_short"] = (
                "Selected because it has the highest macro_f1."
            )
        else:
            candidate["recommendation_status"] = "not_selected"
            candidate["recommendation_reason_short"] = (
                "Not selected because another candidate has higher macro_f1."
            )

    selected_defect_count = selected["true_positives"] + selected["false_negatives"]
    selected_status = "selected"
    selected_reason = "Selected because it has the highest macro_f1."
    macro_f1_difference = abs(first_macro_f1 - second_macro_f1)
    if (
        macro_f1_difference < TRACK_A_MACRO_F1_NEAR_TIE_THRESHOLD
    ):
        selected_status = "review_required"
        selected_reason = (
            "Selected by macro_f1 but candidates are within the near-tie threshold; "
            "manual review required."
        )
        selected["recommendation_status"] = selected_status
        selected["recommendation_reason_short"] = selected_reason
    elif selected["defect_recall"] < TRACK_A_MINIMUM_DEFECT_RECALL:
        selected_status = "review_required"
        selected_reason = (
            "Selected by macro_f1 but defect_recall is below the policy threshold; "
            "manual review required."
        )
        selected["recommendation_status"] = selected_status
        selected["recommendation_reason_short"] = selected_reason

    return {
        "model_name": selected["model_name"],
        "model_type": selected["model_type"],
        "run_config_id": selected["run_config_id"],
        "dataset_id": selected["dataset_id"],
        "run_id": selected["run_id"],
        "macro_f1": selected["macro_f1"],
        "recommendation_status": selected_status,
        "recommendation_reason_short": selected_reason,
        "decision_explanation": _build_decision_explanation(
            selected=selected,
            other=second if selected is first else first,
            selected_reason=selected_reason,
        ),
    }


def _require_section(payload: dict[str, Any], section_name: str) -> dict[str, Any]:
    value = payload.get(section_name)
    return _require_dict(value, section_name)


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _build_track_a_decision_policy() -> dict[str, Any]:
    return {
        "minimum_defect_recall": TRACK_A_MINIMUM_DEFECT_RECALL,
        "macro_f1_near_tie_threshold": TRACK_A_MACRO_F1_NEAR_TIE_THRESHOLD,
        "selection_metric": TRACK_A_SELECTION_METRIC,
        "risk_signal": TRACK_A_RISK_SIGNAL,
    }


def _build_candidate_warnings(
    defect_recall: float,
    false_negatives: int,
    true_positives: int,
) -> list[str]:
    warnings = []
    if defect_recall == 0.0 and true_positives + false_negatives > 0:
        warnings.append("Model failed to detect any defects (defect_recall = 0.0)")
    if false_negatives > true_positives:
        warnings.append("High number of missed defects (false negatives)")
    return warnings


def _build_candidate_explanation(macro_f1: float, defect_recall: float) -> str:
    return (
        f"Model achieved macro_f1={macro_f1:.2f} with "
        f"defect_recall={defect_recall:.2f}, indicating limited defect detection."
    )


def _build_decision_explanation(
    selected: dict[str, Any],
    other: dict[str, Any],
    selected_reason: str,
) -> str:
    selected_name = str(selected["model_name"]).upper()
    other_name = str(other["model_name"]).upper()
    if selected["recommendation_status"] == "review_required":
        if selected["defect_recall"] < TRACK_A_MINIMUM_DEFECT_RECALL:
            return (
                f"{selected_name} was selected by macro_f1 but requires manual review "
                "because defect recall is below the policy threshold."
            )
        return (
            f"{selected_name} was selected by macro_f1 but requires manual review "
            "because candidates are within the near-tie threshold."
        )

    other_risk_note = ""
    if other["defect_recall"] == 0.0 and (
        other["true_positives"] + other["false_negatives"]
    ) > 0:
        other_risk_note = f" {other_name} rejected due to zero defect detection."

    return (
        f"{selected_name} selected due to higher macro_f1 and "
        f"defect_recall={selected['defect_recall']:.2f}. "
        f"{selected_reason}{other_risk_note}"
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
