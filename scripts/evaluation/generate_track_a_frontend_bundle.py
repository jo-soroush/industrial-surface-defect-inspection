"""Generate Track A frontend/demo JSON artifacts from governed evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "frontend" / "track_a"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Track A frontend/demo JSON artifacts."
    )
    parser.add_argument(
        "--comparison",
        required=True,
        help="Path to the governed Track A comparison JSON artifact.",
    )
    parser.add_argument(
        "--validation-evaluation",
        required=True,
        help="Path to the governed ResNet18 validation evaluation JSON artifact.",
    )
    parser.add_argument(
        "--prediction-level-analysis",
        required=True,
        help="Path to the governed ResNet18 prediction-level analysis JSON artifact.",
    )
    parser.add_argument(
        "--threshold-analysis",
        required=True,
        help="Path to the governed ResNet18 threshold analysis JSON artifact.",
    )
    parser.add_argument(
        "--quality-decision",
        required=True,
        help="Path to the governed ResNet18 quality decision JSON artifact.",
    )
    parser.add_argument(
        "--metadata-summary",
        required=True,
        help="Path to the governed ResNet18 metadata summary JSON artifact.",
    )
    parser.add_argument(
        "--artifact-inventory",
        required=True,
        help="Path to the governed ResNet18 artifact inventory JSON artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the frontend bundle JSON artifacts will be written.",
    )
    parser.add_argument(
        "--samples-per-error-type",
        type=int,
        default=8,
        help="How many gallery samples to keep per prediction error type.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples_per_error_type < 1:
        raise ValueError("--samples-per-error-type must be >= 1.")

    comparison_path = Path(args.comparison)
    validation_path = Path(args.validation_evaluation)
    prediction_path = Path(args.prediction_level_analysis)
    threshold_path = Path(args.threshold_analysis)
    quality_path = Path(args.quality_decision)
    metadata_path = Path(args.metadata_summary)
    inventory_path = Path(args.artifact_inventory)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = _load_json_file(comparison_path, "comparison")
    validation = _load_json_file(validation_path, "validation evaluation")
    prediction = _load_json_file(prediction_path, "prediction-level analysis")
    threshold = _load_json_file(threshold_path, "threshold analysis")
    quality = _load_json_file(quality_path, "quality decision")
    metadata = _load_json_file(metadata_path, "metadata summary")
    inventory = _load_json_file(inventory_path, "artifact inventory")

    _validate_comparison(comparison)
    selected = _validate_selected_candidate(comparison)
    _validate_resnet18_artifacts(
        validation=validation,
        prediction=prediction,
        threshold=threshold,
        quality=quality,
        metadata=metadata,
        inventory=inventory,
        selected_run_id=selected["run_id"],
        selected_model_type=selected["model_type"],
        selected_model_version=selected["model_version"],
    )

    comparison_rows = _build_comparison_table(comparison)
    metric_cards = _build_metric_cards(
        selected=selected,
        validation=validation,
        threshold=threshold,
        quality=quality,
        validation_path=validation_path,
        threshold_path=threshold_path,
        quality_path=quality_path,
    )
    confusion_data = _build_confusion_matrix_chart_data(validation, prediction_path)
    threshold_data = _build_threshold_curve_chart_data(threshold)
    per_class_data = _build_per_class_bar_chart_data(validation)
    error_distribution = _build_error_distribution_pie_data(
        validation,
        prediction,
        prediction_path,
    )
    sample_gallery = _build_sample_predictions_gallery(
        prediction=prediction,
        prediction_path=prediction_path,
        samples_per_error_type=args.samples_per_error_type,
    )
    quality_summary = _build_quality_decision_summary(
        quality=quality,
        source_paths={
            "quality_decision": str(quality_path),
            "validation_evaluation": str(validation_path),
            "prediction_level_analysis": str(prediction_path),
            "threshold_analysis": str(threshold_path),
            "comparison": str(comparison_path),
            "metadata_summary": str(metadata_path),
            "artifact_inventory": str(inventory_path),
        },
    )
    recommendation = _build_frontend_model_recommendation(
        comparison=comparison,
        quality=quality,
        threshold=threshold,
        quality_path=quality_path,
        comparison_path=comparison_path,
        threshold_path=threshold_path,
    )

    final_generated_paths = [
        output_dir / "model_comparison_table.json",
        output_dir / "metric_cards.json",
        output_dir / "confusion_matrix_chart_data.json",
        output_dir / "threshold_curve_chart_data.json",
        output_dir / "per_class_bar_chart_data.json",
        output_dir / "error_distribution_pie_data.json",
        output_dir / "sample_predictions_gallery.json",
        output_dir / "quality_decision_summary.json",
        output_dir / "frontend_model_recommendation.json",
        output_dir / "artifact_inventory_frontend.json",
    ]

    generated_files: list[Path] = []
    generated_files.append(
        _write_json(
            final_generated_paths[0],
            comparison_rows,
        )
    )
    generated_files.append(
        _write_json(final_generated_paths[1], metric_cards)
    )
    generated_files.append(
        _write_json(
            final_generated_paths[2],
            confusion_data,
        )
    )
    generated_files.append(
        _write_json(
            final_generated_paths[3],
            threshold_data,
        )
    )
    generated_files.append(
        _write_json(
            final_generated_paths[4],
            per_class_data,
        )
    )
    generated_files.append(
        _write_json(
            final_generated_paths[5],
            error_distribution,
        )
    )
    generated_files.append(
        _write_json(
            final_generated_paths[6],
            sample_gallery,
        )
    )
    generated_files.append(
        _write_json(
            final_generated_paths[7],
            quality_summary,
        )
    )
    generated_files.append(
        _write_json(
            final_generated_paths[8],
            recommendation,
        )
    )

    bundle_inventory = _build_frontend_inventory(
        generated_files=final_generated_paths,
        source_paths={
            "comparison": comparison_path,
            "validation_evaluation": validation_path,
            "prediction_level_analysis": prediction_path,
            "threshold_analysis": threshold_path,
            "quality_decision": quality_path,
            "metadata_summary": metadata_path,
            "artifact_inventory": inventory_path,
        },
        selected=selected,
        inventory_path=final_generated_paths[9],
    )
    generated_files.append(
        _write_json(
            final_generated_paths[9],
            bundle_inventory,
        )
    )

    print(f"frontend_bundle_dir={output_dir}")
    for path in generated_files:
        print(f"generated={path}")
    print(f"selected_model_run_id={selected['run_id']}")
    print(f"selected_model_type={selected['model_type']}")
    print(f"selected_model_version={selected['model_version']}")
    return 0


def _validate_comparison(comparison: dict[str, Any]) -> None:
    if comparison.get("artifact_type") != "track_a_comparison":
        raise ValueError("comparison artifact_type must be track_a_comparison.")
    candidates = comparison.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 3:
        raise ValueError("comparison must contain at least three candidates.")


def _validate_selected_candidate(comparison: dict[str, Any]) -> dict[str, Any]:
    selected_run_id = comparison.get("selected_model_run_id")
    selected_model_type = comparison.get("selected_model_type")
    selected_model_version = comparison.get("selected_model_version")
    if not all(
        isinstance(value, str) and value
        for value in (selected_run_id, selected_model_type, selected_model_version)
    ):
        raise ValueError("comparison selected model fields are incomplete.")
    selected = {
        "run_id": selected_run_id,
        "model_type": selected_model_type,
        "model_version": selected_model_version,
        "model_name": comparison.get("selected_model_name"),
        "quality_status": comparison.get("selected_model_quality_status"),
        "quality_target_status": comparison.get("selected_model_quality_target_status"),
        "production_ready": comparison.get("selected_model_production_ready"),
        "deployment_candidate": comparison.get("selected_model_deployment_candidate"),
        "recommendation_status": comparison.get("selected_model_recommendation_status"),
        "recommended_threshold": comparison.get("selected_model_recommended_threshold"),
        "selection_reason": comparison.get("selection_reason"),
        "comparison_note": comparison.get("comparison_note"),
        "safe_wording": comparison.get("safe_wording"),
        "forbidden_wording": comparison.get("forbidden_wording"),
        "limitations": comparison.get("limitations"),
        "next_recommended_step": comparison.get("next_recommended_step"),
        "source_artifact_paths": comparison.get("source_artifact_paths"),
    }
    if selected["quality_status"] != "TRACK_A_STRONG_CANDIDATE":
        raise ValueError("Selected Track A model must be TRACK_A_STRONG_CANDIDATE.")
    if selected["production_ready"] is not False:
        raise ValueError("Selected Track A model must not be production ready.")
    if selected["deployment_candidate"] is not False:
        raise ValueError("Selected Track A model must not be a deployment candidate.")
    if selected["recommendation_status"] != "selected":
        raise ValueError("Selected Track A model must have recommendation_status=selected.")
    return selected


def _validate_resnet18_artifacts(
    *,
    validation: dict[str, Any],
    prediction: dict[str, Any],
    threshold: dict[str, Any],
    quality: dict[str, Any],
    metadata: dict[str, Any],
    inventory: dict[str, Any],
    selected_run_id: str,
    selected_model_type: str,
    selected_model_version: str,
) -> None:
    if validation.get("artifact_type") != "classification_validation_evaluation":
        raise ValueError("validation evaluation artifact_type mismatch.")
    if validation.get("run_id") != selected_run_id:
        raise ValueError("validation evaluation run_id mismatch.")

    if prediction.get("artifact_type") != "track_a_prediction_level_error_analysis":
        raise ValueError("prediction-level analysis artifact_type mismatch.")
    if prediction.get("run_id") != selected_run_id:
        raise ValueError("prediction-level analysis run_id mismatch.")
    if prediction.get("model_type") != selected_model_type:
        raise ValueError("prediction-level analysis model_type mismatch.")
    if prediction.get("model_version") != selected_model_version:
        raise ValueError("prediction-level analysis model_version mismatch.")

    if threshold.get("artifact_type") != "track_a_threshold_analysis":
        raise ValueError("threshold analysis artifact_type mismatch.")
    if threshold.get("run_id") != selected_run_id:
        raise ValueError("threshold analysis run_id mismatch.")
    if threshold.get("model_type") != selected_model_type:
        raise ValueError("threshold analysis model_type mismatch.")
    if threshold.get("model_version") != selected_model_version:
        raise ValueError("threshold analysis model_version mismatch.")

    if quality.get("artifact_type") != "track_a_resnet18_quality_decision":
        raise ValueError("quality decision artifact_type mismatch.")
    if quality.get("run_id") != selected_run_id:
        raise ValueError("quality decision run_id mismatch.")
    if quality.get("model_type") != selected_model_type:
        raise ValueError("quality decision model_type mismatch.")
    if quality.get("model_version") != selected_model_version:
        raise ValueError("quality decision model_version mismatch.")
    if quality.get("production_ready") is not False:
        raise ValueError("quality decision must not claim production_ready=true.")
    if quality.get("deployment_candidate") is not False:
        raise ValueError("quality decision must not claim deployment_candidate=true.")

    if metadata.get("artifact_type") != "track_a_resnet18_metadata_summary":
        raise ValueError("metadata summary artifact_type mismatch.")
    if metadata.get("run_id") != selected_run_id:
        raise ValueError("metadata summary run_id mismatch.")
    if metadata.get("model_type") != selected_model_type:
        raise ValueError("metadata summary model_type mismatch.")
    if metadata.get("model_version") != selected_model_version:
        raise ValueError("metadata summary model_version mismatch.")

    if inventory.get("artifact_type") != "track_a_resnet18_artifact_inventory":
        raise ValueError("artifact inventory artifact_type mismatch.")
    if inventory.get("run_id") != selected_run_id:
        raise ValueError("artifact inventory run_id mismatch.")
    if inventory.get("model_type") != selected_model_type:
        raise ValueError("artifact inventory model_type mismatch.")
    if inventory.get("model_version") != selected_model_version:
        raise ValueError("artifact inventory model_version mismatch.")

    if validation.get("total_samples") != 803:
        raise ValueError("validation evaluation total_samples must be 803.")
    if validation.get("confusion_matrix") != [[589, 24], [47, 143]]:
        raise ValueError("validation evaluation confusion matrix mismatch.")

    if threshold.get("baseline_threshold") != 0.5:
        raise ValueError("threshold baseline must be 0.5.")
    if threshold.get("recommended_threshold") != 0.65:
        raise ValueError("threshold recommended threshold must be 0.65.")


def _build_comparison_table(comparison: dict[str, Any]) -> dict[str, Any]:
    candidates = comparison.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("comparison candidates must be a list.")
    selected_run_id = comparison.get("selected_model_run_id")
    columns = [
        "rank",
        "selected",
        "model_name",
        "model_type",
        "model_version",
        "run_id",
        "model_quality_status",
        "quality_target_status",
        "production_ready",
        "deployment_candidate",
        "recommendation_status",
        "threshold_used",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "accuracy",
        "false_positive",
        "false_negative",
        "true_positive",
        "true_negative",
        "short_status",
        "source_artifact_paths",
    ]

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.get("macro_f1", 0.0)),
            -float(candidate.get("defect_recall", 0.0)),
            int(candidate.get("false_positives", 0)),
        ),
    )
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(sorted_candidates, start=1):
        rows.append(
            {
                "rank": rank,
                "selected": candidate.get("run_id") == selected_run_id,
                "model_name": candidate.get("model_name"),
                "model_type": candidate.get("model_type"),
                "model_version": _model_version_for_candidate(candidate),
                "run_id": candidate.get("run_id"),
                "model_quality_status": candidate.get("model_quality_status"),
                "quality_target_status": candidate.get("quality_target_status"),
                "production_ready": candidate.get("production_ready"),
                "deployment_candidate": candidate.get("deployment_candidate"),
                "recommendation_status": candidate.get("recommendation_status"),
                "threshold_used": 0.5,
                "precision": candidate.get("defect_precision"),
                "recall": candidate.get("defect_recall"),
                "f1": candidate.get("val_f1"),
                "macro_f1": candidate.get("macro_f1"),
                "accuracy": candidate.get("val_accuracy"),
                "false_positive": candidate.get("false_positives"),
                "false_negative": candidate.get("false_negatives"),
                "true_positive": candidate.get("true_positives"),
                "true_negative": candidate.get("true_negatives"),
                "short_status": _short_status(candidate),
                "source_artifact_paths": candidate.get("source_artifact_paths", {}),
            }
        )

    return {
        "artifact_type": "track_a_model_comparison_table",
        "track_id": comparison.get("track_id"),
        "task_type": comparison.get("task_type"),
        "dataset_id": comparison.get("dataset_id"),
        "split": comparison.get("split"),
        "comparison_id": comparison.get("comparison_id"),
        "selection_metric": comparison.get("selection_metric"),
        "decision_policy": comparison.get("decision_policy"),
        "selected_model_run_id": selected_run_id,
        "selected_model_type": comparison.get("selected_model_type"),
        "selected_model_name": comparison.get("selected_model_name"),
        "selected_model_version": comparison.get("selected_model_version"),
        "selected_model_quality_status": comparison.get(
            "selected_model_quality_status"
        ),
        "selected_model_quality_target_status": comparison.get(
            "selected_model_quality_target_status"
        ),
        "selected_model_production_ready": comparison.get(
            "selected_model_production_ready"
        ),
        "selected_model_deployment_candidate": comparison.get(
            "selected_model_deployment_candidate"
        ),
        "selected_model_recommended_threshold": comparison.get(
            "selected_model_recommended_threshold"
        ),
        "rows": rows,
        "columns": columns,
        "selected_model_reason": comparison.get("selection_reason"),
        "comparison_note": comparison.get("comparison_note"),
        "safe_wording": comparison.get("safe_wording"),
        "forbidden_wording": comparison.get("forbidden_wording"),
        "limitations": comparison.get("limitations"),
        "next_recommended_step": comparison.get("next_recommended_step"),
        "source_artifact_paths": comparison.get("source_artifact_paths", {}),
        "generated_at": comparison.get("generated_at") or comparison.get("created_at"),
        "created_at": _utc_now_iso(),
    }


def _build_metric_cards(
    *,
    selected: dict[str, Any],
    validation: dict[str, Any],
    threshold: dict[str, Any],
    quality: dict[str, Any],
    validation_path: Path,
    threshold_path: Path,
    quality_path: Path,
) -> dict[str, Any]:
    cards = [
        {
            "title": "Selected model",
            "value": f"{selected['model_name']} v{selected['model_version']}",
            "detail": f"Run {selected['run_id']}",
        },
        {
            "title": "Model quality status",
            "value": quality.get("model_quality_status"),
            "detail": quality.get("recommendation_status"),
        },
        {
            "title": "Production readiness",
            "value": "false",
            "detail": "Selected Track A candidate, not production-ready.",
        },
        {
            "title": "Recommended threshold",
            "value": f"{threshold.get('recommended_threshold')}",
            "detail": threshold.get("recommendation_reason"),
        },
        {
            "title": "Precision",
            "value": validation["per_class"]["class_1"]["precision"],
            "detail": "Defect class precision at baseline threshold 0.5.",
        },
        {
            "title": "Recall",
            "value": validation["per_class"]["class_1"]["recall"],
            "detail": "Defect class recall at baseline threshold 0.5.",
        },
        {
            "title": "F1",
            "value": validation["per_class"]["class_1"]["f1"],
            "detail": "Defect class F1 at baseline threshold 0.5.",
        },
        {
            "title": "Macro F1",
            "value": validation["macro_metrics"]["f1"],
            "detail": "Overall validation macro F1.",
        },
        {
            "title": "False positives",
            "value": validation["confusion_matrix"][0][1],
            "detail": "Good parts flagged as defect at baseline threshold 0.5.",
        },
        {
            "title": "False negatives",
            "value": validation["confusion_matrix"][1][0],
            "detail": "Missed defects at baseline threshold 0.5.",
        },
        {
            "title": "Validation samples",
            "value": validation["total_samples"],
            "detail": "Governed validation split sample count.",
        },
    ]

    return {
        "artifact_type": "track_a_metric_cards",
        "track_id": "track_a",
        "task_type": "classification",
        "dataset_id": validation.get("dataset_id"),
        "split": validation.get("split"),
        "selected_model_run_id": selected["run_id"],
        "selected_model_type": selected["model_type"],
        "selected_model_name": selected["model_name"],
        "selected_model_version": selected["model_version"],
        "selected_model_quality_status": selected["quality_status"],
        "quality_target_status": selected["quality_target_status"],
        "production_ready": False,
        "deployment_candidate": False,
        "recommended_threshold": threshold.get("recommended_threshold"),
        "validation_samples": validation["total_samples"],
        "baseline_threshold": threshold.get("baseline_threshold"),
        "cards": cards,
        "safe_interpretation": (
            "ResNet18 v0.4.0 is the strongest governed Track A candidate, but it is "
            "not marked production-ready in this package."
        ),
        "source_artifact_paths": {
            "validation_evaluation": str(validation_path),
            "threshold_analysis": str(threshold_path),
            "quality_decision": str(quality_path),
        },
        "created_at": _utc_now_iso(),
    }


def _build_confusion_matrix_chart_data(
    validation: dict[str, Any],
    validation_path: Path,
) -> dict[str, Any]:
    matrix = validation["confusion_matrix"]
    total = validation["total_samples"]
    normalized = [
        [round(value / total, 6) for value in row]
        for row in matrix
    ]
    return {
        "artifact_type": "track_a_confusion_matrix_chart_data",
        "track_id": "track_a",
        "task_type": "classification",
        "dataset_id": validation.get("dataset_id"),
        "split": validation.get("split"),
        "run_id": validation.get("run_id"),
        "matrix": matrix,
        "labels": {
            "rows": ["good", "defect"],
            "columns": ["good", "defect"],
        },
        "normalized_matrix": normalized,
        "chart_title": "ResNet18 v0.4.0 validation confusion matrix",
        "chart_explanation": (
            "The selected Track A model detects both good parts and defects while "
            "keeping false alarms low."
        ),
        "source_artifact_path": str(validation_path),
        "generated_at": _utc_now_iso(),
    }


def _build_threshold_curve_chart_data(threshold: dict[str, Any]) -> dict[str, Any]:
    rows = threshold.get("per_threshold")
    if not isinstance(rows, list) or not rows:
        raise ValueError("threshold analysis per_threshold rows must be a non-empty list.")
    thresholds = [row["threshold"] for row in rows]
    metrics = {
        "precision": [row["precision"] for row in rows],
        "recall": [row["recall"] for row in rows],
        "f1": [row["f1"] for row in rows],
        "macro_f1": [row["macro_f1"] for row in rows],
        "false_positive": [row["false_positive"] for row in rows],
        "false_negative": [row["false_negative"] for row in rows],
        "accuracy": [row["accuracy"] for row in rows],
    }
    return {
        "artifact_type": "track_a_threshold_curve_chart_data",
        "track_id": "track_a",
        "task_type": "classification",
        "dataset_id": threshold.get("dataset_id"),
        "split": threshold.get("split"),
        "run_id": threshold.get("run_id"),
        "source_threshold_analysis_path": threshold.get(
            "source_prediction_level_analysis_path"
        ),
        "baseline_threshold": threshold.get("baseline_threshold"),
        "recommended_threshold": threshold.get("recommended_threshold"),
        "baseline_metrics": threshold.get("baseline_metrics"),
        "recommended_threshold_metrics": threshold.get("recommended_threshold_metrics"),
        "thresholds": thresholds,
        "series": metrics,
        "rows": rows,
        "markers": [
            {
                "threshold": threshold.get("baseline_threshold"),
                "label": "baseline",
                "kind": "baseline",
            },
            {
                "threshold": threshold.get("recommended_threshold"),
                "label": "recommended",
                "kind": "recommended",
            },
        ],
        "chart_title": "ResNet18 threshold sweep",
        "chart_explanation": (
            "Threshold 0.65 is the recommended operating point because it improves "
            "macro F1 over the baseline threshold 0.5 while keeping false positives low."
        ),
        "generated_at": _utc_now_iso(),
    }


def _build_per_class_bar_chart_data(validation: dict[str, Any]) -> dict[str, Any]:
    per_class = validation["per_class"]
    classes = [
        {
            "label": "good",
            "class_id": 0,
            "precision": per_class["class_0"]["precision"],
            "recall": per_class["class_0"]["recall"],
            "f1": per_class["class_0"]["f1"],
        },
        {
            "label": "defect",
            "class_id": 1,
            "precision": per_class["class_1"]["precision"],
            "recall": per_class["class_1"]["recall"],
            "f1": per_class["class_1"]["f1"],
        },
    ]
    return {
        "artifact_type": "track_a_per_class_bar_chart_data",
        "track_id": "track_a",
        "task_type": "classification",
        "dataset_id": validation.get("dataset_id"),
        "split": validation.get("split"),
        "run_id": validation.get("run_id"),
        "class_labels": ["good", "defect"],
        "series": [
            {
                "metric": "precision",
                "values": [row["precision"] for row in classes],
            },
            {
                "metric": "recall",
                "values": [row["recall"] for row in classes],
            },
            {
                "metric": "f1",
                "values": [row["f1"] for row in classes],
            },
        ],
        "classes": classes,
        "chart_title": "ResNet18 per-class validation metrics",
        "chart_explanation": (
            "The selected ResNet18 candidate performs strongly on both the good and "
            "defect classes, with defect recall above the quality target."
        ),
        "source_artifact_path": validation.get("source_artifact_path"),
        "generated_at": _utc_now_iso(),
    }


def _build_error_distribution_pie_data(
    validation: dict[str, Any],
    prediction: dict[str, Any],
    prediction_path: Path,
) -> dict[str, Any]:
    summary = prediction["summary"]
    total = validation["total_samples"]
    counts = {
        "true_negative": int(summary["true_negative"]),
        "false_positive": int(summary["false_positive"]),
        "false_negative": int(summary["false_negative"]),
        "true_positive": int(summary["true_positive"]),
    }
    segments = [
        {
            "label": "true_negative",
            "count": counts["true_negative"],
            "percentage": round(counts["true_negative"] / total, 6),
        },
        {
            "label": "false_positive",
            "count": counts["false_positive"],
            "percentage": round(counts["false_positive"] / total, 6),
        },
        {
            "label": "false_negative",
            "count": counts["false_negative"],
            "percentage": round(counts["false_negative"] / total, 6),
        },
        {
            "label": "true_positive",
            "count": counts["true_positive"],
            "percentage": round(counts["true_positive"] / total, 6),
        },
    ]
    return {
        "artifact_type": "track_a_error_distribution_pie_data",
        "track_id": "track_a",
        "task_type": "classification",
        "dataset_id": validation.get("dataset_id"),
        "split": validation.get("split"),
        "run_id": validation.get("run_id"),
        "total_samples": total,
        "counts": counts,
        "segments": segments,
        "chart_title": "ResNet18 validation error distribution",
        "chart_explanation": (
            "False positives remain low and false negatives are well below the "
            "Track A acceptance ceiling."
        ),
        "source_artifact_path": str(prediction_path),
        "generated_at": _utc_now_iso(),
    }


def _build_sample_predictions_gallery(
    *,
    prediction: dict[str, Any],
    prediction_path: Path,
    samples_per_error_type: int,
) -> dict[str, Any]:
    records = prediction.get("records")
    if not isinstance(records, list):
        raise ValueError("prediction-level analysis records must be a list.")

    grouped: dict[str, list[dict[str, Any]]] = {
        "true_negative": [],
        "true_positive": [],
        "false_positive": [],
        "false_negative": [],
    }
    for record in records:
        error_type = record.get("error_type")
        if error_type in grouped:
            grouped[error_type].append(record)

    selected_records: list[dict[str, Any]] = []
    selection_order = ["true_positive", "true_negative", "false_positive", "false_negative"]
    for error_type in selection_order:
        pool = sorted(
            grouped[error_type],
            key=lambda item: (
                -float(item.get("confidence", 0.0)),
                int(item.get("sample_index", 0)),
            ),
        )
        selected_records.extend(pool[:samples_per_error_type])

    samples = [_gallery_sample(record) for record in selected_records]
    counts_by_error_type = {
        error_type: len(grouped[error_type]) for error_type in selection_order
    }
    gallery_counts = {
        error_type: min(len(grouped[error_type]), samples_per_error_type)
        for error_type in selection_order
    }
    return {
        "artifact_type": "track_a_sample_predictions_gallery",
        "track_id": "track_a",
        "task_type": "classification",
        "dataset_id": prediction.get("dataset_id"),
        "split": prediction.get("split"),
        "run_id": prediction.get("run_id"),
        "model_type": prediction.get("model_type"),
        "model_version": prediction.get("model_version"),
        "total_samples": prediction.get("total_samples"),
        "gallery_sample_count": len(samples),
        "samples_per_error_type": samples_per_error_type,
        "counts_by_error_type": counts_by_error_type,
        "selected_counts_by_error_type": gallery_counts,
        "samples": samples,
        "gallery_explanation": (
            "Balanced gallery showing correct predictions and the main error types "
            "from the governed validation split."
        ),
        "source_artifact_path": str(prediction_path),
        "generated_at": _utc_now_iso(),
    }


def _gallery_sample(record: dict[str, Any]) -> dict[str, Any]:
    error_type = _require_string(record.get("error_type"), "record.error_type")
    explanations = {
        "true_positive": "Correctly detected defect.",
        "true_negative": "Correctly identified a good sample.",
        "false_positive": "False alarm: good sample predicted as defect.",
        "false_negative": "Missed defect: defect predicted as good.",
    }
    return {
        "sample_index": record.get("sample_index"),
        "image_path": record.get("image_path"),
        "true_label": record.get("true_label"),
        "true_label_name": record.get("true_label_name"),
        "predicted_label": record.get("predicted_label"),
        "predicted_label_name": record.get("predicted_label_name"),
        "probability_good": record.get("probability_good"),
        "probability_defect": record.get("probability_defect"),
        "confidence": record.get("confidence"),
        "error_type": error_type,
        "run_id": record.get("run_id"),
        "model_type": record.get("model_type"),
        "model_version": record.get("model_version"),
        "explanation": explanations.get(error_type, "Representative prediction."),
    }


def _build_quality_decision_summary(
    *,
    quality: dict[str, Any],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "artifact_type": quality.get("artifact_type"),
        "run_id": quality.get("run_id"),
        "run_config_id": quality.get("run_config_id"),
        "model_name": quality.get("model_name"),
        "model_type": quality.get("model_type"),
        "model_version": quality.get("model_version"),
        "dataset_id": quality.get("dataset_id"),
        "dataset_version": quality.get("dataset_version"),
        "split": quality.get("split"),
        "total_validation_samples": quality.get("total_validation_samples"),
        "decision": quality.get("decision"),
        "model_quality_status": quality.get("model_quality_status"),
        "quality_target_status": quality.get("quality_target_status"),
        "production_ready": quality.get("production_ready"),
        "deployment_candidate": quality.get("deployment_candidate"),
        "recommendation_status": quality.get("recommendation_status"),
        "recommended_threshold": quality.get("recommended_threshold"),
        "safe_wording": quality.get("safe_wording"),
        "forbidden_wording": quality.get("forbidden_wording"),
        "limitations": quality.get("limitations"),
        "next_recommended_step": quality.get("next_recommended_step"),
        "source_artifact_paths": source_paths,
        "generated_at": _utc_now_iso(),
    }


def _build_frontend_model_recommendation(
    *,
    comparison: dict[str, Any],
    quality: dict[str, Any],
    threshold: dict[str, Any],
    quality_path: Path,
    comparison_path: Path,
    threshold_path: Path,
) -> dict[str, Any]:
    selected = _validate_selected_candidate(comparison)
    candidates = comparison.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("comparison candidates must be a list.")

    candidate_lookup = {
        candidate.get("model_type"): candidate for candidate in candidates if isinstance(candidate, dict)
    }
    mlp = candidate_lookup.get("mlp")
    cnn = candidate_lookup.get("cnn")
    resnet = candidate_lookup.get("resnet18")
    if mlp is None or cnn is None or resnet is None:
        raise ValueError("comparison must include MLP, CNN, and ResNet18 candidates.")

    return {
        "artifact_type": "track_a_frontend_model_recommendation",
        "track_id": comparison.get("track_id"),
        "task_type": comparison.get("task_type"),
        "dataset_id": comparison.get("dataset_id"),
        "split": comparison.get("split"),
        "selected_model_name": selected["model_name"],
        "selected_model_type": selected["model_type"],
        "selected_model_version": selected["model_version"],
        "selected_run_id": selected["run_id"],
        "selected_threshold": threshold.get("recommended_threshold"),
        "selection_metric": comparison.get("selection_metric"),
        "why_selected": comparison.get("selection_reason"),
        "why_mlp_not_selected": mlp.get("recommendation_reason_short"),
        "why_cnn_not_selected": cnn.get("recommendation_reason_short"),
        "production_ready": False,
        "deployment_candidate": False,
        "safe_demo_wording": (
            "ResNet18 v0.4.0 is the current strongest governed Track A candidate "
            "and the selected model for reporting, while still being marked not "
            "production-ready."
        ),
        "next_step": quality.get("next_recommended_step"),
        "comparison_note": comparison.get("comparison_note"),
        "source_artifact_paths": {
            "comparison": str(comparison_path),
            "quality_decision": str(quality_path),
            "threshold_analysis": str(threshold_path),
        },
        "created_at": _utc_now_iso(),
    }


def _build_frontend_inventory(
    *,
    generated_files: list[Path],
    source_paths: dict[str, Path],
    selected: dict[str, Any],
    inventory_path: Path,
) -> dict[str, Any]:
    bundle_files = []
    for path in generated_files:
        if path == inventory_path:
            bundle_files.append(
                {
                    "path": str(path),
                    "artifact_type": "track_a_frontend_artifact_inventory",
                    "exists": True,
                    "included_in_hash_inventory": False,
                    "self_referential": True,
                }
            )
            continue
        payload = _load_json_file(path, path.name.replace(".json", ""))
        bundle_files.append(
            {
                "path": str(path),
                "artifact_type": payload.get("artifact_type"),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    source_files = []
    for name, path in source_paths.items():
        source_payload = _load_json_file(path, name)
        source_files.append(
            {
                "name": name,
                "path": str(path),
                "artifact_type": source_payload.get("artifact_type"),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    return {
        "artifact_type": "track_a_frontend_artifact_inventory",
        "track_id": "track_a",
        "task_type": "classification",
        "dataset_id": "mvtec_classification_supervised",
        "split": "validation",
        "selected_model_run_id": selected["run_id"],
        "selected_model_type": selected["model_type"],
        "selected_model_version": selected["model_version"],
        "bundle_directory": str(DEFAULT_OUTPUT_DIR),
        "generated_at": _utc_now_iso(),
        "bundle_artifact_count": len(bundle_files),
        "source_artifact_count": len(source_files),
        "generated_file_paths": [str(path) for path in generated_files],
        "source_artifact_paths": [str(path) for path in source_paths.values()],
        "bundle_files": bundle_files,
        "source_files": source_files,
        "safe_demo_wording": (
            "This frontend bundle is a governed presentation layer for Track A only; "
            "it does not imply production readiness."
        ),
    }


def _model_version_for_candidate(candidate: dict[str, Any]) -> str:
    run_id = candidate.get("run_id")
    model_type = candidate.get("model_type")
    if model_type == "mlp":
        return "0.2.0"
    if model_type == "cnn":
        if run_id == "50993bc0-4dcd-48f3-a080-d6cbaf21d804":
            return "0.4.0"
        if run_id == "e170a2a3-52fc-48d4-8e24-e3da35e4ce4d":
            return "0.3.0"
        if run_id == "9837d33d-71ba-4d2d-9aed-ff1f5da6adbc":
            return "0.2.0"
    if model_type == "resnet18":
        return "0.4.0"
    return str(candidate.get("model_version") or "")


def _short_status(candidate: dict[str, Any]) -> str:
    status = candidate.get("model_quality_status")
    if status == "BASELINE_ONLY":
        return "Governed baseline-only reference."
    if status == "FAILED_QUALITY":
        return "Governed failed-quality comparison evidence."
    if status == "TRACK_A_STRONG_CANDIDATE":
        return "Selected governed Track A candidate."
    return "Governed comparison evidence."


def _load_json_file(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} JSON not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact_name} JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON must contain an object.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
