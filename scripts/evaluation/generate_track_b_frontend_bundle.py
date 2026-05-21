"""Generate Track B frontend/demo JSON artifacts from governed evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "frontend" / "track_b"
DEFAULT_TRAINING_RESULT_PATH = REPO_ROOT / "artifacts/models/analysis/training_results/training_result__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.json"
DEFAULT_EVALUATION_PATH = REPO_ROOT / "artifacts/models/metrics/anomaly_detection_evaluation__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
DEFAULT_LEARNING_CURVES_PATH = REPO_ROOT / "artifacts/models/metrics/anomaly_learning_curves__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.json"
DEFAULT_CONFUSION_MATRIX_PATH = REPO_ROOT / "artifacts/models/metrics/anomaly_confusion_matrix__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__train_test.json"
DEFAULT_QUALITATIVE_SAMPLES_PATH = REPO_ROOT / "artifacts/models/explainability/b8ca43f5-0d53-4a42-ab37-b5fca9544a36/anomaly_qualitative_samples__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.json"
DEFAULT_EXPLAINABILITY_PATH = REPO_ROOT / "artifacts/models/explainability/b8ca43f5-0d53-4a42-ab37-b5fca9544a36/anomaly_reconstruction_explainability__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.json"
DEFAULT_PRODUCTION_SUMMARY_PATH = REPO_ROOT / "artifacts/models/metadata/track_b_production_canonical_summary__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.json"
DEFAULT_FULL_SUMMARY_PATH = REPO_ROOT / "artifacts/models/metadata/track_b_full_production_canonical_summary__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.json"
DEFAULT_INVENTORY_PATH = REPO_ROOT / "artifacts/models/inventory/track_b_artifact_inventory__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.json"
DEFAULT_PR_CURVE_PATH = REPO_ROOT / "artifacts/models/metrics/anomaly_pr_curve__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
DEFAULT_THRESHOLD_SWEEP_PATH = REPO_ROOT / "artifacts/models/metrics/anomaly_threshold_sweep__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
DEFAULT_SCORE_DISTRIBUTION_PATH = REPO_ROOT / "artifacts/models/metrics/anomaly_score_distribution__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
DEFAULT_SAMPLE_PREDICTIONS_PATH = REPO_ROOT / "artifacts/models/predictions/anomaly_sample_predictions__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
DEFAULT_QUALITY_DECISION_PATH = REPO_ROOT / "artifacts/models/metrics/anomaly_quality_decision__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
DEFAULT_GOVERNED_EVIDENCE_INVENTORY_PATH = REPO_ROOT / "artifacts/models/inventory/anomaly_governed_evidence_inventory__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
TRACK_ID = "track_b"
TASK_TYPE = "anomaly_detection"
DATASET_ID = "mvtec_anomaly"
MODEL_TYPE = "autoencoder"
MODEL_VERSION = "0.1.0"
RUN_ID = "b8ca43f5-0d53-4a42-ab37-b5fca9544a36"
CONFIG_ID = "autoencoder_train_v0_1_0"
CANONICAL_STATUS = "production-canonical"
FRONTEND_EVIDENCE_STATUS = "governed_review_evidence"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Track B frontend/demo JSON artifacts."
    )
    parser.add_argument(
        "--training-result",
        default=str(DEFAULT_TRAINING_RESULT_PATH),
        help="Path to the governed Track B TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--evaluation",
        default=str(DEFAULT_EVALUATION_PATH),
        help="Path to the governed Track B anomaly evaluation JSON artifact.",
    )
    parser.add_argument(
        "--learning-curves",
        default=str(DEFAULT_LEARNING_CURVES_PATH),
        help="Path to the governed Track B learning-curves JSON artifact.",
    )
    parser.add_argument(
        "--confusion-matrix",
        default=str(DEFAULT_CONFUSION_MATRIX_PATH),
        help="Path to the governed Track B confusion-matrix JSON artifact.",
    )
    parser.add_argument(
        "--qualitative-samples",
        default=str(DEFAULT_QUALITATIVE_SAMPLES_PATH),
        help="Path to the governed Track B qualitative-samples JSON artifact.",
    )
    parser.add_argument(
        "--reconstruction-explainability",
        default=str(DEFAULT_EXPLAINABILITY_PATH),
        help="Path to the governed Track B reconstruction-explainability JSON artifact.",
    )
    parser.add_argument(
        "--production-canonical-summary",
        default=str(DEFAULT_PRODUCTION_SUMMARY_PATH),
        help="Path to the governed Track B production-canonical summary JSON artifact.",
    )
    parser.add_argument(
        "--full-production-canonical-summary",
        default=str(DEFAULT_FULL_SUMMARY_PATH),
        help="Path to the governed Track B full production-canonical summary JSON artifact.",
    )
    parser.add_argument(
        "--inventory",
        default=str(DEFAULT_INVENTORY_PATH),
        help="Path to the governed Track B artifact inventory JSON artifact.",
    )
    parser.add_argument("--pr-curve", default=str(DEFAULT_PR_CURVE_PATH), help="Path to governed anomaly PR curve JSON.")
    parser.add_argument("--threshold-sweep", default=str(DEFAULT_THRESHOLD_SWEEP_PATH), help="Path to governed anomaly threshold sweep JSON.")
    parser.add_argument("--score-distribution", default=str(DEFAULT_SCORE_DISTRIBUTION_PATH), help="Path to governed anomaly score distribution JSON.")
    parser.add_argument("--sample-predictions", default=str(DEFAULT_SAMPLE_PREDICTIONS_PATH), help="Path to governed anomaly sample predictions JSON.")
    parser.add_argument("--quality-decision", default=str(DEFAULT_QUALITY_DECISION_PATH), help="Path to governed anomaly quality decision JSON.")
    parser.add_argument("--governed-evidence-inventory", default=str(DEFAULT_GOVERNED_EVIDENCE_INVENTORY_PATH), help="Path to governed anomaly derived-evidence inventory JSON.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the frontend bundle JSON artifacts will be written.",
    )
    parser.add_argument(
        "--samples-per-error-type",
        type=int,
        default=8,
        help="How many gallery samples to keep per confusion-group category.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples_per_error_type < 1:
        raise ValueError("--samples-per-error-type must be >= 1.")

    training_result_path = Path(args.training_result)
    evaluation_path = Path(args.evaluation)
    learning_curves_path = Path(args.learning_curves)
    confusion_path = Path(args.confusion_matrix)
    qualitative_samples_path = Path(args.qualitative_samples)
    explainability_path = Path(args.reconstruction_explainability)
    production_summary_path = Path(args.production_canonical_summary)
    full_summary_path = Path(args.full_production_canonical_summary)
    inventory_path = Path(args.inventory)
    pr_curve_path = Path(args.pr_curve)
    threshold_sweep_path = Path(args.threshold_sweep)
    score_distribution_path = Path(args.score_distribution)
    sample_predictions_path = Path(args.sample_predictions)
    quality_decision_path = Path(args.quality_decision)
    governed_evidence_inventory_path = Path(args.governed_evidence_inventory)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_result = _load_json_file(training_result_path, "training result")
    evaluation = _load_json_file(evaluation_path, "anomaly evaluation")
    learning_curves = _load_json_file(learning_curves_path, "learning curves")
    confusion = _load_json_file(confusion_path, "confusion matrix")
    qualitative_samples = _load_json_file(qualitative_samples_path, "qualitative samples")
    explainability = _load_json_file(explainability_path, "reconstruction explainability")
    production_summary = _load_json_file(
        production_summary_path, "production canonical summary"
    )
    full_summary = _load_json_file(
        full_summary_path, "full production canonical summary"
    )
    inventory = _load_json_file(inventory_path, "artifact inventory")
    pr_curve = _load_json_file(pr_curve_path, "anomaly PR curve")
    threshold_sweep = _load_json_file(threshold_sweep_path, "anomaly threshold sweep")
    score_distribution = _load_json_file(score_distribution_path, "anomaly score distribution")
    sample_predictions = _load_json_file(sample_predictions_path, "anomaly sample predictions")
    quality_decision = _load_json_file(quality_decision_path, "anomaly quality decision")
    governed_evidence_inventory = _load_json_file(
        governed_evidence_inventory_path, "anomaly governed evidence inventory"
    )

    _validate_training_result(training_result)
    _validate_evaluation(evaluation, training_result)
    _validate_learning_curves(learning_curves, training_result)
    _validate_confusion_matrix(confusion, training_result)
    _validate_qualitative_samples(qualitative_samples, training_result)
    _validate_explainability(explainability, training_result)
    _validate_production_summary(production_summary, training_result)
    _validate_full_summary(full_summary, training_result)
    _validate_inventory(inventory, training_result, evaluation)
    _validate_governed_evidence(pr_curve, "anomaly_pr_curve")
    _validate_governed_evidence(threshold_sweep, "anomaly_threshold_sweep")
    _validate_governed_evidence(score_distribution, "anomaly_score_distribution")
    _validate_governed_evidence(sample_predictions, "anomaly_sample_predictions")
    _validate_governed_evidence(quality_decision, "anomaly_quality_decision")
    _validate_governed_evidence_inventory(governed_evidence_inventory)

    source_paths = {
        "training_result": training_result_path,
        "evaluation": evaluation_path,
        "learning_curves": learning_curves_path,
        "confusion_matrix": confusion_path,
        "qualitative_samples": qualitative_samples_path,
        "reconstruction_explainability": explainability_path,
        "production_canonical_summary": production_summary_path,
        "full_production_canonical_summary": full_summary_path,
        "inventory": inventory_path,
        "anomaly_pr_curve": pr_curve_path,
        "anomaly_threshold_sweep": threshold_sweep_path,
        "anomaly_score_distribution": score_distribution_path,
        "anomaly_sample_predictions": sample_predictions_path,
        "anomaly_quality_decision": quality_decision_path,
        "anomaly_governed_evidence_inventory": governed_evidence_inventory_path,
    }

    score_summary = _build_anomaly_score_summary(
        evaluation=evaluation,
        score_distribution=score_distribution,
        source_paths=source_paths,
    )
    reconstruction_summary = _build_reconstruction_loss_summary(
        training_result=training_result,
        learning_curves=learning_curves,
        score_distribution=score_distribution,
        source_paths=source_paths,
    )
    threshold_behavior = _build_threshold_behavior(
        threshold_sweep=threshold_sweep,
        source_paths=source_paths,
    )
    metric_cards = _build_metric_cards(
        training_result=training_result,
        evaluation=evaluation,
        pr_curve=pr_curve,
        quality_decision=quality_decision,
        source_paths=source_paths,
    )
    sample_gallery = _build_sample_anomaly_gallery(
        evaluation=evaluation,
        qualitative_samples=qualitative_samples,
        explainability=explainability,
        samples_per_error_type=args.samples_per_error_type,
        source_paths=source_paths,
    )
    quality_summary = _build_quality_decision_summary(
        quality_decision=quality_decision,
        source_paths=source_paths,
    )
    frontend_summary = _build_frontend_anomaly_summary(
        evaluation=evaluation,
        pr_curve=pr_curve,
        quality_decision=quality_decision,
        source_paths=source_paths,
    )
    sample_prediction_summary = _build_sample_predictions_summary(
        sample_predictions=sample_predictions,
        source_paths=source_paths,
    )

    generated_files = {
        "anomaly_score_summary": output_dir / "anomaly_score_summary.json",
        "reconstruction_loss_summary": output_dir / "reconstruction_loss_summary.json",
        "threshold_behavior": output_dir / "threshold_behavior.json",
        "metric_cards": output_dir / "metric_cards.json",
        "sample_anomaly_gallery": output_dir / "sample_anomaly_gallery.json",
        "sample_predictions": output_dir / "sample_predictions.json",
        "quality_decision_summary": output_dir / "quality_decision_summary.json",
        "frontend_anomaly_summary": output_dir / "frontend_anomaly_summary.json",
    }

    _write_json(generated_files["anomaly_score_summary"], score_summary)
    _write_json(generated_files["reconstruction_loss_summary"], reconstruction_summary)
    _write_json(generated_files["threshold_behavior"], threshold_behavior)
    _write_json(generated_files["metric_cards"], metric_cards)
    _write_json(generated_files["sample_anomaly_gallery"], sample_gallery)
    _write_json(generated_files["sample_predictions"], sample_prediction_summary)
    _write_json(generated_files["quality_decision_summary"], quality_summary)
    _write_json(generated_files["frontend_anomaly_summary"], frontend_summary)

    inventory_frontend = _build_frontend_inventory(
        generated_files=list(generated_files.values()),
        source_paths=source_paths,
        selected_run_id=RUN_ID,
        selected_model_type=MODEL_TYPE,
        selected_model_version=MODEL_VERSION,
        inventory_path=output_dir / "artifact_inventory_frontend.json",
    )
    _write_json(output_dir / "artifact_inventory_frontend.json", inventory_frontend)

    return 0


def _build_anomaly_score_summary(
    *,
    evaluation: dict[str, Any],
    score_distribution: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    metrics = _require_dict(evaluation.get("metrics"), "evaluation.metrics")

    return {
        "artifact_type": "track_b_anomaly_score_summary",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "dataset_id": DATASET_ID,
        "split": "test",
        "run_id": RUN_ID,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "dataset_version": "mvtec_1.0",
        "score_definition": evaluation.get("score_definition"),
        "threshold_strategy": evaluation.get("threshold_strategy"),
        "threshold": score_distribution.get("threshold"),
        "summary": score_distribution.get("summary"),
        "histograms": score_distribution.get("histograms"),
        "anomaly_score_statistics": score_distribution.get("summary"),
        "normal_vs_anomaly_score_separation": {
            "roc_auc": metrics.get("roc_auc"),
            "plain_language": (
                "ROC AUC is below 0.5 on the governed test split, so the anomaly score "
                "does not show strong normal-vs-anomaly separation in this governed run."
            ),
        },
        "source_artifact_path": str(source_paths["anomaly_score_distribution"]),
        "source_artifact_paths": {
            "evaluation": str(source_paths["evaluation"]),
            "anomaly_score_distribution": str(source_paths["anomaly_score_distribution"]),
            "anomaly_governed_evidence_inventory": str(source_paths["anomaly_governed_evidence_inventory"]),
        },
        "plain_language_explanation": (
            "The autoencoder assigns a higher reconstruction-error anomaly score to images "
            "that look less like the training set. This governed summary includes chart-ready "
            "test-set score distributions derived from existing sample-level anomaly scores."
        ),
        "generated_at": _utc_now_iso(),
    }


def _build_reconstruction_loss_summary(
    *,
    training_result: dict[str, Any],
    learning_curves: dict[str, Any],
    score_distribution: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    metrics = _require_dict(training_result.get("metrics"), "training_result.metrics")
    curves = _require_dict(learning_curves.get("curves"), "learning_curves.curves")
    train_loss = curves.get("train_loss") or []
    val_loss = curves.get("val_loss") or []
    learning_rows = []
    for idx, value in enumerate(train_loss, start=1):
        learning_rows.append({"epoch": idx, "train_loss": value})
    for idx, value in enumerate(val_loss, start=1):
        if idx <= len(learning_rows):
            learning_rows[idx - 1]["val_loss"] = value
        else:
            learning_rows.append({"epoch": idx, "val_loss": value})

    return {
        "artifact_type": "track_b_reconstruction_loss_summary",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "dataset_id": DATASET_ID,
        "split": "train",
        "run_id": RUN_ID,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "training_config_id": CONFIG_ID,
        "reconstruction_loss": metrics.get("reconstruction_loss"),
        "final_reconstruction_loss": metrics.get("reconstruction_loss"),
        "learning_curves": curves,
        "chart_rows": learning_rows,
        "sample_level_reconstruction_loss": {
            "mapping": score_distribution.get("reconstruction_loss_mapping"),
            "summary": score_distribution.get("summary"),
            "histograms": score_distribution.get("histograms"),
            "threshold": score_distribution.get("threshold"),
        },
        "source_artifact_path": str(source_paths["anomaly_score_distribution"]),
        "source_artifact_paths": {
            "training_result": str(source_paths["training_result"]),
            "learning_curves": str(source_paths["learning_curves"]),
            "anomaly_score_distribution": str(source_paths["anomaly_score_distribution"]),
        },
        "plain_language_explanation": (
            "Sample-level reconstruction loss is mapped from anomaly_score because the governed "
            "score_definition is mean_squared_reconstruction_error_per_image. This file does not "
            "claim full reconstruction image evidence for every sample."
        ),
        "generated_at": _utc_now_iso(),
    }


def _build_threshold_behavior(
    *,
    threshold_sweep: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "artifact_type": "track_b_threshold_behavior",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "dataset_id": DATASET_ID,
        "split": "test",
        "run_id": RUN_ID,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "threshold_strategy": threshold_sweep.get("selected_threshold_strategy"),
        "baseline_threshold": threshold_sweep.get("selected_threshold"),
        "selected_threshold": threshold_sweep.get("selected_threshold"),
        "rows": threshold_sweep.get("rows"),
        "threshold_sweep": threshold_sweep.get("rows"),
        "selected_threshold_metrics": threshold_sweep.get("selected_threshold_metrics"),
        "threshold_explanation": (
            "The governed selected threshold is shown together with a chart-ready threshold "
            "sweep derived from existing sample-level anomaly scores. No new inference was run."
        ),
        "source_artifact_path": str(source_paths["anomaly_threshold_sweep"]),
        "source_artifact_paths": {
            "evaluation": str(source_paths["evaluation"]),
            "anomaly_threshold_sweep": str(source_paths["anomaly_threshold_sweep"]),
        },
        "generated_at": _utc_now_iso(),
        "sample_count": threshold_sweep.get("sample_count"),
    }


def _build_metric_cards(
    *,
    training_result: dict[str, Any],
    evaluation: dict[str, Any],
    pr_curve: dict[str, Any],
    quality_decision: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    metadata = _require_dict(training_result.get("metadata"), "training_result.metadata")
    metrics = _require_dict(evaluation.get("metrics"), "evaluation.metrics")
    threshold = evaluation.get("threshold")
    pr_auc = pr_curve.get("pr_auc")

    cards = [
        {"title": "Selected model", "value": f"{MODEL_TYPE} v{MODEL_VERSION}", "detail": f"Run {RUN_ID}"},
        {"title": "Evidence status", "value": FRONTEND_EVIDENCE_STATUS, "detail": "Governed anomaly evidence package for review."},
        {"title": "Production readiness", "value": "not claimed", "detail": "This frontend bundle does not claim production or deployment readiness."},
        {"title": "Threshold", "value": threshold, "detail": "Governed percentile-95 threshold from the anomaly evaluation."},
        {"title": "ROC AUC", "value": metrics.get("roc_auc"), "detail": "Measured on the governed test split."},
        {"title": "PR AUC", "value": pr_auc, "detail": "Average precision derived from governed sample-level anomaly scores."},
        {"title": "Precision", "value": metrics.get("precision"), "detail": "Positive-class precision at the canonical threshold."},
        {"title": "Recall", "value": metrics.get("recall"), "detail": "Positive-class recall at the canonical threshold."},
        {"title": "F1", "value": metrics.get("f1"), "detail": "Positive-class F1 at the canonical threshold."},
        {"title": "Train samples", "value": metadata.get("train_sample_count"), "detail": "Governed training split sample count."},
        {"title": "Test samples", "value": metadata.get("test_sample_count"), "detail": "Governed test split sample count."},
    ]
    return {
        "artifact_type": "track_b_metric_cards",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "dataset_id": DATASET_ID,
        "split": "test",
        "run_id": RUN_ID,
        "model_type": MODEL_TYPE,
        "model_name": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "evidence_status": FRONTEND_EVIDENCE_STATUS,
        "quality_status": quality_decision.get("quality_status"),
        "production_ready": False,
        "deployment_safe": False,
        "deployment_candidate": False,
        "threshold": threshold,
        "validation_samples": metadata.get("test_sample_count"),
        "cards": cards,
        "safe_interpretation": (
            "Surface Anomaly Detection evidence is useful as a review-only supporting signal. "
            "It does not claim production readiness or deployment safety."
        ),
        "source_artifact_paths": {
            "training_result": str(source_paths["training_result"]),
            "evaluation": str(source_paths["evaluation"]),
            "learning_curves": str(source_paths["learning_curves"]),
            "confusion_matrix": str(source_paths["confusion_matrix"]),
            "production_canonical_summary": str(source_paths["production_canonical_summary"]),
            "full_production_canonical_summary": str(source_paths["full_production_canonical_summary"]),
            "inventory": str(source_paths["inventory"]),
            "anomaly_pr_curve": str(source_paths["anomaly_pr_curve"]),
            "anomaly_quality_decision": str(source_paths["anomaly_quality_decision"]),
            "anomaly_governed_evidence_inventory": str(source_paths["anomaly_governed_evidence_inventory"]),
        },
        "created_at": _utc_now_iso(),
    }


def _build_sample_anomaly_gallery(
    *,
    evaluation: dict[str, Any],
    qualitative_samples: dict[str, Any],
    explainability: dict[str, Any],
    samples_per_error_type: int,
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    eval_samples = _require_list(evaluation.get("samples"), "evaluation.samples")
    qual_samples = _require_list(qualitative_samples.get("samples"), "qualitative_samples.samples")
    explainability_map = {
        sample.get("sample_id"): sample for sample in _require_list(explainability.get("heatmaps"), "explainability.heatmaps")
    }
    qual_map = {sample.get("sample_id"): sample for sample in qual_samples}

    grouped: dict[str, list[dict[str, Any]]] = {
        "true_negative": [],
        "false_positive": [],
        "false_negative": [],
        "true_positive": [],
    }
    for sample in eval_samples:
        error_type = _error_type(sample)
        if error_type in grouped:
            grouped[error_type].append(sample)

    selected_samples: list[dict[str, Any]] = []
    counts_by_error_type = {key: len(value) for key, value in grouped.items()}
    selected_counts_by_error_type: dict[str, int] = {}
    for error_type in ("true_positive", "true_negative", "false_positive", "false_negative"):
        chosen = sorted(grouped[error_type], key=lambda item: item.get("sample_id", 0))[:samples_per_error_type]
        selected_counts_by_error_type[error_type] = len(chosen)
        for sample in chosen:
            sample_id = sample.get("sample_id")
            qual_sample = qual_map.get(sample_id, {})
            explain = explainability_map.get(sample_id, {})
            selected_samples.append(
                {
                    "sample_index": sample_id,
                    "image_path": sample.get("image_path"),
                    "true_label": sample.get("true_label"),
                    "true_label_name": _label_name(sample.get("true_label")),
                    "predicted_label": sample.get("predicted_label"),
                    "predicted_label_name": _label_name(sample.get("predicted_label")),
                    "anomaly_score": sample.get("anomaly_score"),
                    "threshold": evaluation.get("threshold"),
                    "confidence": _confidence_from_score(sample.get("anomaly_score")),
                    "error_type": error_type,
                    "run_id": RUN_ID,
                    "model_type": MODEL_TYPE,
                    "model_version": MODEL_VERSION,
                    "input_path": qual_sample.get("input_path") or explain.get("input_path"),
                    "reconstruction_path": qual_sample.get("reconstruction_path")
                    or explain.get("reconstruction_path"),
                    "heatmap_path": qual_sample.get("heatmap_path") or explain.get("heatmap_path"),
                    "overlay_path": qual_sample.get("overlay_path") or explain.get("overlay_path"),
                    "explanation": _sample_explanation(error_type),
                }
            )

    return {
        "artifact_type": "track_b_sample_anomaly_gallery",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "dataset_id": DATASET_ID,
        "split": "test",
        "run_id": RUN_ID,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "threshold": evaluation.get("threshold"),
        "total_samples": len(eval_samples),
        "gallery_sample_count": len(selected_samples),
        "samples_per_error_type": samples_per_error_type,
        "counts_by_error_type": counts_by_error_type,
        "selected_counts_by_error_type": selected_counts_by_error_type,
        "samples": selected_samples,
        "gallery_explanation": (
            "The gallery is balanced across the four confusion-group categories so reviewers can "
            "see both correct and incorrect autoencoder behavior without recomputing any predictions."
        ),
        "source_artifact_path": str(source_paths["evaluation"]),
        "generated_at": _utc_now_iso(),
    }


def _build_quality_decision_summary(
    *,
    quality_decision: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "artifact_type": "track_b_quality_decision_summary",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "run_id": RUN_ID,
        "model_name": MODEL_TYPE,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_version": "mvtec_1.0",
        "split": "test",
        "evidence_status": FRONTEND_EVIDENCE_STATUS,
        "model_quality_status": quality_decision.get("quality_status"),
        "quality_status": quality_decision.get("quality_status"),
        "dashboard_usage_recommendation": quality_decision.get("dashboard_usage_recommendation"),
        "production_ready": False,
        "deployment_safe": False,
        "deployment_candidate": False,
        "threshold": quality_decision.get("metrics_summary", {}).get("threshold"),
        "safe_wording": (
            "Surface Anomaly Detection is currently a review-only supporting signal. "
            "The dashboard may present governed evidence, but it must not claim production "
            "readiness or deployment safety."
        ),
        "forbidden_wording": [
            "production-ready",
            "deployment-safe",
            "final system complete",
        ],
        "limitations": quality_decision.get("limitations"),
        "reasons": quality_decision.get("reasons"),
        "next_recommended_step": "Use as a supporting review signal and investigate weak anomaly recall before stronger claims.",
        "source_artifact_paths": {
            "training_result": str(source_paths["training_result"]),
            "evaluation": str(source_paths["evaluation"]),
            "qualitative_samples": str(source_paths["qualitative_samples"]),
            "explainability": str(source_paths["reconstruction_explainability"]),
            "inventory": str(source_paths["inventory"]),
            "anomaly_quality_decision": str(source_paths["anomaly_quality_decision"]),
            "anomaly_governed_evidence_inventory": str(source_paths["anomaly_governed_evidence_inventory"]),
        },
        "metrics_summary": quality_decision.get("metrics_summary"),
        "metrics": quality_decision.get("metrics_summary"),
        "generated_at": _utc_now_iso(),
    }


def _build_frontend_anomaly_summary(
    *,
    evaluation: dict[str, Any],
    pr_curve: dict[str, Any],
    quality_decision: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    metrics = _require_dict(evaluation.get("metrics"), "evaluation.metrics")
    metrics_summary = _require_dict(quality_decision.get("metrics_summary"), "quality_decision.metrics_summary")
    return {
        "artifact_type": "track_b_frontend_anomaly_summary",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "dataset_id": DATASET_ID,
        "split": "test",
        "run_id": RUN_ID,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "evidence_status": FRONTEND_EVIDENCE_STATUS,
        "summary": (
            "The Surface Anomaly Detection autoencoder converts reconstruction error into an anomaly score. "
            "Higher scores indicate images that are less consistent with the training distribution. "
            "PR AUC and threshold sweep data are governed posthoc evidence derived from existing "
            "sample-level anomaly scores."
        ),
        "what_it_can_claim": [
            "Governed anomaly scores and thresholded test metrics.",
            "Governed PR AUC and threshold sweep derived from existing sample-level scores.",
            "Representative failure and explainability samples.",
        ],
        "what_it_cannot_claim": [
            "Production readiness.",
            "Deployment safety.",
            "Perfect separation between normal and anomalous images.",
        ],
        "limitations": [
            "ROC AUC is below 0.5 in the governed evaluation.",
            "Recall and F1 are very low at the governed selected threshold.",
            "This summary is a presentation layer only.",
        ],
        "next_step": "Use this bundle for honest dashboard presentation and investigate anomaly model quality before stronger claims.",
        "source_artifact_paths": {
            "evaluation": str(source_paths["evaluation"]),
            "anomaly_pr_curve": str(source_paths["anomaly_pr_curve"]),
            "anomaly_threshold_sweep": str(source_paths["anomaly_threshold_sweep"]),
            "anomaly_quality_decision": str(source_paths["anomaly_quality_decision"]),
        },
        "key_metrics": {
            "threshold": evaluation.get("threshold"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": pr_curve.get("pr_auc"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "train_score_count": evaluation.get("counts", {}).get("train_score_count"),
            "test_score_count": evaluation.get("counts", {}).get("test_score_count"),
            "normal_test_count": evaluation.get("counts", {}).get("normal_test_count"),
            "anomaly_test_count": evaluation.get("counts", {}).get("anomaly_test_count"),
        },
        "quality_decision": {
            "quality_status": quality_decision.get("quality_status"),
            "dashboard_usage_recommendation": quality_decision.get("dashboard_usage_recommendation"),
            "production_ready": False,
            "deployment_safe": False,
            "metrics_summary": metrics_summary,
        },
        "generated_at": _utc_now_iso(),
    }


def _build_sample_predictions_summary(
    *,
    sample_predictions: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "artifact_type": "track_b_sample_predictions",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "dataset_id": DATASET_ID,
        "split": "test",
        "run_id": RUN_ID,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "score_field": sample_predictions.get("score_field"),
        "score_definition": sample_predictions.get("score_definition"),
        "reconstruction_loss_mapping": sample_predictions.get("reconstruction_loss_mapping"),
        "threshold": sample_predictions.get("threshold"),
        "sample_count": sample_predictions.get("sample_count"),
        "samples": sample_predictions.get("samples"),
        "limitations": sample_predictions.get("limitations"),
        "source_artifact_path": str(source_paths["anomaly_sample_predictions"]),
        "generated_at": _utc_now_iso(),
    }


def _build_frontend_inventory(
    *,
    generated_files: list[Path],
    source_paths: dict[str, Path],
    selected_run_id: str,
    selected_model_type: str,
    selected_model_version: str,
    inventory_path: Path,
) -> dict[str, Any]:
    bundle_files = []
    for path in generated_files:
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

    inventory_payload = {
        "artifact_type": "track_b_frontend_artifact_inventory",
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "dataset_id": DATASET_ID,
        "split": "test",
        "run_id": selected_run_id,
        "selected_model_run_id": selected_run_id,
        "selected_model_type": selected_model_type,
        "selected_model_version": selected_model_version,
        "model_type": selected_model_type,
        "model_version": selected_model_version,
        "bundle_directory": str(inventory_path.parent),
        "generated_at": _utc_now_iso(),
        "generation_script": "scripts/evaluation/generate_track_b_frontend_bundle.py",
        "regeneration_command": "PYTHONPATH=src python scripts/evaluation/generate_track_b_frontend_bundle.py",
        "frontend_bundle_update_source": "governed anomaly evidence",
        "generated_from_existing_evidence": True,
        "no_new_inference": True,
        "no_retraining": True,
        "frontend_ui_code_modified": False,
        "bundle_artifact_count": len(bundle_files),
        "source_artifact_count": len(source_paths),
        "generated_file_paths": [str(path) for path in generated_files] + [str(inventory_path)],
        "source_artifact_paths": [str(path) for path in source_paths.values()],
        "bundle_files": bundle_files,
        "source_files": [
            {
                "name": name,
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        ],
        "missing_optional_files": [],
        "safe_demo_wording": (
            "This frontend bundle is a governed presentation layer for Track B only; "
            "it does not imply deployment safety."
        ),
        "validation_commands": [
            "python -m compileall scripts/evaluation src/inspection_ai api/app/schemas tests",
            "pytest tests/unit/test_track_b_frontend_bundle.py -q",
            "PYTHONPATH=src python scripts/evaluation/generate_track_b_frontend_bundle.py",
        ],
    }
    inventory_payload["bundle_files"].append(
        {
            "path": str(inventory_path),
            "artifact_type": "track_b_frontend_artifact_inventory",
            "exists": True,
            "included_in_hash_inventory": False,
            "self_referential": True,
        }
    )
    inventory_payload["bundle_artifact_count"] = len(inventory_payload["bundle_files"])
    return inventory_payload


def _validate_training_result(payload: dict[str, Any]) -> None:
    identity = _require_dict(payload.get("identity"), "training_result.identity")
    metadata = _require_dict(payload.get("metadata"), "training_result.metadata")
    artifacts = _require_dict(payload.get("artifacts"), "training_result.artifacts")
    metrics = _require_dict(payload.get("metrics"), "training_result.metrics")
    learning_curves = _require_dict(payload.get("learning_curves"), "training_result.learning_curves")
    if identity.get("run_id") != RUN_ID:
        raise ValueError("Track B run_id must match canonical run.")
    if identity.get("task_type") != TASK_TYPE:
        raise ValueError("TrainingResult task_type must be anomaly_detection.")
    if identity.get("model_type") != MODEL_TYPE:
        raise ValueError("TrainingResult model_type must be autoencoder.")
    if identity.get("run_config_id") != CONFIG_ID:
        raise ValueError("TrainingResult run_config_id must be autoencoder_train_v0_1_0.")
    if metadata.get("dataset_id") != DATASET_ID:
        raise ValueError("TrainingResult dataset_id must be mvtec_anomaly.")
    if metadata.get("model_name") != MODEL_TYPE:
        raise ValueError("TrainingResult model_name must be autoencoder.")
    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("TrainingResult model_version must be 0.1.0.")
    if "reconstruction_loss" not in metrics:
        raise ValueError("TrainingResult metrics must include reconstruction_loss.")
    if "train_loss" not in learning_curves:
        raise ValueError("TrainingResult learning_curves must include train_loss.")
    model_artifact = _require_dict(artifacts.get("model_artifact"), "training_result.artifacts.model_artifact")
    checkpoint_path = Path(_require_string(model_artifact.get("path"), "model_artifact.path"))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")


def _validate_evaluation(payload: dict[str, Any], training_result: dict[str, Any]) -> None:
    identity = _require_dict(training_result.get("identity"), "training_result.identity")
    metadata = _require_dict(training_result.get("metadata"), "training_result.metadata")
    if payload.get("artifact_type") != "anomaly_detection_evaluation":
        raise ValueError("Evaluation artifact_type must be anomaly_detection_evaluation.")
    if payload.get("run_id") != identity.get("run_id"):
        raise ValueError("Evaluation run_id must match TrainingResult run_id.")
    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError("Evaluation model_type must be autoencoder.")
    if payload.get("config_id") != CONFIG_ID:
        raise ValueError("Evaluation config_id must be autoencoder_train_v0_1_0.")
    if payload.get("dataset_id") != metadata.get("dataset_id"):
        raise ValueError("Evaluation dataset_id must match TrainingResult dataset_id.")
    metrics = _require_dict(payload.get("metrics"), "evaluation.metrics")
    if "roc_auc" not in metrics or "precision" not in metrics or "recall" not in metrics or "f1" not in metrics:
        raise ValueError("Evaluation metrics must include roc_auc, precision, recall, and f1.")
    _require_dict(payload.get("train_score_summary"), "evaluation.train_score_summary")
    _require_dict(payload.get("test_score_summary"), "evaluation.test_score_summary")
    counts = _require_dict(payload.get("counts"), "evaluation.counts")
    for field in ("train_score_count", "test_score_count", "normal_test_count", "anomaly_test_count"):
        _require_non_negative_int(counts.get(field), f"evaluation.counts.{field}")
    samples = _require_list(payload.get("samples"), "evaluation.samples")
    if len(samples) != counts.get("test_score_count"):
        raise ValueError("Evaluation samples length must match test_score_count.")


def _validate_learning_curves(payload: dict[str, Any], training_result: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "anomaly_learning_curves":
        raise ValueError("Learning curves artifact_type must be anomaly_learning_curves.")
    if payload.get("run_id") != training_result.get("identity", {}).get("run_id"):
        raise ValueError("Learning curves run_id must match TrainingResult run_id.")
    curves = _require_dict(payload.get("curves"), "learning_curves.curves")
    if "train_loss" not in curves:
        raise ValueError("Learning curves must include train_loss.")


def _validate_confusion_matrix(payload: dict[str, Any], training_result: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "anomaly_confusion_matrix":
        raise ValueError("Confusion-matrix artifact_type must be anomaly_confusion_matrix.")
    if payload.get("run_id") != training_result.get("identity", {}).get("run_id"):
        raise ValueError("Confusion-matrix run_id must match TrainingResult run_id.")
    splits = _require_dict(payload.get("splits"), "confusion_matrix.splits")
    for split_name in ("train", "test"):
        split = _require_dict(splits.get(split_name), f"confusion_matrix.splits.{split_name}")
        matrix = _require_list(split.get("matrix"), f"confusion_matrix.splits.{split_name}.matrix")
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError("Confusion matrix must be 2x2.")


def _validate_qualitative_samples(payload: dict[str, Any], training_result: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "anomaly_qualitative_samples":
        raise ValueError("Qualitative samples artifact_type must be anomaly_qualitative_samples.")
    if payload.get("run_id") != training_result.get("identity", {}).get("run_id"):
        raise ValueError("Qualitative samples run_id must match TrainingResult run_id.")
    samples = _require_list(payload.get("samples"), "qualitative_samples.samples")
    if not samples:
        raise ValueError("Qualitative samples must include at least one sample.")


def _validate_explainability(payload: dict[str, Any], training_result: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "anomaly_reconstruction_explainability":
        raise ValueError("Explainability artifact_type must be anomaly_reconstruction_explainability.")
    if payload.get("run_id") != training_result.get("identity", {}).get("run_id"):
        raise ValueError("Explainability run_id must match TrainingResult run_id.")
    heatmaps = _require_list(payload.get("heatmaps"), "explainability.heatmaps")
    if not heatmaps:
        raise ValueError("Explainability heatmaps must not be empty.")


def _validate_production_summary(payload: dict[str, Any], training_result: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "track_b_production_canonical_summary":
        raise ValueError("Production summary artifact_type must be track_b_production_canonical_summary.")
    if payload.get("run_id") != training_result.get("identity", {}).get("run_id"):
        raise ValueError("Production summary run_id must match TrainingResult run_id.")
    if payload.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("Production summary canonical_status must be production-canonical.")


def _validate_full_summary(payload: dict[str, Any], training_result: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "track_b_full_production_canonical_summary":
        raise ValueError("Full summary artifact_type must be track_b_full_production_canonical_summary.")
    if payload.get("run_id") != training_result.get("identity", {}).get("run_id"):
        raise ValueError("Full summary run_id must match TrainingResult run_id.")
    if payload.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("Full summary canonical_status must be production-canonical.")


def _validate_inventory(payload: dict[str, Any], training_result: dict[str, Any], evaluation: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "track_b_artifact_inventory":
        raise ValueError("Inventory artifact_type must be track_b_artifact_inventory.")
    if payload.get("run_id") != training_result.get("identity", {}).get("run_id"):
        raise ValueError("Inventory run_id must match TrainingResult run_id.")
    if payload.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("Inventory canonical_status must be production-canonical.")
    artifacts = _require_dict(payload.get("artifacts"), "inventory.artifacts")
    expected_keys = {
        "training_result",
        "model_checkpoint",
        "anomaly_evaluation",
        "learning_curves",
        "confusion_matrix",
        "qualitative_samples",
        "explainability",
    }
    if not expected_keys.issubset(artifacts.keys()):
        raise ValueError("Track B inventory is missing expected artifact entries.")


def _validate_governed_evidence(payload: dict[str, Any], expected_artifact_type: str) -> None:
    if payload.get("artifact_type") != expected_artifact_type:
        raise ValueError(f"Governed evidence artifact_type must be {expected_artifact_type}.")
    if payload.get("run_id") != RUN_ID:
        raise ValueError("Governed evidence run_id must match Track B run_id.")
    _require_string(payload.get("source_artifact_path"), f"{expected_artifact_type}.source_artifact_path")
    _require_string(payload.get("generated_at_utc"), f"{expected_artifact_type}.generated_at_utc")
    if expected_artifact_type == "anomaly_pr_curve":
        if payload.get("pr_auc") is None:
            raise ValueError("Anomaly PR curve must include pr_auc.")
    if expected_artifact_type == "anomaly_threshold_sweep":
        rows = _require_list(payload.get("rows"), "anomaly_threshold_sweep.rows")
        if len(rows) <= 1:
            raise ValueError("Anomaly threshold sweep must include more than one row.")
    if expected_artifact_type == "anomaly_score_distribution":
        _require_dict(payload.get("summary"), "anomaly_score_distribution.summary")
        _require_dict(payload.get("histograms"), "anomaly_score_distribution.histograms")
    if expected_artifact_type == "anomaly_sample_predictions":
        samples = _require_list(payload.get("samples"), "anomaly_sample_predictions.samples")
        if not samples:
            raise ValueError("Anomaly sample predictions must include samples.")
    if expected_artifact_type == "anomaly_quality_decision":
        if payload.get("production_ready") is not False:
            raise ValueError("Anomaly quality decision must set production_ready=false.")
        if payload.get("deployment_safe") is not False:
            raise ValueError("Anomaly quality decision must set deployment_safe=false.")


def _validate_governed_evidence_inventory(payload: dict[str, Any]) -> None:
    if payload.get("inventory_type") != "anomaly_governed_evidence_inventory":
        raise ValueError("Governed evidence inventory_type must be anomaly_governed_evidence_inventory.")
    if payload.get("run_id") != RUN_ID:
        raise ValueError("Governed evidence inventory run_id must match Track B run_id.")
    artifacts = _require_dict(payload.get("artifacts"), "governed_evidence_inventory.artifacts")
    expected = {
        "anomaly_pr_curve",
        "anomaly_threshold_sweep",
        "anomaly_score_distribution",
        "anomaly_sample_predictions",
        "anomaly_quality_decision",
    }
    if not expected.issubset(artifacts):
        raise ValueError("Governed evidence inventory is missing required derived artifacts.")


def _error_type(sample: dict[str, Any]) -> str:
    true_label = sample.get("true_label")
    predicted_label = sample.get("predicted_label")
    if true_label == "normal" and predicted_label == "normal":
        return "true_negative"
    if true_label == "normal" and predicted_label == "anomaly":
        return "false_positive"
    if true_label == "anomaly" and predicted_label == "normal":
        return "false_negative"
    if true_label == "anomaly" and predicted_label == "anomaly":
        return "true_positive"
    return "unknown"


def _label_name(label: Any) -> str | None:
    if label == "normal":
        return "normal"
    if label == "anomaly":
        return "anomaly"
    return None


def _confidence_from_score(score: Any) -> float | None:
    if score is None:
        return None
    try:
        numeric = float(score)
    except Exception:
        return None
    return max(0.0, min(1.0, 1.0 - numeric))


def _sample_explanation(error_type: str) -> str:
    if error_type == "true_positive":
        return "Correctly flagged as anomalous."
    if error_type == "true_negative":
        return "Correctly retained as normal."
    if error_type == "false_positive":
        return "Normal image was flagged as anomalous."
    if error_type == "false_negative":
        return "Anomalous image was missed by the canonical threshold."
    return "Governed sample from the Track B evaluation set."


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


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
