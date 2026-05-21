"""Generate posthoc governed anomaly evidence from existing evaluation output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from numbers import Real
from pathlib import Path
import statistics
from typing import Any


DEFAULT_EVALUATION_PATH = Path(
    "artifacts/models/metrics/"
    "anomaly_detection_evaluation__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
)
DEFAULT_METRICS_DIR = Path("artifacts/models/metrics")
DEFAULT_PREDICTIONS_DIR = Path("artifacts/models/predictions")
DEFAULT_INVENTORY_DIR = Path("artifacts/models/inventory")
GENERATION_SCRIPT = "scripts/evaluation/generate_anomaly_governed_evidence.py"

SCORE_DEFINITION = "mean_squared_reconstruction_error_per_image"
POSITIVE_LABEL = 1
QUANTILES = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
ARTIFACT_PURPOSES = {
    "anomaly_pr_curve": "Derived PR curve and PR AUC from governed sample-level anomaly scores.",
    "anomaly_threshold_sweep": "Derived threshold behavior table from governed sample-level anomaly scores.",
    "anomaly_score_distribution": (
        "Derived anomaly score and reconstruction-loss distribution data from governed sample-level scores."
    ),
    "anomaly_sample_predictions": "Frontend-ready sample-level anomaly prediction export from governed evaluation rows.",
    "anomaly_quality_decision": "Governed quality decision for safe dashboard usage of anomaly evidence.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate governed anomaly evidence from existing sample scores."
    )
    parser.add_argument(
        "--evaluation",
        default=str(DEFAULT_EVALUATION_PATH),
        help="Path to anomaly_detection_evaluation JSON.",
    )
    parser.add_argument(
        "--metrics-dir",
        default=str(DEFAULT_METRICS_DIR),
        help="Directory for derived metrics artifacts.",
    )
    parser.add_argument(
        "--predictions-dir",
        default=str(DEFAULT_PREDICTIONS_DIR),
        help="Directory for derived prediction artifacts.",
    )
    parser.add_argument(
        "--inventory-dir",
        default=str(DEFAULT_INVENTORY_DIR),
        help="Directory for the governed derived-evidence inventory artifact.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evaluation_path = Path(args.evaluation)
    metrics_dir = Path(args.metrics_dir)
    predictions_dir = Path(args.predictions_dir)
    inventory_dir = Path(args.inventory_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)

    evaluation = load_evaluation(evaluation_path)
    records = validate_evaluation(evaluation)
    run_id = _require_string(evaluation.get("run_id"), "run_id")

    generated_at = _utc_now_iso()
    source_path = evaluation_path.as_posix()
    pr_curve = build_pr_curve_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path=source_path,
        generated_at_utc=generated_at,
    )
    threshold_sweep = build_threshold_sweep_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path=source_path,
        generated_at_utc=generated_at,
    )
    score_distribution = build_score_distribution_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path=source_path,
        generated_at_utc=generated_at,
    )
    sample_predictions = build_sample_predictions_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path=source_path,
        generated_at_utc=generated_at,
    )
    quality_decision = build_quality_decision_artifact(
        evaluation=evaluation,
        pr_curve=pr_curve,
        threshold_sweep=threshold_sweep,
        score_distribution=score_distribution,
        sample_predictions=sample_predictions,
        source_artifact_path=source_path,
        generated_at_utc=generated_at,
    )

    output_paths = [
        metrics_dir / f"anomaly_pr_curve__{run_id}__test.json",
        metrics_dir / f"anomaly_threshold_sweep__{run_id}__test.json",
        metrics_dir / f"anomaly_score_distribution__{run_id}__test.json",
        predictions_dir / f"anomaly_sample_predictions__{run_id}__test.json",
        metrics_dir / f"anomaly_quality_decision__{run_id}__test.json",
    ]
    payloads = [
        pr_curve,
        threshold_sweep,
        score_distribution,
        sample_predictions,
        quality_decision,
    ]
    for path, payload in zip(output_paths, payloads, strict=True):
        _write_json(path, payload)
        print(f"created={path.as_posix()}")

    inventory_path = inventory_dir / f"anomaly_governed_evidence_inventory__{run_id}__test.json"
    inventory = build_governed_evidence_inventory(
        run_id=run_id,
        source_artifact_path=source_path,
        generated_at_utc=generated_at,
        artifact_paths=output_paths,
    )
    _write_json(inventory_path, inventory)
    print(f"created={inventory_path.as_posix()}")
    return 0


def load_evaluation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Anomaly evaluation artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Anomaly evaluation JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Anomaly evaluation artifact must contain a JSON object.")
    return payload


def validate_evaluation(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    for field in ("run_id", "threshold", "score_definition", "metrics", "counts", "samples"):
        if field not in evaluation:
            raise ValueError(f"Evaluation missing required field: {field}")

    if _require_string(evaluation.get("score_definition"), "score_definition") != SCORE_DEFINITION:
        raise ValueError(f"score_definition must equal {SCORE_DEFINITION!r}.")
    threshold = _require_number(evaluation.get("threshold"), "threshold")
    metrics = _require_dict(evaluation.get("metrics"), "metrics")
    counts = _require_dict(evaluation.get("counts"), "counts")
    samples = _require_list(evaluation.get("samples"), "samples")
    if not samples:
        raise ValueError("samples must be non-empty.")

    expected_count = counts.get("test_score_count")
    if expected_count is not None and len(samples) != _require_non_negative_int(
        expected_count, "counts.test_score_count"
    ):
        raise ValueError("samples length must match counts.test_score_count.")

    records = []
    for index, sample in enumerate(samples):
        item = _require_dict(sample, f"samples[{index}]")
        for field in (
            "sample_id",
            "image_path",
            "true_label_id",
            "anomaly_score",
            "predicted_label_id",
            "correct",
        ):
            if field not in item:
                raise ValueError(f"samples[{index}] missing required field: {field}")
        label = _require_binary_int(item.get("true_label_id"), f"samples[{index}].true_label_id")
        score = _require_number(item.get("anomaly_score"), f"samples[{index}].anomaly_score")
        predicted = _require_binary_int(
            item.get("predicted_label_id"), f"samples[{index}].predicted_label_id"
        )
        recomputed = prediction_from_score(score, threshold)
        if predicted != recomputed:
            raise ValueError(
                "Existing predicted_label_id does not match score > threshold rule "
                f"for sample_id={item.get('sample_id')}."
            )
        correct = item.get("correct")
        if not isinstance(correct, bool):
            raise ValueError(f"samples[{index}].correct must be boolean.")
        if correct != (label == predicted):
            raise ValueError(f"samples[{index}].correct does not match labels.")
        records.append(
            {
                **item,
                "true_label_id": label,
                "anomaly_score": score,
                "predicted_label_id": predicted,
            }
        )

    labels = [record["true_label_id"] for record in records]
    if set(labels) != {0, 1}:
        raise ValueError("true_label_id must contain both 0 and 1.")

    _validate_selected_threshold_counts(records, counts)
    for metric in ("roc_auc", "precision", "recall", "f1"):
        if metric in metrics:
            _require_number(metrics.get(metric), f"metrics.{metric}")
    return records


def build_pr_curve_artifact(
    *,
    evaluation: dict[str, Any],
    records: list[dict[str, Any]],
    source_artifact_path: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    labels = [record["true_label_id"] for record in records]
    scores = [record["anomaly_score"] for record in records]
    pr_auc = average_precision(labels, scores)
    curve_points = precision_recall_curve_points(labels, scores)
    return {
        "artifact_type": "anomaly_pr_curve",
        "task_type": "anomaly_detection",
        "run_id": evaluation["run_id"],
        "source_artifact_path": source_artifact_path,
        "generated_at_utc": generated_at_utc,
        "metric_name": "average_precision_pr_auc",
        "pr_auc": pr_auc,
        "positive_label": POSITIVE_LABEL,
        "score_field": "anomaly_score",
        "score_definition": evaluation["score_definition"],
        "sample_count": len(records),
        "curve_points": curve_points,
        "limitations": [
            "Derived posthoc from existing governed sample-level anomaly scores.",
            "No new inference or retraining was performed.",
        ],
    }


def build_threshold_sweep_artifact(
    *,
    evaluation: dict[str, Any],
    records: list[dict[str, Any]],
    source_artifact_path: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    selected_threshold = _require_number(evaluation.get("threshold"), "threshold")
    scores = [record["anomaly_score"] for record in records]
    thresholds = sorted(
        {
            round(_percentile(scores, quantile), 12)
            for quantile in QUANTILES
        }
        | {round(selected_threshold, 12)}
    )
    rows = [threshold_metrics(records, threshold) for threshold in thresholds]
    selected = threshold_metrics(records, selected_threshold)
    return {
        "artifact_type": "anomaly_threshold_sweep",
        "task_type": "anomaly_detection",
        "run_id": evaluation["run_id"],
        "source_artifact_path": source_artifact_path,
        "generated_at_utc": generated_at_utc,
        "selected_threshold": selected_threshold,
        "selected_threshold_strategy": evaluation.get("threshold_strategy"),
        "score_field": "anomaly_score",
        "score_definition": evaluation["score_definition"],
        "sample_count": len(records),
        "quantiles": QUANTILES,
        "rows": rows,
        "selected_threshold_metrics": selected,
        "limitations": [
            "Threshold sweep is derived from existing governed sample-level scores.",
            "No new inference or retraining was performed.",
        ],
    }


def build_score_distribution_artifact(
    *,
    evaluation: dict[str, Any],
    records: list[dict[str, Any]],
    source_artifact_path: str,
    generated_at_utc: str,
    bin_count: int = 20,
) -> dict[str, Any]:
    all_scores = [record["anomaly_score"] for record in records]
    normal_scores = [record["anomaly_score"] for record in records if record["true_label_id"] == 0]
    anomaly_scores = [record["anomaly_score"] for record in records if record["true_label_id"] == 1]
    return {
        "artifact_type": "anomaly_score_distribution",
        "task_type": "anomaly_detection",
        "run_id": evaluation["run_id"],
        "source_artifact_path": source_artifact_path,
        "generated_at_utc": generated_at_utc,
        "score_field": "anomaly_score",
        "score_definition": evaluation["score_definition"],
        "reconstruction_loss_mapping": (
            "reconstruction_loss is equal to anomaly_score because score_definition is "
            "mean_squared_reconstruction_error_per_image."
        ),
        "threshold": evaluation.get("threshold"),
        "sample_count": len(records),
        "summary": {
            "all": summary_stats(all_scores),
            "true_normal": summary_stats(normal_scores),
            "true_anomaly": summary_stats(anomaly_scores),
        },
        "histograms": {
            "all": histogram(all_scores, bin_count=bin_count),
            "true_normal": histogram(normal_scores, bin_count=bin_count, bounds=score_bounds(all_scores)),
            "true_anomaly": histogram(anomaly_scores, bin_count=bin_count, bounds=score_bounds(all_scores)),
        },
        "limitations": [
            "Distribution is derived from existing governed sample-level anomaly scores.",
            "No full-sample reconstruction images are created by this artifact.",
        ],
    }


def build_sample_predictions_artifact(
    *,
    evaluation: dict[str, Any],
    records: list[dict[str, Any]],
    source_artifact_path: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    threshold = _require_number(evaluation.get("threshold"), "threshold")
    rows = []
    for record in records:
        rows.append(
            {
                "sample_id": record.get("sample_id"),
                "image_path": record.get("image_path"),
                "true_label": record.get("true_label"),
                "true_label_id": record.get("true_label_id"),
                "defect_type": record.get("defect_type"),
                "mask_path": record.get("mask_path"),
                "anomaly_score": record.get("anomaly_score"),
                "reconstruction_loss": record.get("anomaly_score"),
                "threshold": threshold,
                "predicted_label": record.get("predicted_label"),
                "predicted_label_id": record.get("predicted_label_id"),
                "correct": record.get("correct"),
            }
        )
    return {
        "artifact_type": "anomaly_sample_predictions",
        "task_type": "anomaly_detection",
        "run_id": evaluation["run_id"],
        "source_artifact_path": source_artifact_path,
        "generated_at_utc": generated_at_utc,
        "score_field": "anomaly_score",
        "score_definition": evaluation["score_definition"],
        "reconstruction_loss_mapping": (
            "reconstruction_loss is equal to anomaly_score because score_definition is "
            "mean_squared_reconstruction_error_per_image."
        ),
        "threshold": threshold,
        "sample_count": len(rows),
        "samples": rows,
        "limitations": [
            "Sample predictions are copied from existing governed evaluation output.",
            "No reconstruction image paths are added or inferred.",
        ],
    }


def build_quality_decision_artifact(
    *,
    evaluation: dict[str, Any],
    pr_curve: dict[str, Any],
    threshold_sweep: dict[str, Any],
    score_distribution: dict[str, Any],
    sample_predictions: dict[str, Any],
    source_artifact_path: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    metrics = _require_dict(evaluation.get("metrics"), "metrics")
    roc_auc = _require_number(metrics.get("roc_auc"), "metrics.roc_auc")
    recall = _require_number(metrics.get("recall"), "metrics.recall")
    f1 = _require_number(metrics.get("f1"), "metrics.f1")
    reasons = []
    if roc_auc < 0.5:
        reasons.append("ROC AUC is below 0.5 on the governed test split.")
    if recall < 0.1:
        reasons.append("Recall is very low at the governed threshold.")
    if f1 < 0.1:
        reasons.append("F1 is very low at the governed threshold.")
    if not reasons:
        reasons.append("Metrics still require manual review before dashboard claims.")

    return {
        "artifact_type": "anomaly_quality_decision",
        "task_type": "anomaly_detection",
        "run_id": evaluation["run_id"],
        "source_artifact_path": source_artifact_path,
        "generated_at_utc": generated_at_utc,
        "model_family": "autoencoder",
        "quality_status": "review_required_weak_evidence",
        "dashboard_usage_recommendation": "review_only_signal",
        "production_ready": False,
        "deployment_safe": False,
        "reasons": reasons,
        "metrics_summary": {
            "roc_auc": roc_auc,
            "precision": _require_number(metrics.get("precision"), "metrics.precision"),
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_curve["pr_auc"],
            "threshold": evaluation.get("threshold"),
            "sample_count": sample_predictions["sample_count"],
        },
        "derived_artifacts": {
            "pr_curve": "anomaly_pr_curve",
            "threshold_sweep": threshold_sweep["artifact_type"],
            "score_distribution": score_distribution["artifact_type"],
            "sample_predictions": sample_predictions["artifact_type"],
        },
        "limitations": [
            "This quality decision is derived from existing governed evaluation evidence.",
            "It does not claim production readiness or deployment safety.",
        ],
    }


def build_governed_evidence_inventory(
    *,
    run_id: str,
    source_artifact_path: str,
    generated_at_utc: str,
    artifact_paths: list[Path],
) -> dict[str, Any]:
    artifacts = {}
    for path in artifact_paths:
        payload = load_evaluation(path)
        artifact_type = _require_string(payload.get("artifact_type"), f"{path}.artifact_type")
        artifacts[artifact_type] = {
            "artifact_type": artifact_type,
            "path": path.as_posix(),
            "exists": True,
            "run_id": _require_string(payload.get("run_id"), f"{path}.run_id"),
            "source_artifact_path": _require_string(
                payload.get("source_artifact_path"), f"{path}.source_artifact_path"
            ),
            "generated_at_utc": _require_string(
                payload.get("generated_at_utc"), f"{path}.generated_at_utc"
            ),
            "sha256": _sha256(path),
            "file_size_bytes": path.stat().st_size,
            "generation_script": GENERATION_SCRIPT,
            "generated_from_existing_evidence": True,
            "no_new_inference": True,
            "no_retraining": True,
            "frontend_ready": True,
            "required_for_frontend": True,
            "purpose": ARTIFACT_PURPOSES.get(artifact_type, "Derived governed anomaly evidence."),
        }

    expected_artifacts = set(ARTIFACT_PURPOSES)
    missing = sorted(expected_artifacts - set(artifacts))
    if missing:
        raise ValueError(f"Governed evidence inventory missing artifacts: {missing}")

    return {
        "inventory_type": "anomaly_governed_evidence_inventory",
        "task_type": "anomaly_detection",
        "track_id": "track_b",
        "run_id": run_id,
        "split": "test",
        "source_artifact_path": source_artifact_path,
        "generated_at_utc": generated_at_utc,
        "generation_script": GENERATION_SCRIPT,
        "regeneration_command": (
            "PYTHONPATH=src python scripts/evaluation/generate_anomaly_governed_evidence.py"
        ),
        "validation_commands": [
            "python -m compileall scripts/evaluation src/inspection_ai api/app/schemas tests",
            "pytest tests/unit/test_anomaly_governed_evidence.py -q",
        ],
        "generated_from_existing_evidence": True,
        "no_new_inference": True,
        "no_retraining": True,
        "registry_update_deferred": True,
        "frontend_bundle_update_deferred": True,
        "artifacts": artifacts,
        "validation_checks": [
            {
                "name": "source_artifact_declared",
                "status": "PASS",
                "details": source_artifact_path,
            },
            {
                "name": "derived_artifact_count",
                "status": "PASS",
                "details": f"count={len(artifacts)}",
            },
            {
                "name": "hashes_computed",
                "status": "PASS",
                "details": "sha256 recorded for every derived anomaly evidence artifact.",
            },
            {
                "name": "no_new_inference_or_retraining",
                "status": "PASS",
                "details": "Inventory covers posthoc derivations from existing governed evaluation output only.",
            },
            {
                "name": "registry_update_deferred",
                "status": "PASS",
                "details": "Global artifact_registry.yaml publication is deferred to a later explicit step.",
            },
        ],
        "known_limitations": [
            "This inventory governs derived evidence artifacts only.",
            "The artifacts are generated from existing sample-level anomaly evaluation output.",
            "No model inference, model training, or metric invention is performed.",
            "Global artifact registry publication is deferred to a later explicit step.",
            "Frontend bundle integration is deferred to a later explicit step.",
        ],
    }


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(1 for label in labels if label == POSITIVE_LABEL)
    if positives == 0:
        raise ValueError("average precision requires at least one positive label.")
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label == POSITIVE_LABEL:
            true_positive += 1
            precision_sum += true_positive / index
    return float(precision_sum / positives)


def precision_recall_curve_points(labels: list[int], scores: list[float]) -> list[dict[str, float | None]]:
    thresholds = sorted(set(scores), reverse=True)
    points = []
    for threshold in thresholds:
        metrics = _binary_metrics(labels, [1 if score >= threshold else 0 for score in scores])
        points.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
    points.append({"threshold": None, "precision": 1.0, "recall": 0.0})
    return points


def threshold_metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [record["true_label_id"] for record in records]
    predictions = [prediction_from_score(record["anomaly_score"], threshold) for record in records]
    metrics = _binary_metrics(labels, predictions)
    return {
        "threshold": float(threshold),
        **metrics,
        "false_positive_rate": _safe_ratio(metrics["fp"], metrics["fp"] + metrics["tn"]),
        "false_negative_rate": _safe_ratio(metrics["fn"], metrics["fn"] + metrics["tp"]),
        "predicted_anomaly_count": metrics["tp"] + metrics["fp"],
        "predicted_normal_count": metrics["tn"] + metrics["fn"],
    }


def prediction_from_score(score: float, threshold: float) -> int:
    return 1 if score > threshold else 0


def summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
        "p05": _percentile(values, 5),
        "p25": _percentile(values, 25),
        "p75": _percentile(values, 75),
        "p95": _percentile(values, 95),
    }


def histogram(
    values: list[float],
    *,
    bin_count: int = 20,
    bounds: tuple[float, float] | None = None,
) -> list[dict[str, float | int]]:
    if bin_count < 1:
        raise ValueError("bin_count must be >= 1.")
    if not values:
        return []
    lower, upper = bounds if bounds is not None else score_bounds(values)
    if lower == upper:
        return [{"bin_start": lower, "bin_end": upper, "count": len(values)}]
    width = (upper - lower) / bin_count
    bins = [
        {"bin_start": lower + (index * width), "bin_end": lower + ((index + 1) * width), "count": 0}
        for index in range(bin_count)
    ]
    for value in values:
        index = int((value - lower) / width)
        if index >= bin_count:
            index = bin_count - 1
        if index < 0:
            index = 0
        bins[index]["count"] += 1
    return bins


def score_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty.")
    return min(values), max(values)


def _binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")
    tp = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 0 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 0 and pred == 0)
    fn = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred == 0)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else float(2 * precision * recall / (precision + recall))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _validate_selected_threshold_counts(records: list[dict[str, Any]], counts: dict[str, Any]) -> None:
    labels = [record["true_label_id"] for record in records]
    predictions = [record["predicted_label_id"] for record in records]
    metrics = _binary_metrics(labels, predictions)
    comparisons = {
        "normal_test_count": metrics["tn"] + metrics["fp"],
        "anomaly_test_count": metrics["tp"] + metrics["fn"],
        "predicted_normal_count": metrics["tn"] + metrics["fn"],
        "predicted_anomaly_count": metrics["tp"] + metrics["fp"],
        "correct_count": metrics["tn"] + metrics["tp"],
        "incorrect_count": metrics["fp"] + metrics["fn"],
    }
    for field, actual in comparisons.items():
        if field in counts and _require_non_negative_int(counts[field], f"counts.{field}") != actual:
            raise ValueError(f"Derived {field} does not match evaluation counts.")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must be non-empty.")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    return float(
        sorted_values[lower_index]
        + ((sorted_values[upper_index] - sorted_values[lower_index]) * fraction)
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dictionary.")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric.")
    return float(value)


def _require_non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _require_binary_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise ValueError(f"{field} must be 0 or 1.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
