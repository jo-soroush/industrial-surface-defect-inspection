"""Generate governed Track A threshold analysis from prediction-level artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]
DEFAULT_OUTPUT_DIR = Path("artifacts/models/error_analysis")
ALLOWED_SPLITS = {"validation"}
REQUIRED_RECORD_FIELDS = (
    "sample_index",
    "image_path",
    "true_label",
    "true_label_name",
    "predicted_label",
    "predicted_label_name",
    "probability_good",
    "probability_defect",
    "confidence",
    "error_type",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Track A threshold analysis from prediction-level artifacts."
    )
    parser.add_argument(
        "--prediction-level-analysis",
        required=True,
        help="Path to a governed Track A prediction-level analysis JSON artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the threshold analysis JSON will be written.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help=(
            "Threshold sweep values for probability_defect. Defaults to "
            "0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned output path without writing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    prediction_path = Path(args.prediction_level_analysis)
    output_dir = Path(args.output_dir)
    thresholds = _validate_thresholds(args.thresholds)

    payload = _load_json_file(prediction_path, "prediction-level analysis")
    analysis = _validate_prediction_level_analysis(payload, prediction_path)
    run_id = _require_string(analysis.get("run_id"), "run_id")
    split = _require_string(analysis.get("split"), "split")
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Prediction-level analysis split must be one of {sorted(ALLOWED_SPLITS)}."
        )

    planned_output_path = output_dir / f"track_a_threshold_analysis__{run_id}__{split}.json"

    if args.dry_run:
        print("track_a_threshold_analysis_dry_run=pass")
        print(f"run_id={run_id}")
        print(f"split={split}")
        print(f"planned_output_path={planned_output_path}")
        print(f"threshold_count={len(thresholds)}")
        print(f"records_count={len(analysis['records'])}")
        return 0

    rows, baseline_row, recommended_row = _compute_threshold_rows(
        records=analysis["records"],
        thresholds=thresholds,
    )
    threshold_analysis = {
        "artifact_type": "track_a_threshold_analysis",
        "run_id": run_id,
        "source_prediction_level_analysis_path": str(prediction_path),
        "split": split,
        "decision_score": "probability_defect",
        "threshold_rule": "predict defect if probability_defect >= threshold",
        "thresholds": thresholds,
        "per_threshold": rows,
        "recommended_threshold": recommended_row["threshold"],
        "recommendation_reason": _build_recommendation_reason(recommended_row),
        "baseline_threshold": 0.5,
        "baseline_metrics": baseline_row,
        "created_at": _utc_now_iso(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with planned_output_path.open("w", encoding="utf-8") as handle:
        json.dump(threshold_analysis, handle, indent=2)

    print(f"output_path={planned_output_path}")
    print(f"artifact_type={threshold_analysis['artifact_type']}")
    print(f"run_id={threshold_analysis['run_id']}")
    print(f"split={threshold_analysis['split']}")
    print(f"recommended_threshold={threshold_analysis['recommended_threshold']}")
    print(f"baseline_threshold={threshold_analysis['baseline_threshold']}")
    print(f"records_count={len(analysis['records'])}")
    print(f"threshold_count={len(thresholds)}")
    return 0


def _compute_threshold_rows(
    *, records: list[dict[str, Any]], thresholds: list[float]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_row: dict[str, Any] | None = None

    for threshold in thresholds:
        row = _evaluate_threshold(records, threshold)
        rows.append(row)
        if abs(threshold - 0.5) < 1e-12:
            baseline_row = row

    if baseline_row is None:
        baseline_row = _evaluate_threshold(records, 0.5)

    recommended_row = max(
        rows,
        key=lambda row: (
            row["macro_f1"],
            row["recall"],
            -row["false_positive"],
            -abs(row["threshold"] - 0.5),
            -row["threshold"],
        ),
    )
    return rows, baseline_row, recommended_row


def _evaluate_threshold(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for record in records:
        true_label = _require_binary_label(record.get("true_label"), "true_label")
        probability_defect = _require_probability(
            record.get("probability_defect"), "probability_defect"
        )

        predicted_label = 1 if probability_defect >= threshold else 0
        if true_label == 0 and predicted_label == 0:
            tn += 1
        elif true_label == 0 and predicted_label == 1:
            fp += 1
        elif true_label == 1 and predicted_label == 0:
            fn += 1
        else:
            tp += 1

    total = tn + fp + fn + tp
    if total <= 0:
        raise ValueError("Threshold analysis requires at least one record.")

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_f1(precision, recall)
    class_0_precision = _safe_ratio(tn, tn + fn)
    class_0_recall = _safe_ratio(tn, tn + fp)
    class_0_f1 = _safe_f1(class_0_precision, class_0_recall)
    macro_f1 = (class_0_f1 + f1) / 2.0
    accuracy = _safe_ratio(tp + tn, total)

    return {
        "threshold": threshold,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _build_recommendation_reason(recommended_row: dict[str, Any]) -> str:
    return (
        "Selected threshold "
        f"{recommended_row['threshold']:.2f} because it maximized macro F1 "
        f"({recommended_row['macro_f1']:.6f}); ties were broken by higher defect "
        f"recall, then fewer false positives, then proximity to 0.5."
    )


def _validate_prediction_level_analysis(
    payload: dict[str, Any], artifact_path: Path
) -> dict[str, Any]:
    if payload.get("artifact_type") != "track_a_prediction_level_error_analysis":
        raise ValueError(
            "prediction-level analysis artifact_type must be track_a_prediction_level_error_analysis."
        )
    run_id = _require_string(payload.get("run_id"), "run_id")
    split = _require_string(payload.get("split"), "split")
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"prediction-level analysis split must be one of {sorted(ALLOWED_SPLITS)}."
        )

    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("prediction-level analysis records must be a list.")
    if len(records) == 0:
        raise ValueError("prediction-level analysis records must not be empty.")

    if payload.get("partial_sample") is True:
        raise ValueError(
            "prediction-level analysis must be full validation evidence, not partial_sample=true."
        )

    summary = _require_dict(payload.get("summary"), "summary")
    _validate_confusion_matrix(summary.get("confusion_matrix"), "summary.confusion_matrix")

    for index, record in enumerate(records):
        _validate_prediction_record(record, index)

    if _count_records(records) != payload.get("total_samples"):
        raise ValueError(
            "prediction-level analysis total_samples must match the number of records."
        )

    return payload


def _validate_prediction_record(record: Any, index: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Record {index} must be a dictionary.")
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            raise ValueError(f"Record {index} is missing required field: {field}")

    sample_index = record.get("sample_index")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError(f"Record {index} sample_index must be a non-negative integer.")

    _require_string(record.get("image_path"), f"record[{index}].image_path")
    true_label = _require_binary_label(record.get("true_label"), f"record[{index}].true_label")
    predicted_label = _require_binary_label(
        record.get("predicted_label"), f"record[{index}].predicted_label"
    )
    for label_field in ("true_label_name", "predicted_label_name", "error_type"):
        _require_string(record.get(label_field), f"record[{index}].{label_field}")

    _require_probability(record.get("probability_good"), f"record[{index}].probability_good")
    _require_probability(
        record.get("probability_defect"), f"record[{index}].probability_defect"
    )
    _require_probability(record.get("confidence"), f"record[{index}].confidence")

    expected_error_type = _error_type(true_label, predicted_label)
    if record.get("error_type") != expected_error_type:
        raise ValueError(
            f"Record {index} error_type must be {expected_error_type!r}."
        )


def _count_records(records: list[dict[str, Any]]) -> int:
    return len(records)


def _validate_confusion_matrix(matrix: Any, field_name: str) -> None:
    if not isinstance(matrix, list) or len(matrix) != 2:
        raise ValueError(f"{field_name} must be a 2x2 list.")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{field_name} must be a 2x2 list.")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} values must be non-negative integers.")


def _validate_thresholds(thresholds: list[float]) -> list[float]:
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    validated: list[float] = []
    for threshold in thresholds:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("Thresholds must be numeric.")
        value = float(threshold)
        if value < 0.0 or value > 1.0:
            raise ValueError("Thresholds must be in the inclusive range [0, 1].")
        validated.append(value)
    return validated


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _error_type(true_label: int, predicted_label: int) -> str:
    if true_label == 0 and predicted_label == 0:
        return "true_negative"
    if true_label == 0 and predicted_label == 1:
        return "false_positive"
    if true_label == 1 and predicted_label == 0:
        return "false_negative"
    return "true_positive"


def _require_binary_label(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a binary integer label.")
    if value not in (0, 1):
        raise ValueError(f"{field_name} must be 0 or 1.")
    return value


def _require_probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _load_json_file(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON must contain an object.")
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


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
