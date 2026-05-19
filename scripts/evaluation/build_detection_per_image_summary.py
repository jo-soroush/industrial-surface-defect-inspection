"""Build a governed Detection/YOLO per-image prediction summary."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_BBOX_PREDICTION_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_per_image_summary__yolo_train_v0_2_0__validation.json"
)
EXPECTED_RUN_ID = "yolo_train_v0_2_0"
EXPECTED_RUN_CONFIG_ID = "yolo_train_v0_2_0"
EXPECTED_MODEL_NAME = "yolo"
EXPECTED_MODEL_TYPE = "yolo"
EXPECTED_MODEL_VERSION = "0.2.0"
EXPECTED_DATASET_ID = "gc10det_detection"
EXPECTED_DATASET_VERSION = "gc10det_1.0"
EXPECTED_TRACK_ID = "detection"
EXPECTED_TASK_TYPE = "object_detection"
EXPECTED_SPLIT = "validation"


def main() -> int:
    source_path = SOURCE_BBOX_PREDICTION_PATH
    output_path = DEFAULT_OUTPUT_PATH

    try:
        source = _load_json(source_path, "source bbox prediction artifact")
        _validate_source(source, source_path)

        source_hash = _sha256_file(source_path)
        summary_rows, counts = _build_summary_rows(source)
        summary = _build_summary(source, source_hash, summary_rows, counts)

        _validate_summary(summary, source)
        _write_json_atomic(output_path, summary)

        written = _load_json(output_path, "written summary artifact")
        _validate_written_output(written, source)

        print("# Detection Per-Image Summary Builder")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(source_path)}")
        print(f"- exists: PASS")
        print(f"- valid_json: PASS")
        print(f"- source_hash: {source_hash}")
        print(f"- image_count: {source.get('image_count')}")
        print(f"- bbox_count: {source.get('bbox_count')}")
        print()
        print("## Summary Output")
        print(f"- output path: {_repo_relative(output_path)}")
        print(f"- image_count: {summary['image_count']}")
        print(f"- image_with_detections_count: {summary['image_with_detections_count']}")
        print(f"- image_without_detections_count: {summary['image_without_detections_count']}")
        print(f"- total_bbox_count: {summary['total_bbox_count']}")
        print("- registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("PASS")
        return 0
    except Exception as exc:
        print("# Detection Per-Image Summary Builder")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(source_path)}")
        print(f"- exists: {'PASS' if source_path.is_file() else 'FAIL'}")
        print(f"- valid_json: {'PASS' if source_path.is_file() else 'FAIL'}")
        print(f"- source_hash: {_sha256_file(source_path) if source_path.is_file() else ''}")
        print()
        print("## Summary Output")
        print(f"- output path: {_repo_relative(output_path)}")
        print("- registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("FAIL")
        print(f"failure_reason={exc}")
        return 1


def _build_summary_rows(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    summary_rows: list[dict[str, Any]] = []
    image_with_detections_count = 0
    image_without_detections_count = 0
    total_bbox_count = 0

    rows = source.get("prediction_rows", [])
    for row in rows:
        boxes = list(row.get("boxes", []))
        predicted_box_count = row.get("predicted_box_count")
        if predicted_box_count != len(boxes):
            raise ValueError(
                f"predicted_box_count mismatch for image_id={row.get('image_id')}"
            )

        confidences = [_coerce_confidence(box.get("confidence"), row.get("image_id")) for box in boxes]
        predicted_class_ids = [box.get("class_id") for box in boxes]
        predicted_class_labels = [box.get("class_label") for box in boxes]

        has_detections = predicted_box_count > 0
        if has_detections:
            image_with_detections_count += 1
        else:
            image_without_detections_count += 1

        total_bbox_count += predicted_box_count

        best_prediction = row.get("best_prediction")
        if not has_detections:
            if best_prediction is not None:
                raise ValueError(
                    f"best_prediction must be null when there are no detections for image_id={row.get('image_id')}"
                )
            max_confidence = None
            mean_confidence = None
        else:
            if best_prediction is None:
                raise ValueError(
                    f"best_prediction must be present when detections exist for image_id={row.get('image_id')}"
                )
            max_confidence = max(confidences)
            mean_confidence = sum(confidences) / len(confidences)

        summary_rows.append(
            {
                "image_id": row.get("image_id"),
                "image_path": row.get("image_path"),
                "image_width": row.get("image_width"),
                "image_height": row.get("image_height"),
                "predicted_box_count": predicted_box_count,
                "has_detections": has_detections,
                "defect_count": row.get("defect_count"),
                "best_prediction": best_prediction,
                "max_confidence": max_confidence,
                "mean_confidence": mean_confidence,
                "predicted_class_ids": predicted_class_ids,
                "predicted_class_labels": predicted_class_labels,
                "warnings": list(row.get("warnings", [])),
                "errors": list(row.get("errors", [])),
            }
        )

    return summary_rows, {
        "image_with_detections_count": image_with_detections_count,
        "image_without_detections_count": image_without_detections_count,
        "total_bbox_count": total_bbox_count,
    }


def _build_summary(
    source: dict[str, Any],
    source_hash: str,
    summary_rows: list[dict[str, Any]],
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "artifact_type": "detection_per_image_summary",
        "track_id": EXPECTED_TRACK_ID,
        "task_type": EXPECTED_TASK_TYPE,
        "run_id": EXPECTED_RUN_ID,
        "run_config_id": EXPECTED_RUN_CONFIG_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "model_type": EXPECTED_MODEL_TYPE,
        "model_version": EXPECTED_MODEL_VERSION,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split": EXPECTED_SPLIT,
        "source_bbox_prediction_artifact_path": _repo_relative(SOURCE_BBOX_PREDICTION_PATH),
        "source_bbox_prediction_artifact_hash": source_hash,
        "created_at": _utc_now_iso(),
        "image_count": source.get("image_count"),
        "image_with_detections_count": counts["image_with_detections_count"],
        "image_without_detections_count": counts["image_without_detections_count"],
        "total_bbox_count": counts["total_bbox_count"],
        "summary_rows": summary_rows,
    }


def _validate_source(source: dict[str, Any], source_path: Path) -> None:
    required_fields = {
        "artifact_type",
        "track_id",
        "task_type",
        "run_id",
        "run_config_id",
        "model_name",
        "model_type",
        "model_version",
        "dataset_id",
        "dataset_version",
        "split",
        "created_at",
        "source_artifact_paths",
        "prediction_parameters",
        "image_count",
        "prediction_count",
        "bbox_count",
        "prediction_rows",
    }
    missing = [field for field in required_fields if field not in source]
    if missing:
        raise ValueError(f"source bbox prediction artifact missing required fields: {missing}")
    if source.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("source bbox prediction artifact track_id mismatch.")
    if source.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("source bbox prediction artifact task_type mismatch.")
    if source.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("source bbox prediction artifact run_id mismatch.")
    if source.get("run_config_id") != EXPECTED_RUN_CONFIG_ID:
        raise ValueError("source bbox prediction artifact run_config_id mismatch.")
    if source.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("source bbox prediction artifact model_name mismatch.")
    if source.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("source bbox prediction artifact model_type mismatch.")
    if source.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("source bbox prediction artifact model_version mismatch.")
    if source.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("source bbox prediction artifact dataset_id mismatch.")
    if source.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("source bbox prediction artifact dataset_version mismatch.")
    if source.get("split") != EXPECTED_SPLIT:
        raise ValueError("source bbox prediction artifact split mismatch.")
    if source.get("image_count") != len(source.get("prediction_rows", [])):
        raise ValueError("source bbox prediction artifact image_count mismatch.")
    if source.get("prediction_count") != len(source.get("prediction_rows", [])):
        raise ValueError("source bbox prediction artifact prediction_count mismatch.")
    if source.get("bbox_count") != _sum_boxes(source.get("prediction_rows", [])):
        raise ValueError("source bbox prediction artifact bbox_count mismatch.")
    if source.get("bbox_count") != 573:
        raise ValueError("source bbox prediction artifact bbox_count must be 573.")
    if not source_path.is_file():
        raise FileNotFoundError(f"source bbox prediction artifact not found: {_repo_relative(source_path)}")


def _validate_summary(summary: dict[str, Any], source: dict[str, Any]) -> None:
    required_fields = {
        "artifact_type",
        "track_id",
        "task_type",
        "run_id",
        "run_config_id",
        "model_name",
        "model_type",
        "model_version",
        "dataset_id",
        "dataset_version",
        "split",
        "source_bbox_prediction_artifact_path",
        "source_bbox_prediction_artifact_hash",
        "created_at",
        "image_count",
        "image_with_detections_count",
        "image_without_detections_count",
        "total_bbox_count",
        "summary_rows",
    }
    missing = [field for field in required_fields if field not in summary]
    if missing:
        raise ValueError(f"summary missing required fields: {missing}")
    if summary.get("image_count") != len(summary.get("summary_rows", [])):
        raise ValueError("summary image_count must match number of summary_rows.")
    if summary.get("image_with_detections_count") + summary.get("image_without_detections_count") != summary.get("image_count"):
        raise ValueError("summary detection counts must sum to image_count.")
    if summary.get("total_bbox_count") != source.get("bbox_count"):
        raise ValueError("summary total_bbox_count mismatch with source bbox_count.")

    for row in summary.get("summary_rows", []):
        if row.get("predicted_box_count") != len(row.get("predicted_class_ids", [])):
            raise ValueError(f"summary row predicted_box_count mismatch for image_id={row.get('image_id')}")
        if row.get("predicted_box_count") != len(row.get("predicted_class_labels", [])):
            raise ValueError(f"summary row predicted_box_count mismatch for labels image_id={row.get('image_id')}")
        if row.get("predicted_box_count") == 0:
            if row.get("best_prediction") is not None:
                raise ValueError(f"summary best_prediction must be null for empty rows image_id={row.get('image_id')}")
            if row.get("max_confidence") is not None:
                raise ValueError(f"summary max_confidence must be null for empty rows image_id={row.get('image_id')}")
            if row.get("mean_confidence") is not None:
                raise ValueError(f"summary mean_confidence must be null for empty rows image_id={row.get('image_id')}")
            continue

        if row.get("best_prediction") is None:
            raise ValueError(f"summary best_prediction must be present for populated rows image_id={row.get('image_id')}")
        if not isinstance(row.get("max_confidence"), (float, int)):
            raise ValueError(f"summary max_confidence must be numeric for image_id={row.get('image_id')}")
        if not isinstance(row.get("mean_confidence"), (float, int)):
            raise ValueError(f"summary mean_confidence must be numeric for image_id={row.get('image_id')}")


def _validate_written_output(written: dict[str, Any], source: dict[str, Any]) -> None:
    _validate_summary(written, source)
    if written.get("source_bbox_prediction_artifact_path") != _repo_relative(SOURCE_BBOX_PREDICTION_PATH):
        raise ValueError("written summary source artifact path mismatch.")
    if written.get("source_bbox_prediction_artifact_hash") != _sha256_file(SOURCE_BBOX_PREDICTION_PATH):
        raise ValueError("written summary source hash mismatch.")


def _coerce_confidence(value: Any, image_id: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"confidence must be numeric for image_id={image_id}")
    if value < 0 or value > 1:
        raise ValueError(f"confidence must be between 0 and 1 for image_id={image_id}")
    return float(value)


def _sum_boxes(rows: list[dict[str, Any]]) -> int:
    return sum(len(row.get("boxes", [])) for row in rows)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {_repo_relative(path)}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
