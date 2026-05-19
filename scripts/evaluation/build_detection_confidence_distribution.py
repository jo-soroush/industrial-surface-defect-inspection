"""Build a governed Detection/YOLO confidence distribution summary."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_BBOX_PREDICTION_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
)
SOURCE_PER_IMAGE_SUMMARY_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_per_image_summary__yolo_train_v0_2_0__validation.json"
)
OUTPUT_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_confidence_distribution__yolo_train_v0_2_0__validation.json"
)
EXPECTED_RUN_ID = "yolo_train_v0_2_0"
EXPECTED_RUN_CONFIG_ID = "yolo_train_v0_2_0"
EXPECTED_TRACK_ID = "detection"
EXPECTED_TASK_TYPE = "object_detection"
EXPECTED_MODEL_NAME = "yolo"
EXPECTED_MODEL_TYPE = "yolo"
EXPECTED_MODEL_VERSION = "0.2.0"
EXPECTED_DATASET_ID = "gc10det_detection"
EXPECTED_DATASET_VERSION = "gc10det_1.0"
EXPECTED_SPLIT = "validation"

CONFIDENCE_BINS = [
    {
        "label": "0.00-0.25",
        "lower_bound": 0.0,
        "upper_bound": 0.25,
        "include_lower": True,
        "include_upper": False,
    },
    {
        "label": "0.25-0.50",
        "lower_bound": 0.25,
        "upper_bound": 0.5,
        "include_lower": True,
        "include_upper": False,
    },
    {
        "label": "0.50-0.75",
        "lower_bound": 0.5,
        "upper_bound": 0.75,
        "include_lower": True,
        "include_upper": False,
    },
    {
        "label": "0.75-1.00",
        "lower_bound": 0.75,
        "upper_bound": 1.0,
        "include_lower": True,
        "include_upper": True,
    },
]


def main() -> int:
    try:
        bbox_exists = SOURCE_BBOX_PREDICTION_PATH.is_file()
        summary_exists = SOURCE_PER_IMAGE_SUMMARY_PATH.is_file()
        bbox_size_bytes = SOURCE_BBOX_PREDICTION_PATH.stat().st_size if bbox_exists else 0
        summary_size_bytes = SOURCE_PER_IMAGE_SUMMARY_PATH.stat().st_size if summary_exists else 0
        bbox_hash = _sha256_file(SOURCE_BBOX_PREDICTION_PATH) if bbox_exists else ""
        summary_hash = _sha256_file(SOURCE_PER_IMAGE_SUMMARY_PATH) if summary_exists else ""

        if not bbox_exists:
            raise FileNotFoundError(
                f"source bbox prediction artifact not found: {_repo_relative(SOURCE_BBOX_PREDICTION_PATH)}"
            )
        if not summary_exists:
            raise FileNotFoundError(
                f"source per-image summary artifact not found: {_repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH)}"
            )

        bbox_payload = _load_json(SOURCE_BBOX_PREDICTION_PATH, "bbox prediction artifact")
        summary_payload = _load_json(SOURCE_PER_IMAGE_SUMMARY_PATH, "per-image summary artifact")

        _validate_bbox_payload(bbox_payload)
        _validate_summary_payload(summary_payload)
        _validate_cross_artifacts(bbox_payload, summary_payload)

        confidences = _extract_confidences(bbox_payload)
        if len(confidences) != bbox_payload["bbox_count"]:
            raise ValueError("confidence count mismatch with bbox_count.")

        bins = _build_bins(confidences)
        class_summary = _build_class_summary(bbox_payload)
        global_summary = _build_global_summary(confidences)

        confidence_distribution = {
            "artifact_type": "detection_confidence_distribution",
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
            "source_bbox_prediction_artifact_hash": bbox_hash,
            "source_per_image_summary_artifact_path": _repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH),
            "source_per_image_summary_artifact_hash": summary_hash,
            "created_at": _utc_now_iso(),
            "image_count": bbox_payload["image_count"],
            "image_with_detections_count": summary_payload["image_with_detections_count"],
            "image_without_detections_count": summary_payload["image_without_detections_count"],
            "total_bbox_count": bbox_payload["bbox_count"],
            "confidence_bin_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
            "confidence_bins": bins,
            "class_confidence_summary": class_summary,
            "global_confidence_summary": global_summary,
        }

        _validate_output(confidence_distribution, bbox_payload, summary_payload)
        _write_json_atomic(OUTPUT_PATH, confidence_distribution)

        written = _load_json(OUTPUT_PATH, "written confidence distribution artifact")
        _validate_written_output(written, bbox_payload, summary_payload)

        print("# Detection Confidence Distribution Builder")
        print()
        print("## Source Artifacts")
        print(f"- bbox prediction path: {_repo_relative(SOURCE_BBOX_PREDICTION_PATH)}")
        print(f"- bbox exists: {'PASS' if bbox_exists else 'FAIL'}")
        print(f"- bbox valid_json: PASS")
        print(f"- bbox hash: {bbox_hash}")
        print(f"- per-image summary path: {_repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH)}")
        print(f"- per-image exists: {'PASS' if summary_exists else 'FAIL'}")
        print(f"- per-image valid_json: PASS")
        print(f"- per-image hash: {summary_hash}")
        print()
        print("## Confidence Summary")
        print(f"- image_count: {confidence_distribution['image_count']}")
        print(f"- total_bbox_count: {confidence_distribution['total_bbox_count']}")
        print(f"- confidence_count: {global_summary['confidence_count']}")
        print(f"- low_confidence_count: {global_summary['low_confidence_count']}")
        print(f"- medium_confidence_count: {global_summary['medium_confidence_count']}")
        print(f"- high_confidence_count: {global_summary['high_confidence_count']}")
        print()
        print("## Output")
        print(f"- output path: {_repo_relative(OUTPUT_PATH)}")
        print("- registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("PASS")
        return 0
    except Exception as exc:
        print("# Detection Confidence Distribution Builder")
        print()
        print("## Source Artifacts")
        print(f"- bbox prediction path: {_repo_relative(SOURCE_BBOX_PREDICTION_PATH)}")
        print(f"- bbox exists: {'PASS' if SOURCE_BBOX_PREDICTION_PATH.is_file() else 'FAIL'}")
        print(f"- bbox valid_json: {'PASS' if SOURCE_BBOX_PREDICTION_PATH.is_file() else 'FAIL'}")
        print(
            f"- bbox hash: {_sha256_file(SOURCE_BBOX_PREDICTION_PATH) if SOURCE_BBOX_PREDICTION_PATH.is_file() else ''}"
        )
        print(f"- per-image summary path: {_repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH)}")
        print(f"- per-image exists: {'PASS' if SOURCE_PER_IMAGE_SUMMARY_PATH.is_file() else 'FAIL'}")
        print(f"- per-image valid_json: {'PASS' if SOURCE_PER_IMAGE_SUMMARY_PATH.is_file() else 'FAIL'}")
        print(
            f"- per-image hash: {_sha256_file(SOURCE_PER_IMAGE_SUMMARY_PATH) if SOURCE_PER_IMAGE_SUMMARY_PATH.is_file() else ''}"
        )
        print()
        print("## Confidence Summary")
        print("- image_count: fail")
        print("- total_bbox_count: fail")
        print("- confidence_count: fail")
        print("- low_confidence_count: fail")
        print("- medium_confidence_count: fail")
        print("- high_confidence_count: fail")
        print()
        print("## Output")
        print(f"- output path: {_repo_relative(OUTPUT_PATH)}")
        print("- registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("FAIL")
        print(f"failure_reason={exc}")
        return 1


def _validate_bbox_payload(payload: dict[str, Any]) -> None:
    required = {
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
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"bbox prediction artifact missing required fields: {missing}")
    if payload.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("bbox prediction artifact track_id mismatch.")
    if payload.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("bbox prediction artifact task_type mismatch.")
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("bbox prediction artifact run_id mismatch.")
    if payload.get("run_config_id") != EXPECTED_RUN_CONFIG_ID:
        raise ValueError("bbox prediction artifact run_config_id mismatch.")
    if payload.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("bbox prediction artifact model_name mismatch.")
    if payload.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("bbox prediction artifact model_type mismatch.")
    if payload.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("bbox prediction artifact model_version mismatch.")
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("bbox prediction artifact dataset_id mismatch.")
    if payload.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("bbox prediction artifact dataset_version mismatch.")
    if payload.get("split") != EXPECTED_SPLIT:
        raise ValueError("bbox prediction artifact split mismatch.")
    if payload.get("image_count") != len(payload.get("prediction_rows", [])):
        raise ValueError("bbox prediction artifact image_count mismatch.")
    if payload.get("prediction_count") != len(payload.get("prediction_rows", [])):
        raise ValueError("bbox prediction artifact prediction_count mismatch.")
    if payload.get("bbox_count") != _sum_boxes(payload.get("prediction_rows", [])):
        raise ValueError("bbox prediction artifact bbox_count mismatch.")


def _validate_summary_payload(payload: dict[str, Any]) -> None:
    required = {
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
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"per-image summary artifact missing required fields: {missing}")
    if payload.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("per-image summary artifact track_id mismatch.")
    if payload.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("per-image summary artifact task_type mismatch.")
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("per-image summary artifact run_id mismatch.")
    if payload.get("split") != EXPECTED_SPLIT:
        raise ValueError("per-image summary artifact split mismatch.")


def _validate_cross_artifacts(bbox_payload: dict[str, Any], summary_payload: dict[str, Any]) -> None:
    if bbox_payload["image_count"] != summary_payload["image_count"]:
        raise ValueError("bbox and summary image_count mismatch.")
    if bbox_payload["bbox_count"] != summary_payload["total_bbox_count"]:
        raise ValueError("bbox and summary total bbox count mismatch.")


def _extract_confidences(bbox_payload: dict[str, Any]) -> list[float]:
    confidences: list[float] = []
    for row in bbox_payload.get("prediction_rows", []):
        for box in row.get("boxes", []):
            confidence = box.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError(f"confidence must be numeric for image_id={row.get('image_id')}")
            if confidence < 0 or confidence > 1:
                raise ValueError(f"confidence must be between 0 and 1 for image_id={row.get('image_id')}")
            confidences.append(float(confidence))
    return confidences


def _build_bins(confidences: list[float]) -> list[dict[str, Any]]:
    total = len(confidences)
    bin_counts = [0, 0, 0, 0]
    for confidence in confidences:
        if 0.0 <= confidence < 0.25:
            bin_counts[0] += 1
        elif 0.25 <= confidence < 0.5:
            bin_counts[1] += 1
        elif 0.5 <= confidence < 0.75:
            bin_counts[2] += 1
        elif 0.75 <= confidence <= 1.0:
            bin_counts[3] += 1
        else:
            raise ValueError(f"confidence out of range: {confidence}")

    bins: list[dict[str, Any]] = []
    for template, count in zip(CONFIDENCE_BINS, bin_counts, strict=True):
        bins.append(
            {
                **template,
                "count": count,
                "percentage": (count / total * 100.0) if total else 0.0,
            }
        )
    return bins


def _build_global_summary(confidences: list[float]) -> dict[str, Any]:
    if not confidences:
        raise ValueError("confidence distribution requires at least one confidence value.")

    low_count = sum(1 for confidence in confidences if confidence < 0.5)
    medium_count = sum(1 for confidence in confidences if 0.5 <= confidence < 0.75)
    high_count = sum(1 for confidence in confidences if confidence >= 0.75)

    return {
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
        "mean_confidence": mean(confidences),
        "median_confidence": median(confidences),
        "confidence_count": len(confidences),
        "low_confidence_count": low_count,
        "medium_confidence_count": medium_count,
        "high_confidence_count": high_count,
    }


def _build_class_summary(bbox_payload: dict[str, Any]) -> list[dict[str, Any]]:
    per_class: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in bbox_payload.get("prediction_rows", []):
        for box in row.get("boxes", []):
            class_id = box.get("class_id")
            class_label = box.get("class_label")
            confidence = box.get("confidence")
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ValueError(f"class_id must be an integer for image_id={row.get('image_id')}")
            if not isinstance(class_label, str) or not class_label.strip():
                raise ValueError(f"class_label must be a non-empty string for image_id={row.get('image_id')}")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError(f"confidence must be numeric for image_id={row.get('image_id')}")
            if confidence < 0 or confidence > 1:
                raise ValueError(f"confidence must be between 0 and 1 for image_id={row.get('image_id')}")
            per_class[(int(class_id), class_label)].append(float(confidence))

    class_summary: list[dict[str, Any]] = []
    for (class_id, class_label), confidences in sorted(per_class.items(), key=lambda item: item[0][0]):
        bin_counts = _bin_counts(confidences)
        class_summary.append(
            {
                "class_id": class_id,
                "class_label": class_label,
                "bbox_count": len(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences),
                "mean_confidence": mean(confidences),
                "median_confidence": median(confidences),
                "bin_counts": [
                    {
                        "label": template["label"],
                        "count": count,
                        "percentage": (count / len(confidences) * 100.0) if confidences else 0.0,
                    }
                    for template, count in zip(CONFIDENCE_BINS, bin_counts, strict=True)
                ],
            }
        )
    return class_summary


def _bin_counts(confidences: list[float]) -> list[int]:
    counts = [0, 0, 0, 0]
    for confidence in confidences:
        if 0.0 <= confidence < 0.25:
            counts[0] += 1
        elif 0.25 <= confidence < 0.5:
            counts[1] += 1
        elif 0.5 <= confidence < 0.75:
            counts[2] += 1
        elif 0.75 <= confidence <= 1.0:
            counts[3] += 1
        else:
            raise ValueError(f"confidence out of range: {confidence}")
    return counts


def _validate_output(
    payload: dict[str, Any],
    bbox_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> None:
    required = {
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
        "source_per_image_summary_artifact_path",
        "source_per_image_summary_artifact_hash",
        "created_at",
        "image_count",
        "image_with_detections_count",
        "image_without_detections_count",
        "total_bbox_count",
        "confidence_bin_edges",
        "confidence_bins",
        "class_confidence_summary",
        "global_confidence_summary",
    }
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"confidence distribution missing required fields: {missing}")
    if payload.get("image_count") != bbox_payload.get("image_count"):
        raise ValueError("confidence distribution image_count mismatch.")
    if payload.get("total_bbox_count") != bbox_payload.get("bbox_count"):
        raise ValueError("confidence distribution total_bbox_count mismatch.")
    summary = payload.get("global_confidence_summary", {})
    if summary.get("confidence_count") != payload.get("total_bbox_count"):
        raise ValueError("confidence_count must match total_bbox_count.")
    if sum(item.get("count", 0) for item in payload.get("confidence_bins", [])) != payload.get("total_bbox_count"):
        raise ValueError("confidence bin counts must sum to total_bbox_count.")
    if sum(item.get("bbox_count", 0) for item in payload.get("class_confidence_summary", [])) != payload.get("total_bbox_count"):
        raise ValueError("class bbox counts must sum to total_bbox_count.")
    if payload.get("image_with_detections_count") != summary_payload.get("image_with_detections_count"):
        raise ValueError("image_with_detections_count mismatch with source summary.")
    if payload.get("image_without_detections_count") != summary_payload.get("image_without_detections_count"):
        raise ValueError("image_without_detections_count mismatch with source summary.")
    if summary.get("low_confidence_count", 0) + summary.get("medium_confidence_count", 0) + summary.get("high_confidence_count", 0) != payload.get("total_bbox_count"):
        raise ValueError("global confidence band counts must sum to total_bbox_count.")


def _validate_written_output(
    payload: dict[str, Any],
    bbox_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> None:
    _validate_output(payload, bbox_payload, summary_payload)
    if payload.get("source_bbox_prediction_artifact_hash") != _sha256_file(SOURCE_BBOX_PREDICTION_PATH):
        raise ValueError("written confidence distribution source bbox hash mismatch.")
    if payload.get("source_per_image_summary_artifact_hash") != _sha256_file(SOURCE_PER_IMAGE_SUMMARY_PATH):
        raise ValueError("written confidence distribution source per-image hash mismatch.")


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


def _sum_boxes(rows: list[dict[str, Any]]) -> int:
    return sum(len(row.get("boxes", [])) for row in rows)


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
