"""Build a governed inventory for the YOLO confidence distribution artifact."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIDENCE_DISTRIBUTION_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_confidence_distribution__yolo_train_v0_2_0__validation.json"
)
SOURCE_BBOX_PREDICTION_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
)
SOURCE_PER_IMAGE_SUMMARY_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_per_image_summary__yolo_train_v0_2_0__validation.json"
)
OUTPUT_PATH = REPO_ROOT / (
    "artifacts/models/inventory/"
    "track_detection_confidence_distribution_artifact_inventory__yolo_train_v0_2_0__validation.json"
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
EXPECTED_IMAGE_COUNT = 345
EXPECTED_TOTAL_BBOX_COUNT = 573
EXPECTED_CONFIDENCE_BIN_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0]
EXPECTED_CONFIDENCE_BIN_LABELS = [
    "0.00-0.25",
    "0.25-0.50",
    "0.50-0.75",
    "0.75-1.00",
]
EXPECTED_SOURCE_ARTIFACT_PATHS = [
    SOURCE_CONFIDENCE_DISTRIBUTION_PATH.relative_to(REPO_ROOT).as_posix(),
    SOURCE_BBOX_PREDICTION_PATH.relative_to(REPO_ROOT).as_posix(),
    SOURCE_PER_IMAGE_SUMMARY_PATH.relative_to(REPO_ROOT).as_posix(),
]

REQUIRED_TOP_LEVEL_FIELDS = [
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
]

PRODUCTION_TERMS = ("production-ready", "deployment-safe")


def main() -> int:
    try:
        source_exists = SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file()
        source_size_bytes = SOURCE_CONFIDENCE_DISTRIBUTION_PATH.stat().st_size if source_exists else 0
        source_sha256 = _sha256_file(SOURCE_CONFIDENCE_DISTRIBUTION_PATH) if source_exists else ""
        source_valid_json = False
        source_payload: dict[str, Any] | None = None
        bbox_payload: dict[str, Any] | None = None
        summary_payload: dict[str, Any] | None = None
        source_artifact_paths: list[str] = []
        validation_checks: list[dict[str, Any]] = []

        if not source_exists:
            validation_checks.append(
                _check(
                    "source_confidence_distribution_artifact_exists",
                    "FAIL",
                    "source confidence distribution artifact is missing.",
                )
            )
            raise FileNotFoundError(
                f"source confidence distribution artifact not found: {_repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH)}"
            )

        source_payload = _load_json(SOURCE_CONFIDENCE_DISTRIBUTION_PATH, "confidence distribution artifact")
        source_valid_json = True

        bbox_payload = _load_json(SOURCE_BBOX_PREDICTION_PATH, "bbox prediction artifact")
        summary_payload = _load_json(SOURCE_PER_IMAGE_SUMMARY_PATH, "per-image summary artifact")

        bbox_sha256 = _sha256_file(SOURCE_BBOX_PREDICTION_PATH)
        summary_sha256 = _sha256_file(SOURCE_PER_IMAGE_SUMMARY_PATH)
        bbox_size_bytes = SOURCE_BBOX_PREDICTION_PATH.stat().st_size
        summary_size_bytes = SOURCE_PER_IMAGE_SUMMARY_PATH.stat().st_size
        source_artifact_paths = _collect_source_artifact_paths()

        _validate_bbox_payload(bbox_payload)
        _validate_summary_payload(summary_payload)
        _validate_confidence_payload(source_payload, bbox_payload, summary_payload)

        confidence_bins = source_payload["confidence_bins"]
        class_confidence_summary = source_payload["class_confidence_summary"]
        global_confidence_summary = source_payload["global_confidence_summary"]
        confidence_bin_count_sum = sum(int(item["count"]) for item in confidence_bins)
        class_bbox_count_sum = sum(int(item["bbox_count"]) for item in class_confidence_summary)
        confidence_count = int(global_confidence_summary["confidence_count"])
        low_confidence_count = int(global_confidence_summary["low_confidence_count"])
        medium_confidence_count = int(global_confidence_summary["medium_confidence_count"])
        high_confidence_count = int(global_confidence_summary["high_confidence_count"])

        validation_checks.extend(
            [
                _check(
                    "source_confidence_distribution_artifact_exists",
                    "PASS",
                    _repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH),
                ),
                _check(
                    "source_confidence_distribution_artifact_valid_json",
                    "PASS",
                    "source confidence distribution artifact parsed successfully.",
                ),
                _check("required_top_level_fields_present", "PASS", "all required top-level fields are present."),
                _check(
                    "track_id_matches",
                    _status(source_payload.get("track_id") == EXPECTED_TRACK_ID),
                    f"track_id={source_payload.get('track_id')}",
                ),
                _check(
                    "task_type_matches",
                    _status(source_payload.get("task_type") == EXPECTED_TASK_TYPE),
                    f"task_type={source_payload.get('task_type')}",
                ),
                _check(
                    "run_id_matches",
                    _status(source_payload.get("run_id") == EXPECTED_RUN_ID),
                    f"run_id={source_payload.get('run_id')}",
                ),
                _check(
                    "split_matches",
                    _status(source_payload.get("split") == EXPECTED_SPLIT),
                    f"split={source_payload.get('split')}",
                ),
                _check(
                    "image_count_matches_expected",
                    _status(source_payload.get("image_count") == EXPECTED_IMAGE_COUNT),
                    f"image_count={source_payload.get('image_count')}, expected={EXPECTED_IMAGE_COUNT}",
                ),
                _check(
                    "image_count_matches_bbox_and_summary",
                    _status(
                        source_payload.get("image_count") == bbox_payload.get("image_count") == summary_payload.get("image_count")
                    ),
                    f"confidence={source_payload.get('image_count')}, bbox={bbox_payload.get('image_count')}, summary={summary_payload.get('image_count')}",
                ),
                _check(
                    "total_bbox_count_matches_expected",
                    _status(source_payload.get("total_bbox_count") == EXPECTED_TOTAL_BBOX_COUNT),
                    f"total_bbox_count={source_payload.get('total_bbox_count')}, expected={EXPECTED_TOTAL_BBOX_COUNT}",
                ),
                _check(
                    "total_bbox_count_matches_supporting_artifacts",
                    _status(
                        source_payload.get("total_bbox_count")
                        == bbox_payload.get("bbox_count")
                        == summary_payload.get("total_bbox_count")
                    ),
                    f"confidence={source_payload.get('total_bbox_count')}, bbox={bbox_payload.get('bbox_count')}, summary={summary_payload.get('total_bbox_count')}",
                ),
                _check(
                    "source_bbox_artifact_hash_matches",
                    _status(source_payload.get("source_bbox_prediction_artifact_hash") == bbox_sha256),
                    f"hash={source_payload.get('source_bbox_prediction_artifact_hash')}",
                ),
                _check(
                    "source_per_image_summary_hash_matches",
                    _status(source_payload.get("source_per_image_summary_artifact_hash") == summary_sha256),
                    f"hash={source_payload.get('source_per_image_summary_artifact_hash')}",
                ),
                _check(
                    "source_artifact_paths_collected",
                    _status(_source_artifact_paths_match(source_artifact_paths)),
                    "source artifact paths match the governed source set.",
                ),
                _check(
                    "confidence_count_matches_total_bbox_count",
                    _status(confidence_count == source_payload.get("total_bbox_count")),
                    f"confidence_count={confidence_count}, total_bbox_count={source_payload.get('total_bbox_count')}",
                ),
                _check(
                    "confidence_bin_count_sum_matches_total_bbox_count",
                    _status(confidence_bin_count_sum == source_payload.get("total_bbox_count")),
                    f"bin_count_sum={confidence_bin_count_sum}",
                ),
                _check(
                    "class_bbox_count_sum_matches_total_bbox_count",
                    _status(class_bbox_count_sum == source_payload.get("total_bbox_count")),
                    f"class_bbox_count_sum={class_bbox_count_sum}",
                ),
                _check(
                    "global_confidence_band_counts_balance",
                    _status(
                        low_confidence_count + medium_confidence_count + high_confidence_count
                        == source_payload.get("total_bbox_count")
                    ),
                    f"low={low_confidence_count}, medium={medium_confidence_count}, high={high_confidence_count}",
                ),
                _check(
                    "confidence_bin_edges_match",
                    _status(source_payload.get("confidence_bin_edges") == EXPECTED_CONFIDENCE_BIN_EDGES),
                    f"confidence_bin_edges={source_payload.get('confidence_bin_edges')}",
                ),
                _check(
                    "confidence_bins_structure_valid",
                    _status(_confidence_bins_are_valid(confidence_bins)),
                    "confidence bins follow the governed four-bin layout.",
                ),
                _check(
                    "global_confidence_summary_numeric",
                    _status(_global_summary_is_valid(global_confidence_summary)),
                    "global confidence summary statistics are numeric and bounded.",
                ),
                _check(
                    "class_confidence_summary_numeric",
                    _status(_class_summary_is_valid(class_confidence_summary)),
                    "class confidence summary statistics are numeric and bounded.",
                ),
                _check(
                    "file_size_positive",
                    _status(source_size_bytes > 0),
                    f"size_bytes={source_size_bytes}",
                ),
                _check(
                    "sha256_computed",
                    _status(bool(source_sha256)),
                    f"sha256={source_sha256}",
                ),
                _check(
                    "no_production_or_deployment_claims",
                    _status(
                        not _contains_forbidden_terms(
                            {
                                "confidence_distribution": source_payload,
                                "bbox_prediction": bbox_payload,
                                "per_image_summary": summary_payload,
                            }
                        )
                    ),
                    "source artifacts contain no forbidden production or deployment wording.",
                ),
                _check("registry_update_deferred", "PASS", "registry publication is deferred to a later task."),
                _check("frontend_bundle_deferred", "PASS", "frontend bundle generation is deferred to a later task."),
                _check("notebook_update_deferred", "PASS", "notebook update is deferred to a later task."),
                _check(
                    "source_artifact_paths_collected",
                    _status(_source_artifact_paths_match(source_artifact_paths)),
                    "source artifact paths match the governed source set.",
                ),
            ]
        )

        first_failure = next((check for check in validation_checks if check["status"] == "FAIL"), None)
        if first_failure is not None:
            raise ValueError(f"{first_failure['name']} failed: {first_failure['details']}")

        inventory = {
            "inventory_type": "track_detection_confidence_distribution_artifact_inventory",
            "track_id": source_payload["track_id"],
            "task_type": source_payload["task_type"],
            "run_id": source_payload["run_id"],
            "run_config_id": source_payload["run_config_id"],
            "model_name": source_payload["model_name"],
            "model_type": source_payload["model_type"],
            "model_version": source_payload["model_version"],
            "dataset_id": source_payload["dataset_id"],
            "dataset_version": source_payload["dataset_version"],
            "split": source_payload["split"],
            "source_confidence_distribution_artifact_path": _repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH),
            "source_confidence_distribution_artifact_sha256": source_sha256,
            "source_confidence_distribution_artifact_size_bytes": source_size_bytes,
            "source_confidence_distribution_artifact_exists": source_exists,
            "source_confidence_distribution_artifact_valid_json": source_valid_json,
            "image_count": source_payload["image_count"],
            "total_bbox_count": source_payload["total_bbox_count"],
            "confidence_count": confidence_count,
            "low_confidence_count": low_confidence_count,
            "medium_confidence_count": medium_confidence_count,
            "high_confidence_count": high_confidence_count,
            "confidence_bin_count_sum": confidence_bin_count_sum,
            "class_bbox_count_sum": class_bbox_count_sum,
            "source_bbox_prediction_artifact_path": _repo_relative(SOURCE_BBOX_PREDICTION_PATH),
            "source_bbox_prediction_artifact_hash": bbox_sha256,
            "source_per_image_summary_artifact_path": _repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH),
            "source_per_image_summary_artifact_hash": summary_sha256,
            "source_artifact_paths": source_artifact_paths,
            "created_at": _utc_now_iso(),
            "inventory_status": "pass",
            "validation_checks": validation_checks,
            "known_limitations": [
                "This inventory is derivative confidence distribution evidence.",
                "It is not a training artifact.",
                "It is not a model-quality improvement artifact.",
                "It is not a production-ready claim.",
                "Registry update is deferred to a later explicit step.",
                "Frontend bundle generation is deferred to a later explicit step.",
                "Notebook update is deferred to a later explicit step.",
            ],
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(OUTPUT_PATH, inventory)

        written = _load_json(OUTPUT_PATH, "written confidence distribution inventory")
        _validate_written_output(written, source_payload, bbox_payload, summary_payload)

        print("# Detection Confidence Distribution Artifact Inventory Builder")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH)}")
        print(f"- exists: {'PASS' if source_exists else 'FAIL'}")
        print(f"- valid_json: {'PASS' if source_valid_json else 'FAIL'}")
        print(f"- size_bytes: {source_size_bytes}")
        print(f"- sha256: {source_sha256}")
        print()
        print("## Validation Checks")
        for check in validation_checks:
            print(f"- {check['name']}: {check['status']} ({check['details']})")
        print()
        print("## Inventory Output")
        print(f"- output path: {_repo_relative(OUTPUT_PATH)}")
        print(f"- inventory_status: {inventory['inventory_status']}")
        print("- registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("PASS")
        return 0
    except Exception as exc:
        print("# Detection Confidence Distribution Artifact Inventory Builder")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH)}")
        print(f"- exists: {'PASS' if SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file() else 'FAIL'}")
        print(f"- valid_json: {'PASS' if SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file() else 'FAIL'}")
        print(
            f"- size_bytes: {SOURCE_CONFIDENCE_DISTRIBUTION_PATH.stat().st_size if SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file() else 0}"
        )
        print(
            f"- sha256: {_sha256_file(SOURCE_CONFIDENCE_DISTRIBUTION_PATH) if SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file() else ''}"
        )
        print()
        print("## Validation Checks")
        print(f"- FAIL: {exc}")
        print()
        print("## Inventory Output")
        print(f"- output path: {_repo_relative(OUTPUT_PATH)}")
        print("- inventory_status: fail")
        print("- registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("FAIL")
        print(f"failure_reason={exc}")
        return 1


def _validate_confidence_payload(
    payload: dict[str, Any],
    bbox_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> None:
    required = set(REQUIRED_TOP_LEVEL_FIELDS)
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"confidence distribution artifact missing required fields: {missing}")
    if payload.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("confidence distribution artifact track_id mismatch.")
    if payload.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("confidence distribution artifact task_type mismatch.")
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("confidence distribution artifact run_id mismatch.")
    if payload.get("run_config_id") != EXPECTED_RUN_CONFIG_ID:
        raise ValueError("confidence distribution artifact run_config_id mismatch.")
    if payload.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("confidence distribution artifact model_name mismatch.")
    if payload.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("confidence distribution artifact model_type mismatch.")
    if payload.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("confidence distribution artifact model_version mismatch.")
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("confidence distribution artifact dataset_id mismatch.")
    if payload.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("confidence distribution artifact dataset_version mismatch.")
    if payload.get("split") != EXPECTED_SPLIT:
        raise ValueError("confidence distribution artifact split mismatch.")

    confidence_bins = payload.get("confidence_bins", [])
    if not isinstance(confidence_bins, list):
        raise ValueError("confidence_bins must be a list.")
    if not _confidence_bins_are_valid(confidence_bins):
        raise ValueError("confidence bins are malformed.")
    if payload.get("confidence_bin_edges") != EXPECTED_CONFIDENCE_BIN_EDGES:
        raise ValueError("confidence_bin_edges mismatch.")

    global_summary = payload.get("global_confidence_summary", {})
    if not _global_summary_is_valid(global_summary):
        raise ValueError("global confidence summary is malformed.")

    class_summary = payload.get("class_confidence_summary", [])
    if not _class_summary_is_valid(class_summary):
        raise ValueError("class confidence summary is malformed.")

    if payload.get("image_count") != EXPECTED_IMAGE_COUNT:
        raise ValueError("confidence distribution image_count mismatch.")
    if payload.get("total_bbox_count") != EXPECTED_TOTAL_BBOX_COUNT:
        raise ValueError("confidence distribution total_bbox_count mismatch.")
    if payload.get("image_count") != bbox_payload.get("image_count") or payload.get("image_count") != summary_payload.get("image_count"):
        raise ValueError("confidence distribution image_count mismatch with supporting artifacts.")
    if payload.get("total_bbox_count") != bbox_payload.get("bbox_count") or payload.get("total_bbox_count") != summary_payload.get("total_bbox_count"):
        raise ValueError("confidence distribution total_bbox_count mismatch with supporting artifacts.")
    if global_summary.get("confidence_count") != payload.get("total_bbox_count"):
        raise ValueError("confidence_count must match total_bbox_count.")
    if sum(int(item.get("count", 0)) for item in confidence_bins) != payload.get("total_bbox_count"):
        raise ValueError("confidence bin counts must sum to total_bbox_count.")
    if sum(int(item.get("bbox_count", 0)) for item in class_summary) != payload.get("total_bbox_count"):
        raise ValueError("class bbox counts must sum to total_bbox_count.")
    if (
        int(global_summary.get("low_confidence_count", -1))
        + int(global_summary.get("medium_confidence_count", -1))
        + int(global_summary.get("high_confidence_count", -1))
        != payload.get("total_bbox_count")
    ):
        raise ValueError("global confidence band counts must sum to total_bbox_count.")
    if payload.get("source_bbox_prediction_artifact_path") != _repo_relative(SOURCE_BBOX_PREDICTION_PATH):
        raise ValueError("confidence distribution source bbox path mismatch.")
    if payload.get("source_per_image_summary_artifact_path") != _repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH):
        raise ValueError("confidence distribution source per-image path mismatch.")
    if payload.get("source_bbox_prediction_artifact_hash") != _sha256_file(SOURCE_BBOX_PREDICTION_PATH):
        raise ValueError("confidence distribution source bbox hash mismatch.")
    if payload.get("source_per_image_summary_artifact_hash") != _sha256_file(SOURCE_PER_IMAGE_SUMMARY_PATH):
        raise ValueError("confidence distribution source per-image hash mismatch.")
    if _contains_forbidden_terms({"confidence_distribution": payload, "bbox_prediction": bbox_payload, "per_image_summary": summary_payload}):
        raise ValueError("source artifacts contain forbidden production or deployment wording.")


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
    if payload.get("image_count") != len(payload.get("summary_rows", [])):
        raise ValueError("per-image summary artifact image_count mismatch.")
    if payload.get("image_with_detections_count") + payload.get("image_without_detections_count") != payload.get("image_count"):
        raise ValueError("per-image summary artifact detection counts mismatch.")
    if payload.get("total_bbox_count") != _sum_predicted_boxes(payload.get("summary_rows", [])):
        raise ValueError("per-image summary artifact total bbox count mismatch.")


def _confidence_bins_are_valid(bins: list[dict[str, Any]]) -> bool:
    if len(bins) != len(EXPECTED_CONFIDENCE_BIN_LABELS):
        return False
    for index, bin_payload in enumerate(bins):
        expected_label = EXPECTED_CONFIDENCE_BIN_LABELS[index]
        expected_lower = EXPECTED_CONFIDENCE_BIN_EDGES[index]
        expected_upper = EXPECTED_CONFIDENCE_BIN_EDGES[index + 1]
        if bin_payload.get("label") != expected_label:
            return False
        if bin_payload.get("lower_bound") != expected_lower:
            return False
        if bin_payload.get("upper_bound") != expected_upper:
            return False
        if bin_payload.get("include_lower") is not True:
            return False
        if bin_payload.get("include_upper") is not (index == len(EXPECTED_CONFIDENCE_BIN_LABELS) - 1):
            return False
        if not isinstance(bin_payload.get("count"), int) or isinstance(bin_payload.get("count"), bool):
            return False
        if not _is_number(bin_payload.get("percentage")):
            return False
    return True


def _global_summary_is_valid(summary: dict[str, Any]) -> bool:
    required = {
        "min_confidence",
        "max_confidence",
        "mean_confidence",
        "median_confidence",
        "confidence_count",
        "low_confidence_count",
        "medium_confidence_count",
        "high_confidence_count",
    }
    if any(field not in summary for field in required):
        return False
    if not all(_is_number(summary[field]) for field in ("min_confidence", "max_confidence", "mean_confidence", "median_confidence")):
        return False
    if not isinstance(summary.get("confidence_count"), int) or isinstance(summary.get("confidence_count"), bool):
        return False
    if not isinstance(summary.get("low_confidence_count"), int) or isinstance(summary.get("low_confidence_count"), bool):
        return False
    if not isinstance(summary.get("medium_confidence_count"), int) or isinstance(summary.get("medium_confidence_count"), bool):
        return False
    if not isinstance(summary.get("high_confidence_count"), int) or isinstance(summary.get("high_confidence_count"), bool):
        return False
    return True


def _class_summary_is_valid(class_summary: list[dict[str, Any]]) -> bool:
    for class_payload in class_summary:
        required = {
            "class_id",
            "class_label",
            "bbox_count",
            "min_confidence",
            "max_confidence",
            "mean_confidence",
            "median_confidence",
            "bin_counts",
        }
        if any(field not in class_payload for field in required):
            return False
        if not isinstance(class_payload.get("class_id"), int) or isinstance(class_payload.get("class_id"), bool):
            return False
        if not isinstance(class_payload.get("class_label"), str) or not class_payload.get("class_label").strip():
            return False
        if not isinstance(class_payload.get("bbox_count"), int) or isinstance(class_payload.get("bbox_count"), bool):
            return False
        if not all(_is_number(class_payload[field]) for field in ("min_confidence", "max_confidence", "mean_confidence", "median_confidence")):
            return False
        bin_counts = class_payload.get("bin_counts", [])
        if not isinstance(bin_counts, list) or len(bin_counts) != len(EXPECTED_CONFIDENCE_BIN_LABELS):
            return False
        for index, bin_payload in enumerate(bin_counts):
            if bin_payload.get("label") != EXPECTED_CONFIDENCE_BIN_LABELS[index]:
                return False
            if not isinstance(bin_payload.get("count"), int) or isinstance(bin_payload.get("count"), bool):
                return False
            if not _is_number(bin_payload.get("percentage")):
                return False
    return True


def _source_artifact_paths_match(paths: list[str]) -> bool:
    return paths == EXPECTED_SOURCE_ARTIFACT_PATHS


def _validate_written_output(
    payload: dict[str, Any],
    source_payload: dict[str, Any],
    bbox_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> None:
    required = {
        "inventory_type",
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
        "source_confidence_distribution_artifact_path",
        "source_confidence_distribution_artifact_sha256",
        "source_confidence_distribution_artifact_size_bytes",
        "source_confidence_distribution_artifact_exists",
        "source_confidence_distribution_artifact_valid_json",
        "image_count",
        "total_bbox_count",
        "confidence_count",
        "low_confidence_count",
        "medium_confidence_count",
        "high_confidence_count",
        "confidence_bin_count_sum",
        "class_bbox_count_sum",
        "source_bbox_prediction_artifact_path",
        "source_bbox_prediction_artifact_hash",
        "source_per_image_summary_artifact_path",
        "source_per_image_summary_artifact_hash",
        "source_artifact_paths",
        "created_at",
        "inventory_status",
        "validation_checks",
        "known_limitations",
    }
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"written confidence distribution inventory missing required fields: {missing}")
    if payload.get("inventory_status") != "pass":
        raise ValueError("written confidence distribution inventory status must be pass.")
    if payload.get("source_confidence_distribution_artifact_sha256") != _sha256_file(SOURCE_CONFIDENCE_DISTRIBUTION_PATH):
        raise ValueError("written confidence distribution source hash mismatch.")
    if payload.get("source_confidence_distribution_artifact_size_bytes") != SOURCE_CONFIDENCE_DISTRIBUTION_PATH.stat().st_size:
        raise ValueError("written confidence distribution source size mismatch.")
    if payload.get("source_bbox_prediction_artifact_hash") != _sha256_file(SOURCE_BBOX_PREDICTION_PATH):
        raise ValueError("written confidence distribution source bbox hash mismatch.")
    if payload.get("source_per_image_summary_artifact_hash") != _sha256_file(SOURCE_PER_IMAGE_SUMMARY_PATH):
        raise ValueError("written confidence distribution source per-image hash mismatch.")
    if payload.get("image_count") != source_payload.get("image_count"):
        raise ValueError("written confidence distribution image_count mismatch.")
    if payload.get("total_bbox_count") != source_payload.get("total_bbox_count"):
        raise ValueError("written confidence distribution total_bbox_count mismatch.")
    if payload.get("confidence_count") != source_payload["global_confidence_summary"]["confidence_count"]:
        raise ValueError("written confidence distribution confidence_count mismatch.")
    if payload.get("low_confidence_count") != source_payload["global_confidence_summary"]["low_confidence_count"]:
        raise ValueError("written confidence distribution low confidence count mismatch.")
    if payload.get("medium_confidence_count") != source_payload["global_confidence_summary"]["medium_confidence_count"]:
        raise ValueError("written confidence distribution medium confidence count mismatch.")
    if payload.get("high_confidence_count") != source_payload["global_confidence_summary"]["high_confidence_count"]:
        raise ValueError("written confidence distribution high confidence count mismatch.")
    if payload.get("confidence_bin_count_sum") != sum(int(item["count"]) for item in source_payload["confidence_bins"]):
        raise ValueError("written confidence distribution bin count sum mismatch.")
    if payload.get("class_bbox_count_sum") != sum(int(item["bbox_count"]) for item in source_payload["class_confidence_summary"]):
        raise ValueError("written confidence distribution class bbox count sum mismatch.")
    if payload.get("source_bbox_prediction_artifact_path") != _repo_relative(SOURCE_BBOX_PREDICTION_PATH):
        raise ValueError("written confidence distribution source bbox path mismatch.")
    if payload.get("source_per_image_summary_artifact_path") != _repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH):
        raise ValueError("written confidence distribution source per-image path mismatch.")
    if payload.get("source_artifact_paths") != EXPECTED_SOURCE_ARTIFACT_PATHS:
        raise ValueError("written confidence distribution source artifact paths mismatch.")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def _collect_source_artifact_paths() -> list[str]:
    return list(EXPECTED_SOURCE_ARTIFACT_PATHS)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _contains_forbidden_terms(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(_contains_forbidden_terms(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_forbidden_terms(item) for item in payload)
    if isinstance(payload, str):
        lowered = payload.lower()
        return any(term in lowered for term in PRODUCTION_TERMS)
    return False


def _check(name: str, status: str, details: str) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


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


def _sum_predicted_boxes(rows: list[dict[str, Any]]) -> int:
    return sum(int(row.get("predicted_box_count", 0)) for row in rows)


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
