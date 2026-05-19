"""Build a governed sample gallery for the YOLO detection validation split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_BBOX_PREDICTION_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
)
SOURCE_PER_IMAGE_SUMMARY_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_per_image_summary__yolo_train_v0_2_0__validation.json"
)
SOURCE_CONFIDENCE_DISTRIBUTION_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_confidence_distribution__yolo_train_v0_2_0__validation.json"
)
OUTPUT_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_sample_gallery__yolo_train_v0_2_0__validation.json"
)

EXPECTED_ARTIFACT_TYPE = "detection_sample_gallery"
EXPECTED_TRACK_ID = "detection"
EXPECTED_TASK_TYPE = "object_detection"
EXPECTED_RUN_ID = "yolo_train_v0_2_0"
EXPECTED_RUN_CONFIG_ID = "yolo_train_v0_2_0"
EXPECTED_MODEL_NAME = "yolo"
EXPECTED_MODEL_TYPE = "yolo"
EXPECTED_MODEL_VERSION = "0.2.0"
EXPECTED_DATASET_ID = "gc10det_detection"
EXPECTED_DATASET_VERSION = "gc10det_1.0"
EXPECTED_SPLIT = "validation"
EXPECTED_SAMPLE_LIMIT = 5

REQUIRED_BBOX_FIELDS = {
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

REQUIRED_SUMMARY_FIELDS = {
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

REQUIRED_CONFIDENCE_FIELDS = {
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

CONFIDENCE_BANDS = (
    ("low", 0.0, 0.5),
    ("medium", 0.5, 0.75),
    ("high", 0.75, 1.0),
)

GALLERY_CATEGORIES = (
    {
        "category_id": "no_detection_examples",
        "category_label": "No detection examples",
        "selection_rule": "Rows where predicted_box_count == 0, sorted by image_id ascending.",
        "reason_selected": "No detections predicted.",
        "limit": EXPECTED_SAMPLE_LIMIT,
        "predicate": lambda row: row["predicted_box_count"] == 0,
        "sort_key": lambda row: (row["image_id"],),
    },
    {
        "category_id": "multi_detection_examples",
        "category_label": "Multiple detection examples",
        "selection_rule": (
            "Rows where predicted_box_count >= 3, sorted by predicted_box_count descending, "
            "then max_confidence descending, then image_id ascending."
        ),
        "reason_selected": "Multiple detections predicted.",
        "limit": EXPECTED_SAMPLE_LIMIT,
        "predicate": lambda row: row["predicted_box_count"] >= 3,
        "sort_key": lambda row: (
            -row["predicted_box_count"],
            -row["max_confidence"],
            row["image_id"],
        ),
    },
    {
        "category_id": "high_confidence_examples",
        "category_label": "High confidence examples",
        "selection_rule": (
            "Rows where predicted_box_count is 1 or 2 and max_confidence >= 0.75, "
            "sorted by max_confidence descending, then mean_confidence descending, then image_id ascending."
        ),
        "reason_selected": "High-confidence detection example.",
        "limit": EXPECTED_SAMPLE_LIMIT,
        "predicate": lambda row: row["predicted_box_count"] in {1, 2} and row["max_confidence"] >= 0.75,
        "sort_key": lambda row: (-row["max_confidence"], -row["mean_confidence"], row["image_id"]),
    },
    {
        "category_id": "medium_confidence_examples",
        "category_label": "Medium confidence examples",
        "selection_rule": (
            "Rows where predicted_box_count is 1 or 2 and 0.50 <= max_confidence < 0.75, "
            "sorted by max_confidence descending, then mean_confidence descending, then image_id ascending."
        ),
        "reason_selected": "Medium-confidence detection example.",
        "limit": EXPECTED_SAMPLE_LIMIT,
        "predicate": lambda row: row["predicted_box_count"] in {1, 2} and 0.5 <= row["max_confidence"] < 0.75,
        "sort_key": lambda row: (-row["max_confidence"], -row["mean_confidence"], row["image_id"]),
    },
    {
        "category_id": "low_confidence_examples",
        "category_label": "Low confidence examples",
        "selection_rule": (
            "Rows where predicted_box_count is 1 or 2 and max_confidence < 0.50, "
            "sorted by max_confidence ascending, then image_id ascending."
        ),
        "reason_selected": "Low-confidence detection example.",
        "limit": EXPECTED_SAMPLE_LIMIT,
        "predicate": lambda row: row["predicted_box_count"] in {1, 2} and row["max_confidence"] < 0.5,
        "sort_key": lambda row: (row["max_confidence"], row["image_id"]),
    },
    {
        "category_id": "representative_examples",
        "category_label": "Representative examples",
        "selection_rule": (
            "Remaining rows with detections, sorted by absolute distance from the global median mean_confidence, "
            "then absolute distance from the median predicted_box_count, then image_id ascending."
        ),
        "reason_selected": "Representative detection example closest to the global medians.",
        "limit": EXPECTED_SAMPLE_LIMIT,
        "predicate": lambda row: row["predicted_box_count"] > 0,
        "sort_key": None,
    },
)


def main() -> int:
    try:
        bbox_payload = _load_json(SOURCE_BBOX_PREDICTION_PATH, "bbox prediction artifact")
        summary_payload = _load_json(SOURCE_PER_IMAGE_SUMMARY_PATH, "per-image summary artifact")
        confidence_payload = _load_json(
            SOURCE_CONFIDENCE_DISTRIBUTION_PATH, "confidence distribution artifact"
        )

        bbox_hash = _sha256_file(SOURCE_BBOX_PREDICTION_PATH)
        summary_hash = _sha256_file(SOURCE_PER_IMAGE_SUMMARY_PATH)
        confidence_hash = _sha256_file(SOURCE_CONFIDENCE_DISTRIBUTION_PATH)

        _validate_bbox_payload(bbox_payload, bbox_hash)
        _validate_summary_payload(summary_payload, summary_hash)
        _validate_confidence_payload(confidence_payload, confidence_hash)
        _validate_cross_artifacts(bbox_payload, summary_payload, confidence_payload)

        summary_rows, summary_index = _index_summary_rows(summary_payload["summary_rows"])
        bbox_rows = _index_bbox_rows(bbox_payload["prediction_rows"])
        _validate_row_alignment(summary_rows, bbox_rows)

        representative_mean_median = _median(
            row["mean_confidence"] for row in summary_rows if row["predicted_box_count"] > 0
        )
        representative_box_median = _median(
            row["predicted_box_count"] for row in summary_rows if row["predicted_box_count"] > 0
        )

        categories, selected_ids = _build_categories(
            summary_rows=summary_rows,
            summary_index=summary_index,
            bbox_rows=bbox_rows,
            median_mean_confidence=representative_mean_median,
            median_predicted_box_count=representative_box_median,
        )

        gallery = {
            "artifact_type": EXPECTED_ARTIFACT_TYPE,
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
            "source_confidence_distribution_artifact_path": _repo_relative(
                SOURCE_CONFIDENCE_DISTRIBUTION_PATH
            ),
            "source_confidence_distribution_artifact_hash": confidence_hash,
            "source_artifact_paths": [
                _repo_relative(SOURCE_BBOX_PREDICTION_PATH),
                _repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH),
                _repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH),
            ],
            "created_at": _utc_now_iso(),
            "image_count": bbox_payload["image_count"],
            "total_bbox_count": bbox_payload["bbox_count"],
            "gallery_category_count": len(categories),
            "gallery_sample_count": sum(category["sample_count"] for category in categories),
            "categories": categories,
        }

        _validate_gallery_output(gallery, summary_rows, selected_ids)
        _write_json_atomic(OUTPUT_PATH, gallery)
        written = _load_json(OUTPUT_PATH, "written sample gallery artifact")
        _validate_written_output(written, summary_rows, selected_ids)

        print("# Detection Sample Gallery Builder")
        print()
        print("## Source Artifacts")
        print(f"- bbox prediction path: {_repo_relative(SOURCE_BBOX_PREDICTION_PATH)}")
        print(f"- bbox exists: {'PASS' if SOURCE_BBOX_PREDICTION_PATH.is_file() else 'FAIL'}")
        print(f"- bbox valid_json: PASS")
        print(f"- bbox hash: {bbox_hash}")
        print(f"- per-image summary path: {_repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH)}")
        print(f"- per-image exists: {'PASS' if SOURCE_PER_IMAGE_SUMMARY_PATH.is_file() else 'FAIL'}")
        print(f"- per-image valid_json: PASS")
        print(f"- per-image hash: {summary_hash}")
        print(f"- confidence distribution path: {_repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH)}")
        print(
            f"- confidence distribution exists: {'PASS' if SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file() else 'FAIL'}"
        )
        print("- confidence distribution valid_json: PASS")
        print(f"- confidence distribution hash: {confidence_hash}")
        print()
        print("## Gallery Summary")
        print(f"- image_count: {gallery['image_count']}")
        print(f"- total_bbox_count: {gallery['total_bbox_count']}")
        print(f"- gallery_category_count: {gallery['gallery_category_count']}")
        print(f"- gallery_sample_count: {gallery['gallery_sample_count']}")
        print(f"- sample limit per category: {EXPECTED_SAMPLE_LIMIT}")
        print()
        print("## Output")
        print(f"- output path: {_repo_relative(OUTPUT_PATH)}")
        print("- registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print("- annotated images created: NO")
        print()
        print("## Final Verdict")
        print("PASS")
        return 0
    except Exception as exc:
        print("# Detection Sample Gallery Builder")
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
        print(f"- confidence distribution path: {_repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH)}")
        print(
            f"- confidence distribution exists: {'PASS' if SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file() else 'FAIL'}"
        )
        print(
            f"- confidence distribution valid_json: {'PASS' if SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file() else 'FAIL'}"
        )
        print(
            f"- confidence distribution hash: {_sha256_file(SOURCE_CONFIDENCE_DISTRIBUTION_PATH) if SOURCE_CONFIDENCE_DISTRIBUTION_PATH.is_file() else ''}"
        )
        print()
        print("## Gallery Summary")
        print("- image_count: fail")
        print("- total_bbox_count: fail")
        print("- gallery_category_count: fail")
        print("- gallery_sample_count: fail")
        print(f"- sample limit per category: {EXPECTED_SAMPLE_LIMIT}")
        print()
        print("## Output")
        print(f"- output path: {_repo_relative(OUTPUT_PATH)}")
        print("- registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print("- annotated images created: NO")
        print()
        print("## Final Verdict")
        print("FAIL")
        print(f"failure_reason={exc}")
        return 1


def _build_categories(
    *,
    summary_rows: list[dict[str, Any]],
    summary_index: dict[str, int],
    bbox_rows: dict[str, dict[str, Any]],
    median_mean_confidence: float,
    median_predicted_box_count: float,
) -> tuple[list[dict[str, Any]], set[str]]:
    selected_ids: set[str] = set()
    categories: list[dict[str, Any]] = []

    for category_def in GALLERY_CATEGORIES:
        if category_def["category_id"] == "representative_examples":
            pool = [
                row
                for row in summary_rows
                if row["predicted_box_count"] > 0 and row["image_id"] not in selected_ids
            ]
            ordered = sorted(
                pool,
                key=lambda row: (
                    abs(float(row["mean_confidence"]) - median_mean_confidence),
                    abs(float(row["predicted_box_count"]) - median_predicted_box_count),
                    row["image_id"],
                ),
            )
        else:
            pool = [
                row
                for row in summary_rows
                if category_def["predicate"](row) and row["image_id"] not in selected_ids
            ]
            ordered = sorted(pool, key=category_def["sort_key"])

        chosen_rows = ordered[: category_def["limit"]]
        samples = [
            _build_sample(row, bbox_rows[row["image_id"]], summary_index[row["image_id"]], category_def["reason_selected"])
            for row in chosen_rows
        ]
        for sample in samples:
            selected_ids.add(sample["image_id"])

        categories.append(
            {
                "category_id": category_def["category_id"],
                "category_label": category_def["category_label"],
                "selection_rule": category_def["selection_rule"],
                "sample_count": len(samples),
                "samples": samples,
            }
        )

    return categories, selected_ids


def _build_sample(
    summary_row: dict[str, Any],
    bbox_row: dict[str, Any],
    summary_row_index: int,
    reason_selected: str,
) -> dict[str, Any]:
    if summary_row["image_id"] != bbox_row["image_id"]:
        raise ValueError(f"image_id mismatch for sample {summary_row['image_id']}")
    if summary_row["image_path"] != bbox_row["image_path"]:
        raise ValueError(f"image_path mismatch for sample {summary_row['image_id']}")
    if summary_row["image_width"] != bbox_row["image_width"]:
        raise ValueError(f"image_width mismatch for sample {summary_row['image_id']}")
    if summary_row["image_height"] != bbox_row["image_height"]:
        raise ValueError(f"image_height mismatch for sample {summary_row['image_id']}")
    if summary_row["best_prediction"] != bbox_row["best_prediction"]:
        raise ValueError(f"best_prediction mismatch for sample {summary_row['image_id']}")

    image_path = _repo_relative(Path(summary_row["image_path"]))
    sample = {
        "image_id": summary_row["image_id"],
        "image_path": image_path,
        "image_width": summary_row["image_width"],
        "image_height": summary_row["image_height"],
        "predicted_box_count": summary_row["predicted_box_count"],
        "has_detections": summary_row["has_detections"],
        "defect_count": summary_row["defect_count"],
        "best_prediction": bbox_row["best_prediction"],
        "max_confidence": summary_row["max_confidence"],
        "mean_confidence": summary_row["mean_confidence"],
        "predicted_class_labels": summary_row["predicted_class_labels"],
        "reason_selected": reason_selected,
        "source_summary_row_index": summary_row_index,
    }
    if not _repo_path_exists(image_path):
        raise FileNotFoundError(f"gallery sample image does not exist: {image_path}")
    return sample


def _validate_gallery_output(
    gallery: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    selected_ids: set[str],
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
        "source_confidence_distribution_artifact_path",
        "source_confidence_distribution_artifact_hash",
        "source_artifact_paths",
        "created_at",
        "image_count",
        "total_bbox_count",
        "gallery_category_count",
        "gallery_sample_count",
        "categories",
    }
    missing = [field for field in required if field not in gallery]
    if missing:
        raise ValueError(f"sample gallery missing required fields: {missing}")
    if gallery.get("artifact_type") != EXPECTED_ARTIFACT_TYPE:
        raise ValueError("sample gallery artifact_type mismatch.")
    if gallery.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("sample gallery track_id mismatch.")
    if gallery.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("sample gallery task_type mismatch.")
    if gallery.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("sample gallery run_id mismatch.")
    if gallery.get("run_config_id") != EXPECTED_RUN_CONFIG_ID:
        raise ValueError("sample gallery run_config_id mismatch.")
    if gallery.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("sample gallery model_name mismatch.")
    if gallery.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("sample gallery model_type mismatch.")
    if gallery.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("sample gallery model_version mismatch.")
    if gallery.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("sample gallery dataset_id mismatch.")
    if gallery.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("sample gallery dataset_version mismatch.")
    if gallery.get("split") != EXPECTED_SPLIT:
        raise ValueError("sample gallery split mismatch.")
    if gallery.get("image_count") != 345:
        raise ValueError("sample gallery image_count mismatch.")
    if gallery.get("total_bbox_count") != 573:
        raise ValueError("sample gallery total_bbox_count mismatch.")
    if gallery.get("gallery_category_count") != len(gallery.get("categories", [])):
        raise ValueError("gallery_category_count mismatch.")
    if gallery.get("gallery_sample_count") != sum(category.get("sample_count", 0) for category in gallery.get("categories", [])):
        raise ValueError("gallery_sample_count mismatch.")
    if gallery.get("source_artifact_paths") != [
        _repo_relative(SOURCE_BBOX_PREDICTION_PATH),
        _repo_relative(SOURCE_PER_IMAGE_SUMMARY_PATH),
        _repo_relative(SOURCE_CONFIDENCE_DISTRIBUTION_PATH),
    ]:
        raise ValueError("source_artifact_paths mismatch.")

    seen_ids: set[str] = set()
    for category in gallery.get("categories", []):
        if category.get("sample_count") != len(category.get("samples", [])):
            raise ValueError(f"sample_count mismatch for category {category.get('category_id')}")
        if category.get("sample_count", 0) > EXPECTED_SAMPLE_LIMIT:
            raise ValueError(f"category {category.get('category_id')} exceeds the sample limit.")
        for sample in category.get("samples", []):
            image_id = sample.get("image_id")
            if image_id in seen_ids:
                raise ValueError(f"duplicate image_id across categories: {image_id}")
            seen_ids.add(image_id)
            if image_id not in {row["image_id"] for row in summary_rows}:
                raise ValueError(f"selected sample not found in per-image summary: {image_id}")
            summary_row = summary_rows[sample["source_summary_row_index"]]
            if summary_row["image_id"] != image_id:
                raise ValueError(f"source_summary_row_index mismatch for {image_id}")
            if not isinstance(sample.get("image_path"), str) or not sample["image_path"]:
                raise ValueError(f"image_path must be a string for {image_id}")
            if not _repo_path_exists(sample["image_path"]):
                raise FileNotFoundError(f"gallery sample image does not exist: {sample['image_path']}")
            if sample.get("best_prediction") != summary_row.get("best_prediction"):
                raise ValueError(f"best_prediction mismatch for {image_id}")
    if seen_ids != selected_ids:
        raise ValueError("selected sample identity set mismatch.")


def _validate_written_output(
    gallery: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    selected_ids: set[str],
) -> None:
    _validate_gallery_output(gallery, summary_rows, selected_ids)
    if gallery.get("gallery_category_count") != len(gallery.get("categories", [])):
        raise ValueError("written gallery category count mismatch.")


def _validate_bbox_payload(payload: dict[str, Any], expected_hash: str) -> None:
    missing = [field for field in REQUIRED_BBOX_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"bbox prediction artifact missing required fields: {missing}")
    if payload.get("artifact_type") != "detection_bbox_predictions":
        raise ValueError("bbox prediction artifact_type mismatch.")
    if payload.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("bbox prediction track_id mismatch.")
    if payload.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("bbox prediction task_type mismatch.")
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("bbox prediction run_id mismatch.")
    if payload.get("run_config_id") != EXPECTED_RUN_CONFIG_ID:
        raise ValueError("bbox prediction run_config_id mismatch.")
    if payload.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("bbox prediction model_name mismatch.")
    if payload.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("bbox prediction model_type mismatch.")
    if payload.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("bbox prediction model_version mismatch.")
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("bbox prediction dataset_id mismatch.")
    if payload.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("bbox prediction dataset_version mismatch.")
    if payload.get("split") != EXPECTED_SPLIT:
        raise ValueError("bbox prediction split mismatch.")
    if payload.get("image_count") != len(payload.get("prediction_rows", [])):
        raise ValueError("bbox prediction image_count mismatch.")
    if payload.get("prediction_count") != len(payload.get("prediction_rows", [])):
        raise ValueError("bbox prediction prediction_count mismatch.")
    if payload.get("bbox_count") != _sum_boxes(payload.get("prediction_rows", [])):
        raise ValueError("bbox prediction bbox_count mismatch.")
    if payload.get("source_artifact_paths") != [
        _repo_relative(SOURCE_BBOX_PREDICTION_PATH),
    ] and not payload.get("source_artifact_paths"):
        raise ValueError("bbox prediction source_artifact_paths mismatch.")
    if _sha256_file(SOURCE_BBOX_PREDICTION_PATH) != expected_hash:
        raise ValueError("bbox prediction hash mismatch.")


def _validate_summary_payload(payload: dict[str, Any], expected_hash: str) -> None:
    missing = [field for field in REQUIRED_SUMMARY_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"per-image summary artifact missing required fields: {missing}")
    if payload.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("per-image summary track_id mismatch.")
    if payload.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("per-image summary task_type mismatch.")
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("per-image summary run_id mismatch.")
    if payload.get("run_config_id") != EXPECTED_RUN_CONFIG_ID:
        raise ValueError("per-image summary run_config_id mismatch.")
    if payload.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("per-image summary model_name mismatch.")
    if payload.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("per-image summary model_type mismatch.")
    if payload.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("per-image summary model_version mismatch.")
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("per-image summary dataset_id mismatch.")
    if payload.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("per-image summary dataset_version mismatch.")
    if payload.get("split") != EXPECTED_SPLIT:
        raise ValueError("per-image summary split mismatch.")
    if payload.get("image_count") != len(payload.get("summary_rows", [])):
        raise ValueError("per-image summary image_count mismatch.")
    if payload.get("image_with_detections_count") + payload.get("image_without_detections_count") != payload.get("image_count"):
        raise ValueError("per-image summary detection count balance mismatch.")
    if payload.get("total_bbox_count") != _sum_predicted_boxes(payload.get("summary_rows", [])):
        raise ValueError("per-image summary total_bbox_count mismatch.")
    if _sha256_file(SOURCE_PER_IMAGE_SUMMARY_PATH) != expected_hash:
        raise ValueError("per-image summary hash mismatch.")


def _validate_confidence_payload(payload: dict[str, Any], expected_hash: str) -> None:
    missing = [field for field in REQUIRED_CONFIDENCE_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"confidence distribution artifact missing required fields: {missing}")
    if payload.get("artifact_type") != "detection_confidence_distribution":
        raise ValueError("confidence distribution artifact_type mismatch.")
    if payload.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("confidence distribution track_id mismatch.")
    if payload.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("confidence distribution task_type mismatch.")
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("confidence distribution run_id mismatch.")
    if payload.get("run_config_id") != EXPECTED_RUN_CONFIG_ID:
        raise ValueError("confidence distribution run_config_id mismatch.")
    if payload.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("confidence distribution model_name mismatch.")
    if payload.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("confidence distribution model_type mismatch.")
    if payload.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("confidence distribution model_version mismatch.")
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("confidence distribution dataset_id mismatch.")
    if payload.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("confidence distribution dataset_version mismatch.")
    if payload.get("split") != EXPECTED_SPLIT:
        raise ValueError("confidence distribution split mismatch.")
    if payload.get("image_count") != 345:
        raise ValueError("confidence distribution image_count mismatch.")
    if payload.get("total_bbox_count") != 573:
        raise ValueError("confidence distribution total_bbox_count mismatch.")
    if payload.get("global_confidence_summary", {}).get("confidence_count") != 573:
        raise ValueError("confidence distribution confidence_count mismatch.")
    if sum(int(item.get("count", 0)) for item in payload.get("confidence_bins", [])) != 573:
        raise ValueError("confidence distribution confidence_bin_count_sum mismatch.")
    if sum(int(item.get("bbox_count", 0)) for item in payload.get("class_confidence_summary", [])) != 573:
        raise ValueError("confidence distribution class_bbox_count_sum mismatch.")
    if (
        payload.get("global_confidence_summary", {}).get("low_confidence_count", 0)
        + payload.get("global_confidence_summary", {}).get("medium_confidence_count", 0)
        + payload.get("global_confidence_summary", {}).get("high_confidence_count", 0)
        != 573
    ):
        raise ValueError("confidence distribution band count sum mismatch.")
    if _sha256_file(SOURCE_CONFIDENCE_DISTRIBUTION_PATH) != expected_hash:
        raise ValueError("confidence distribution hash mismatch.")


def _validate_cross_artifacts(
    bbox_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    confidence_payload: dict[str, Any],
) -> None:
    if bbox_payload["image_count"] != summary_payload["image_count"]:
        raise ValueError("bbox and per-image summary image_count mismatch.")
    if bbox_payload["bbox_count"] != summary_payload["total_bbox_count"]:
        raise ValueError("bbox and per-image summary total_bbox_count mismatch.")
    if bbox_payload["image_count"] != confidence_payload["image_count"]:
        raise ValueError("bbox and confidence distribution image_count mismatch.")
    if bbox_payload["bbox_count"] != confidence_payload["total_bbox_count"]:
        raise ValueError("bbox and confidence distribution total_bbox_count mismatch.")
    if summary_payload["image_count"] != confidence_payload["image_count"]:
        raise ValueError("per-image summary and confidence distribution image_count mismatch.")
    if summary_payload["total_bbox_count"] != confidence_payload["total_bbox_count"]:
        raise ValueError("per-image summary and confidence distribution total_bbox_count mismatch.")


def _index_summary_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    indexed_rows: list[dict[str, Any]] = []
    index_map: dict[str, int] = {}
    for index, row in enumerate(rows):
        row_copy = dict(row)
        row_copy["source_summary_row_index"] = index
        indexed_rows.append(row_copy)
        index_map[row_copy["image_id"]] = index
    return indexed_rows, index_map


def _index_bbox_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        indexed[row["image_id"]] = dict(row)
    return indexed


def _validate_row_alignment(
    summary_rows: list[dict[str, Any]],
    bbox_rows: dict[str, dict[str, Any]],
) -> None:
    if set(row["image_id"] for row in summary_rows) != set(bbox_rows):
        raise ValueError("bbox and summary image_id sets do not match.")
    for row in summary_rows:
        bbox_row = bbox_rows[row["image_id"]]
        if row["image_path"] != bbox_row["image_path"]:
            raise ValueError(f"image_path mismatch for image_id={row['image_id']}")
        if row["image_width"] != bbox_row["image_width"]:
            raise ValueError(f"image_width mismatch for image_id={row['image_id']}")
        if row["image_height"] != bbox_row["image_height"]:
            raise ValueError(f"image_height mismatch for image_id={row['image_id']}")
        if row["predicted_box_count"] != bbox_row["predicted_box_count"]:
            raise ValueError(f"predicted_box_count mismatch for image_id={row['image_id']}")
        if row["defect_count"] != bbox_row["defect_count"]:
            raise ValueError(f"defect_count mismatch for image_id={row['image_id']}")
        if row["best_prediction"] != bbox_row["best_prediction"]:
            raise ValueError(f"best_prediction mismatch for image_id={row['image_id']}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def _sum_boxes(rows: list[dict[str, Any]]) -> int:
    return sum(len(row.get("boxes", [])) for row in rows)


def _sum_predicted_boxes(rows: list[dict[str, Any]]) -> int:
    return sum(int(row.get("predicted_box_count", 0)) for row in rows)


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


def _median(values: list[float | int]) -> float:
    if not values:
        raise ValueError("median requires at least one value.")
    return float(median(values))


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _repo_path_exists(path_value: str) -> bool:
    path = Path(path_value)
    if path.is_absolute():
        return path.is_file()
    return (REPO_ROOT / path).is_file()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
