"""Build a governed inventory for the YOLO detection sample gallery artifact."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SAMPLE_GALLERY_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_sample_gallery__yolo_train_v0_2_0__validation.json"
)
OUTPUT_PATH = REPO_ROOT / (
    "artifacts/models/inventory/"
    "track_detection_sample_gallery_artifact_inventory__yolo_train_v0_2_0__validation.json"
)

EXPECTED_TRACK_ID = "detection"
EXPECTED_TASK_TYPE = "object_detection"
EXPECTED_RUN_ID = "yolo_train_v0_2_0"
EXPECTED_SPLIT = "validation"
EXPECTED_IMAGE_COUNT = 345
EXPECTED_TOTAL_BBOX_COUNT = 573
EXPECTED_CATEGORY_IDS = [
    "no_detection_examples",
    "multi_detection_examples",
    "high_confidence_examples",
    "medium_confidence_examples",
    "low_confidence_examples",
    "representative_examples",
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
    "source_confidence_distribution_artifact_path",
    "source_confidence_distribution_artifact_hash",
    "source_artifact_paths",
    "created_at",
    "image_count",
    "total_bbox_count",
    "gallery_category_count",
    "gallery_sample_count",
    "categories",
]

REQUIRED_SAMPLE_FIELDS = [
    "image_id",
    "image_path",
    "image_width",
    "image_height",
    "predicted_box_count",
    "has_detections",
    "defect_count",
    "best_prediction",
    "max_confidence",
    "mean_confidence",
    "predicted_class_labels",
    "reason_selected",
    "source_summary_row_index",
]

PRODUCTION_TERMS = ("production-ready", "deployment-safe")


def main() -> int:
    validation_checks: list[dict[str, Any]] = []
    source_exists = SOURCE_SAMPLE_GALLERY_PATH.is_file()
    source_size_bytes = SOURCE_SAMPLE_GALLERY_PATH.stat().st_size if source_exists else 0
    source_sha256 = _sha256_file(SOURCE_SAMPLE_GALLERY_PATH) if source_exists else ""
    source_valid_json = False
    inventory_status = "fail"

    try:
        if not source_exists:
            validation_checks.append(
                _check("source_sample_gallery_artifact_exists", "FAIL", "source sample gallery artifact is missing.")
            )
            raise FileNotFoundError(
                f"source sample gallery artifact not found: {_repo_relative(SOURCE_SAMPLE_GALLERY_PATH)}"
            )

        source_payload = _load_json(SOURCE_SAMPLE_GALLERY_PATH, "sample gallery artifact")
        source_valid_json = True

        missing_fields = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in source_payload]
        categories = source_payload.get("categories")
        if not isinstance(categories, list):
            categories = []

        category_ids = [
            category.get("category_id")
            for category in categories
            if isinstance(category, dict) and isinstance(category.get("category_id"), str)
        ]
        category_sample_counts = _category_sample_counts(categories)
        gallery_sample_count_from_categories = sum(category_sample_counts.values())
        duplicate_image_id_count = _duplicate_image_id_count(categories)

        bbox_path = _repo_path(source_payload.get("source_bbox_prediction_artifact_path", ""))
        summary_path = _repo_path(source_payload.get("source_per_image_summary_artifact_path", ""))
        confidence_path = _repo_path(source_payload.get("source_confidence_distribution_artifact_path", ""))
        bbox_exists = bbox_path.is_file()
        summary_exists = summary_path.is_file()
        confidence_exists = confidence_path.is_file()
        bbox_sha256 = _sha256_file(bbox_path) if bbox_exists else ""
        summary_sha256 = _sha256_file(summary_path) if summary_exists else ""
        confidence_sha256 = _sha256_file(confidence_path) if confidence_exists else ""
        summary_payload = _load_json(summary_path, "per-image summary artifact") if summary_exists else {}
        summary_rows = summary_payload.get("summary_rows", [])

        validation_checks.extend(
            [
                _check("source_sample_gallery_artifact_exists", "PASS", _repo_relative(SOURCE_SAMPLE_GALLERY_PATH)),
                _check("source_sample_gallery_artifact_valid_json", "PASS", "source sample gallery artifact parsed successfully."),
                _check(
                    "required_top_level_fields_present",
                    _status(not missing_fields),
                    f"missing_fields={missing_fields}",
                ),
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
                    "total_bbox_count_matches_expected",
                    _status(source_payload.get("total_bbox_count") == EXPECTED_TOTAL_BBOX_COUNT),
                    f"total_bbox_count={source_payload.get('total_bbox_count')}, expected={EXPECTED_TOTAL_BBOX_COUNT}",
                ),
                _check(
                    "gallery_category_count_matches_categories",
                    _status(source_payload.get("gallery_category_count") == len(categories)),
                    f"gallery_category_count={source_payload.get('gallery_category_count')}, len_categories={len(categories)}",
                ),
                _check(
                    "gallery_sample_count_matches_category_total",
                    _status(source_payload.get("gallery_sample_count") == gallery_sample_count_from_categories),
                    (
                        f"gallery_sample_count={source_payload.get('gallery_sample_count')}, "
                        f"category_total={gallery_sample_count_from_categories}"
                    ),
                ),
                _check(
                    "category_sample_counts_match_samples",
                    _status(_category_sample_counts_match_samples(categories)),
                    "each category sample_count matches len(samples).",
                ),
                _check(
                    "category_sample_counts_within_limit",
                    _status(all(count <= 5 for count in category_sample_counts.values())),
                    f"category_sample_counts={category_sample_counts}",
                ),
                _check(
                    "expected_categories_present",
                    _status(set(category_ids) == set(EXPECTED_CATEGORY_IDS)),
                    f"category_ids={category_ids}",
                ),
                _check(
                    "duplicate_image_ids_absent",
                    _status(duplicate_image_id_count == 0),
                    f"duplicate_image_id_count={duplicate_image_id_count}",
                ),
                _check(
                    "sample_required_fields_present",
                    _status(_samples_have_required_fields(categories)),
                    "every sample includes required fields.",
                ),
                _check(
                    "source_summary_row_indexes_valid",
                    _status(_summary_row_indexes_are_valid(categories, summary_rows)),
                    "every sample source_summary_row_index is present, in range, and matches image_id.",
                ),
                _check(
                    "sample_image_paths_repo_relative_and_exist",
                    _status(_sample_image_paths_are_repo_relative_and_exist(categories)),
                    "every sample image_path is repo-relative and exists on disk.",
                ),
                _check("file_size_positive", _status(source_size_bytes > 0), f"size_bytes={source_size_bytes}"),
                _check("sha256_computed", _status(len(source_sha256) == 64), f"sha256={source_sha256}"),
                _check(
                    "source_bbox_prediction_artifact_hash_matches",
                    _status(bbox_exists and source_payload.get("source_bbox_prediction_artifact_hash") == bbox_sha256),
                    f"path={_repo_relative(bbox_path)}, exists={bbox_exists}",
                ),
                _check(
                    "source_per_image_summary_artifact_hash_matches",
                    _status(summary_exists and source_payload.get("source_per_image_summary_artifact_hash") == summary_sha256),
                    f"path={_repo_relative(summary_path)}, exists={summary_exists}",
                ),
                _check(
                    "source_confidence_distribution_artifact_hash_matches",
                    _status(
                        confidence_exists
                        and source_payload.get("source_confidence_distribution_artifact_hash") == confidence_sha256
                    ),
                    f"path={_repo_relative(confidence_path)}, exists={confidence_exists}",
                ),
                _check(
                    "no_production_ready_claim",
                    _status(not _contains_term(source_payload, "production-ready")),
                    "source artifact contains no production-ready wording.",
                ),
                _check(
                    "no_deployment_safe_claim",
                    _status(not _contains_term(source_payload, "deployment-safe")),
                    "source artifact contains no deployment-safe wording.",
                ),
                _check("registry_update_deferred", "PASS", "registry update is not performed by this builder."),
                _check("frontend_bundle_deferred", "PASS", "frontend bundle generation is not performed by this builder."),
                _check("notebook_update_deferred", "PASS", "notebook update is not performed by this builder."),
                _check("annotated_image_generation_not_performed", "PASS", "annotated images are not created by this builder."),
            ]
        )

        if any(check["status"] == "FAIL" for check in validation_checks):
            raise ValueError("source sample gallery artifact validation failed.")

        source_artifact_paths = _collect_source_artifact_paths(source_payload)
        inventory_status = "pass"
        inventory = {
            "inventory_type": "track_detection_sample_gallery_artifact_inventory",
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
            "source_sample_gallery_artifact_path": _repo_relative(SOURCE_SAMPLE_GALLERY_PATH),
            "source_sample_gallery_artifact_sha256": source_sha256,
            "source_sample_gallery_artifact_size_bytes": source_size_bytes,
            "source_sample_gallery_artifact_exists": source_exists,
            "source_sample_gallery_artifact_valid_json": source_valid_json,
            "image_count": source_payload["image_count"],
            "total_bbox_count": source_payload["total_bbox_count"],
            "gallery_category_count": source_payload["gallery_category_count"],
            "gallery_sample_count": source_payload["gallery_sample_count"],
            "category_ids": category_ids,
            "category_sample_counts": category_sample_counts,
            "duplicate_image_id_count": duplicate_image_id_count,
            "source_bbox_prediction_artifact_path": source_payload["source_bbox_prediction_artifact_path"],
            "source_bbox_prediction_artifact_hash": source_payload["source_bbox_prediction_artifact_hash"],
            "source_per_image_summary_artifact_path": source_payload["source_per_image_summary_artifact_path"],
            "source_per_image_summary_artifact_hash": source_payload["source_per_image_summary_artifact_hash"],
            "source_confidence_distribution_artifact_path": source_payload["source_confidence_distribution_artifact_path"],
            "source_confidence_distribution_artifact_hash": source_payload["source_confidence_distribution_artifact_hash"],
            "source_artifact_paths": source_artifact_paths,
            "created_at": _utc_now_iso(),
            "inventory_status": inventory_status,
            "validation_checks": validation_checks,
            "known_limitations": [
                "This is derivative sample gallery inventory evidence.",
                "It is not training.",
                "It is not model-quality improvement.",
                "It is not a production-ready claim.",
                "Registry update is deferred.",
                "Frontend bundle generation is deferred.",
                "Notebook update is deferred.",
                "Annotated image generation is not performed.",
            ],
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(OUTPUT_PATH, inventory)
        _print_report(source_exists, source_valid_json, source_size_bytes, source_sha256, validation_checks, inventory_status)
        return 0
    except Exception as exc:
        if not validation_checks:
            validation_checks.append(_check("builder_failed", "FAIL", str(exc)))
        _print_report(source_exists, source_valid_json, source_size_bytes, source_sha256, validation_checks, inventory_status)
        print(f"failure_reason={exc}")
        return 1


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def _category_sample_counts(categories: Any) -> dict[str, int]:
    if not isinstance(categories, list):
        return {}
    counts: dict[str, int] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        category_id = category.get("category_id")
        if not isinstance(category_id, str):
            continue
        counts[category_id] = int(category.get("sample_count", 0))
    return counts


def _check(name: str, status: str, details: str) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def _category_sample_counts_match_samples(categories: Any) -> bool:
    if not isinstance(categories, list):
        return False
    for category in categories:
        if not isinstance(category, dict):
            return False
        samples = category.get("samples")
        if not isinstance(samples, list):
            return False
        if category.get("sample_count") != len(samples):
            return False
    return True


def _collect_source_artifact_paths(source_payload: dict[str, Any]) -> list[str]:
    source_paths = [_repo_relative(SOURCE_SAMPLE_GALLERY_PATH)]
    for field in [
        "source_bbox_prediction_artifact_path",
        "source_per_image_summary_artifact_path",
        "source_confidence_distribution_artifact_path",
    ]:
        path = source_payload.get(field)
        if isinstance(path, str) and path not in source_paths:
            source_paths.append(path)
    for path in source_payload.get("source_artifact_paths", []):
        if isinstance(path, str) and path not in source_paths:
            source_paths.append(path)
    return source_paths


def _contains_term(payload: Any, term: str) -> bool:
    if isinstance(payload, dict):
        return any(_contains_term(value, term) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_term(item, term) for item in payload)
    if isinstance(payload, str):
        return term in payload.lower()
    return False


def _duplicate_image_id_count(categories: Any) -> int:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for sample in _iter_samples(categories):
        image_id = sample.get("image_id")
        if not isinstance(image_id, str):
            continue
        if image_id in seen:
            duplicates.add(image_id)
        seen.add(image_id)
    return len(duplicates)


def _iter_samples(categories: Any) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not isinstance(categories, list):
        return samples
    for category in categories:
        if not isinstance(category, dict):
            continue
        category_samples = category.get("samples")
        if not isinstance(category_samples, list):
            continue
        samples.extend(sample for sample in category_samples if isinstance(sample, dict))
    return samples


def _load_json(path: Path, artifact_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON payload must be an object: {_repo_relative(path)}")
    return payload


def _print_report(
    source_exists: bool,
    source_valid_json: bool,
    source_size_bytes: int,
    source_sha256: str,
    validation_checks: list[dict[str, Any]],
    inventory_status: str,
) -> None:
    print("# Detection Sample Gallery Artifact Inventory Builder")
    print()
    print("## Source Artifact")
    print(f"- path: {_repo_relative(SOURCE_SAMPLE_GALLERY_PATH)}")
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
    print(f"- inventory_status: {inventory_status}")
    print("- registry update: NOT PERFORMED")
    print("- frontend bundle: NOT PERFORMED")
    print("- notebook update: NOT PERFORMED")
    print("- annotated images created: NO")
    print()
    print("## Final Verdict")
    print("PASS" if inventory_status == "pass" else "FAIL")


def _repo_path(path: Any) -> Path:
    if not isinstance(path, str):
        return REPO_ROOT / "__missing_path__"
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _sample_image_paths_are_repo_relative_and_exist(categories: Any) -> bool:
    for sample in _iter_samples(categories):
        image_path = sample.get("image_path")
        if not isinstance(image_path, str):
            return False
        candidate = Path(image_path)
        if candidate.is_absolute():
            return False
        if not (REPO_ROOT / candidate).is_file():
            return False
    return True


def _samples_have_required_fields(categories: Any) -> bool:
    samples = _iter_samples(categories)
    if not samples:
        return False
    for sample in samples:
        if any(field not in sample for field in REQUIRED_SAMPLE_FIELDS):
            return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _summary_row_indexes_are_valid(categories: Any, summary_rows: Any) -> bool:
    if not isinstance(summary_rows, list):
        return False
    for sample in _iter_samples(categories):
        source_index = sample.get("source_summary_row_index")
        if not isinstance(source_index, int):
            return False
        if source_index < 0 or source_index >= len(summary_rows):
            return False
        row = summary_rows[source_index]
        if not isinstance(row, dict):
            return False
        if row.get("image_id") != sample.get("image_id"):
            return False
    return True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
