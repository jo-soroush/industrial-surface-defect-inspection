"""Build a governed inventory for the YOLO bbox prediction export artifact."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PREDICTION_ARTIFACT_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
)
OUTPUT_PATH = REPO_ROOT / (
    "artifacts/models/inventory/"
    "track_detection_bbox_prediction_artifact_inventory__yolo_train_v0_2_0__validation.json"
)
EXPECTED_RUN_ID = "yolo_train_v0_2_0"
EXPECTED_TRACK_ID = "detection"
EXPECTED_TASK_TYPE = "object_detection"
EXPECTED_SPLIT = "validation"

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
    "created_at",
    "source_artifact_paths",
    "prediction_parameters",
    "image_count",
    "prediction_count",
    "bbox_count",
    "prediction_rows",
]

PRODUCTION_TERMS = ("production-ready", "deployment-safe")


def main() -> int:
    try:
        source_exists = SOURCE_PREDICTION_ARTIFACT_PATH.is_file()
        source_size_bytes = SOURCE_PREDICTION_ARTIFACT_PATH.stat().st_size if source_exists else 0
        source_sha256 = _sha256_file(SOURCE_PREDICTION_ARTIFACT_PATH) if source_exists else ""
        source_valid_json = False
        source_payload: dict[str, Any] | None = None
        validation_checks: list[dict[str, Any]] = []

        if not source_exists:
            validation_checks.append(
                _check("source_prediction_artifact_exists", "FAIL", "source prediction artifact is missing.")
            )
            raise FileNotFoundError(
                f"source prediction artifact not found: {_repo_relative(SOURCE_PREDICTION_ARTIFACT_PATH)}"
            )

        source_payload = _load_json(SOURCE_PREDICTION_ARTIFACT_PATH)
        source_valid_json = True
        required_fields_missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in source_payload]
        if required_fields_missing:
            raise ValueError(f"source prediction artifact missing fields: {required_fields_missing}")

        row_count = len(source_payload["prediction_rows"])
        image_count = source_payload["image_count"]
        prediction_count = source_payload["prediction_count"]
        bbox_count = source_payload["bbox_count"]

        validation_checks.extend(
            [
                _check("source_prediction_artifact_exists", "PASS", _repo_relative(SOURCE_PREDICTION_ARTIFACT_PATH)),
                _check("source_prediction_artifact_valid_json", "PASS", "source prediction artifact parsed successfully."),
                _check("required_top_level_fields_present", "PASS", "all required top-level fields are present."),
                _check("run_id_matches", _status(source_payload.get("run_id") == EXPECTED_RUN_ID), f"run_id={source_payload.get('run_id')}"),
                _check("track_id_matches", _status(source_payload.get("track_id") == EXPECTED_TRACK_ID), f"track_id={source_payload.get('track_id')}"),
                _check("task_type_matches", _status(source_payload.get("task_type") == EXPECTED_TASK_TYPE), f"task_type={source_payload.get('task_type')}"),
                _check("split_matches", _status(source_payload.get("split") == EXPECTED_SPLIT), f"split={source_payload.get('split')}"),
                _check("image_count_matches_row_count", _status(image_count == row_count), f"image_count={image_count}, row_count={row_count}"),
                _check("prediction_count_matches_row_count", _status(prediction_count == row_count), f"prediction_count={prediction_count}, row_count={row_count}"),
                _check("bbox_count_matches_box_total", _status(bbox_count == _sum_boxes(source_payload["prediction_rows"])), f"bbox_count={bbox_count}"),
                _check("file_size_positive", _status(source_size_bytes > 0), f"size_bytes={source_size_bytes}"),
                _check("sha256_computed", _status(bool(source_sha256)), f"sha256={source_sha256}"),
                _check("no_production_or_deployment_claims", _status(not _contains_forbidden_terms(source_payload)), "source artifact contains no forbidden production or deployment wording."),
                _check("registry_update_deferred", "PASS", "registry publication is deferred to a later task."),
                _check("frontend_bundle_deferred", "PASS", "frontend bundle generation is deferred to a later task."),
            ]
        )

        if any(check["status"] == "FAIL" for check in validation_checks):
            raise ValueError("source prediction artifact validation failed.")

        source_artifact_paths = _collect_source_artifact_paths(source_payload)
        inventory = {
            "inventory_type": "track_detection_bbox_prediction_artifact_inventory",
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
            "source_prediction_artifact_path": _repo_relative(SOURCE_PREDICTION_ARTIFACT_PATH),
            "source_prediction_artifact_sha256": source_sha256,
            "source_prediction_artifact_size_bytes": source_size_bytes,
            "source_prediction_artifact_exists": source_exists,
            "source_prediction_artifact_valid_json": source_valid_json,
            "image_count": image_count,
            "prediction_count": prediction_count,
            "bbox_count": bbox_count,
            "row_count": row_count,
            "source_artifact_paths": source_artifact_paths,
            "prediction_parameters": source_payload["prediction_parameters"],
            "created_at": _utc_now_iso(),
            "inventory_status": "pass",
            "validation_checks": validation_checks,
            "known_limitations": [
                "This inventory is evidence for a prediction export artifact.",
                "It is not a training artifact.",
                "It is not a model-quality improvement artifact.",
                "It is not a production-ready claim.",
                "Registry update is deferred to a later explicit step.",
                "Frontend bundle generation is deferred to a later explicit step.",
            ],
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(OUTPUT_PATH, inventory)

        print("# Detection Prediction Artifact Inventory Builder")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(SOURCE_PREDICTION_ARTIFACT_PATH)}")
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
        print("# Detection Prediction Artifact Inventory Builder")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(SOURCE_PREDICTION_ARTIFACT_PATH)}")
        print(f"- exists: {'PASS' if SOURCE_PREDICTION_ARTIFACT_PATH.is_file() else 'FAIL'}")
        print(f"- valid_json: {'PASS' if SOURCE_PREDICTION_ARTIFACT_PATH.is_file() else 'FAIL'}")
        print(f"- size_bytes: {SOURCE_PREDICTION_ARTIFACT_PATH.stat().st_size if SOURCE_PREDICTION_ARTIFACT_PATH.is_file() else 0}")
        print(f"- sha256: { _sha256_file(SOURCE_PREDICTION_ARTIFACT_PATH) if SOURCE_PREDICTION_ARTIFACT_PATH.is_file() else '' }")
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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def _collect_source_artifact_paths(source_payload: dict[str, Any]) -> list[str]:
    source_paths = [ _repo_relative(SOURCE_PREDICTION_ARTIFACT_PATH) ]
    for path in source_payload.get("source_artifact_paths", []):
        if isinstance(path, str) and path not in source_paths:
            source_paths.append(path)
    return source_paths


def _sum_boxes(rows: list[dict[str, Any]]) -> int:
    return sum(len(row.get("boxes", [])) for row in rows)


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {_repo_relative(path)}")
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
