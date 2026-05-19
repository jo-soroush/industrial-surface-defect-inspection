"""Register the governed YOLO confidence distribution artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "artifacts/models/registry/artifact_registry.yaml"
DEFAULT_INVENTORY_PATH = REPO_ROOT / (
    "artifacts/models/inventory/"
    "track_detection_confidence_distribution_artifact_inventory__yolo_train_v0_2_0__validation.json"
)
DEFAULT_ARTIFACT_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_confidence_distribution__yolo_train_v0_2_0__validation.json"
)
EXPECTED_ARTIFACT_ID = "track_detection__yolo_train_v0_2_0__confidence_distribution_validation"
EXPECTED_ARTIFACT_TYPE = "detection_confidence_distribution"
EXPECTED_RUN_ID = "yolo_train_v0_2_0"
EXPECTED_CONFIG_ID = "yolo_train_v0_2_0"
EXPECTED_TRACK_ID = "detection"
EXPECTED_TASK_TYPE = "object_detection"
EXPECTED_MODEL_NAME = "yolo"
EXPECTED_MODEL_TYPE = "yolo"
EXPECTED_MODEL_VERSION = "0.2.0"
EXPECTED_DATASET_ID = "gc10det_detection"
EXPECTED_DATASET_VERSION = "gc10det_1.0"
EXPECTED_SPLIT = "validation"
EXPECTED_CONFIDENCE_COUNT = 573
EXPECTED_BIN_COUNT_SUM = 573
EXPECTED_CLASS_BBOX_COUNT_SUM = 573
EXPECTED_BAND_COUNT_SUM = 573
EXPECTED_STATUS = "active"
EXPECTED_STORAGE_BACKEND = "local"
PRODUCTION_TERMS = ("production-ready", "deployment-safe")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register the governed YOLO confidence distribution artifact."
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to artifact_registry.yaml.",
    )
    parser.add_argument(
        "--inventory-path",
        default=str(DEFAULT_INVENTORY_PATH),
        help="Path to the confidence distribution artifact inventory JSON.",
    )
    parser.add_argument(
        "--artifact-path",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Path to the confidence distribution JSON.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        default=False,
        help="Persist the registry update. Without this flag, the script runs in dry-run mode.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry_path = Path(args.registry_path)
    inventory_path = Path(args.inventory_path)
    artifact_path = Path(args.artifact_path)
    publish = bool(args.publish)

    try:
        source_artifact = _load_json(artifact_path, "source confidence distribution artifact")
        inventory = _load_json(inventory_path, "confidence distribution artifact inventory")
        registry = _load_yaml(registry_path, "artifact registry")

        _validate_inventory(inventory)
        _validate_source_artifact(source_artifact, inventory, artifact_path)
        _validate_registry_shape(registry)

        entry = _build_registry_entry(inventory, artifact_path)
        validation_checks = _validate_registry_entry(entry, registry)

        registry_status = "not performed"
        if publish:
            updated_registry = _register_entry(registry, entry)
            if updated_registry is not registry:
                _write_yaml_atomic(registry_path, updated_registry)
                reloaded = _load_yaml(registry_path, "artifact registry")
                _validate_registry_shape(reloaded)
                _validate_post_write(reloaded, entry, inventory)
                registry_status = "performed"
            else:
                registry_status = "not performed"

        print("# Detection Confidence Distribution Artifact Registry Update")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(artifact_path)}")
        print("- exists: PASS")
        print("- valid_json: PASS")
        print(f"- size_bytes: {artifact_path.stat().st_size}")
        print(f"- sha256: {inventory['source_confidence_distribution_artifact_sha256']}")
        print()
        print("## Inventory")
        print(f"- path: {_repo_relative(inventory_path)}")
        print(f"- inventory_status: {inventory.get('inventory_status')}")
        print(f"- image_count: {inventory.get('image_count')}")
        print(f"- total_bbox_count: {inventory.get('total_bbox_count')}")
        print(f"- confidence_count: {inventory.get('confidence_count')}")
        print(f"- low_confidence_count: {inventory.get('low_confidence_count')}")
        print(f"- medium_confidence_count: {inventory.get('medium_confidence_count')}")
        print(f"- high_confidence_count: {inventory.get('high_confidence_count')}")
        print()
        print("## Registry Checks")
        for check in validation_checks:
            print(f"- {check['name']}: {check['status']} ({check['details']})")
        print()
        print("## Registry Entry")
        print(f"- artifact_id: {entry['artifact_id']}")
        print(f"- artifact_path: {entry['artifact_path']}")
        print(f"- artifact_hash: {entry['artifact_hash']}")
        print(f"- artifact_size_bytes: {entry['artifact_size_bytes']}")
        print(f"- status: {entry['status']}")
        print(f"- storage_backend: {entry['storage_backend']}")
        print()
        print("## Registry Output")
        print(f"- registry_path: {_repo_relative(registry_path)}")
        print(f"- registry update: {'PERFORMED' if registry_status == 'performed' else 'NOT PERFORMED'}")
        print("- run_registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("PASS")
        return 0
    except Exception as exc:
        print("# Detection Confidence Distribution Artifact Registry Update")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(artifact_path)}")
        print(f"- exists: {'PASS' if artifact_path.is_file() else 'FAIL'}")
        print(f"- valid_json: {'PASS' if artifact_path.is_file() else 'FAIL'}")
        print(f"- size_bytes: {artifact_path.stat().st_size if artifact_path.is_file() else 0}")
        print(
            f"- sha256: {inventory.get('source_confidence_distribution_artifact_sha256') if 'inventory' in locals() and isinstance(inventory, dict) else ''}"
        )
        print()
        print("## Inventory")
        print(f"- path: {_repo_relative(inventory_path)}")
        print(f"- inventory_status: {inventory.get('inventory_status') if 'inventory' in locals() and isinstance(inventory, dict) else 'fail'}")
        print()
        print("## Registry Checks")
        print(f"- FAIL: {exc}")
        print()
        print("## Registry Entry")
        print(f"- artifact_id: {EXPECTED_ARTIFACT_ID}")
        print(f"- artifact_path: {_repo_relative(artifact_path)}")
        print()
        print("## Registry Output")
        print(f"- registry_path: {_repo_relative(registry_path)}")
        print("- registry update: NOT PERFORMED")
        print("- run_registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("FAIL")
        print(f"failure_reason={exc}")
        return 1


def _build_registry_entry(inventory: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    return {
        "artifact_id": EXPECTED_ARTIFACT_ID,
        "artifact_type": EXPECTED_ARTIFACT_TYPE,
        "artifact_filename": artifact_path.name,
        "artifact_path": _repo_relative(artifact_path),
        "artifact_uri": None,
        "artifact_format": "json",
        "artifact_hash": inventory["source_confidence_distribution_artifact_sha256"],
        "artifact_size_bytes": inventory["source_confidence_distribution_artifact_size_bytes"],
        "run_id": EXPECTED_RUN_ID,
        "config_id": EXPECTED_CONFIG_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "model_version": EXPECTED_MODEL_VERSION,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "status": EXPECTED_STATUS,
        "storage_backend": EXPECTED_STORAGE_BACKEND,
        "created_at": _utc_now_iso(),
    }


def _validate_inventory(inventory: dict[str, Any]) -> None:
    required = {
        "inventory_status",
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
        "validation_checks",
        "known_limitations",
    }
    missing = [field for field in required if field not in inventory]
    if missing:
        raise ValueError(f"inventory missing required fields: {missing}")
    if inventory.get("inventory_status") != "pass":
        raise ValueError("inventory_status must be pass.")
    if inventory.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("inventory track_id mismatch.")
    if inventory.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("inventory task_type mismatch.")
    if inventory.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("inventory run_id mismatch.")
    if inventory.get("run_config_id") != EXPECTED_CONFIG_ID:
        raise ValueError("inventory run_config_id mismatch.")
    if inventory.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("inventory model_name mismatch.")
    if inventory.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("inventory model_type mismatch.")
    if inventory.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("inventory model_version mismatch.")
    if inventory.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("inventory dataset_id mismatch.")
    if inventory.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("inventory dataset_version mismatch.")
    if inventory.get("split") != EXPECTED_SPLIT:
        raise ValueError("inventory split must be validation.")
    if inventory.get("image_count") != 345:
        raise ValueError("inventory image_count mismatch.")
    if inventory.get("total_bbox_count") != 573:
        raise ValueError("inventory total_bbox_count mismatch.")
    if inventory.get("confidence_count") != EXPECTED_CONFIDENCE_COUNT:
        raise ValueError("inventory confidence_count mismatch.")
    if inventory.get("confidence_bin_count_sum") != EXPECTED_BIN_COUNT_SUM:
        raise ValueError("inventory confidence_bin_count_sum mismatch.")
    if inventory.get("class_bbox_count_sum") != EXPECTED_CLASS_BBOX_COUNT_SUM:
        raise ValueError("inventory class_bbox_count_sum mismatch.")
    if inventory.get("low_confidence_count") + inventory.get("medium_confidence_count") + inventory.get("high_confidence_count") != EXPECTED_BAND_COUNT_SUM:
        raise ValueError("inventory confidence band counts mismatch.")


def _validate_source_artifact(
    source_artifact: dict[str, Any],
    inventory: dict[str, Any],
    artifact_path: Path,
) -> None:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"source artifact not found: {_repo_relative(artifact_path)}")
    if inventory.get("source_confidence_distribution_artifact_exists") is not True:
        raise ValueError("inventory must confirm the source artifact exists.")
    if inventory.get("source_confidence_distribution_artifact_valid_json") is not True:
        raise ValueError("inventory must confirm the source artifact JSON is valid.")

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
    missing = [field for field in required_fields if field not in source_artifact]
    if missing:
        raise ValueError(f"source artifact missing required fields: {missing}")
    if source_artifact.get("artifact_type") != EXPECTED_ARTIFACT_TYPE:
        raise ValueError("source artifact artifact_type mismatch.")
    if source_artifact.get("track_id") != EXPECTED_TRACK_ID:
        raise ValueError("source artifact track_id must be detection.")
    if source_artifact.get("task_type") != EXPECTED_TASK_TYPE:
        raise ValueError("source artifact task_type must be object_detection.")
    if source_artifact.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("source artifact run_id mismatch.")
    if source_artifact.get("run_config_id") != EXPECTED_CONFIG_ID:
        raise ValueError("source artifact run_config_id mismatch.")
    if source_artifact.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("source artifact model_name mismatch.")
    if source_artifact.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("source artifact model_type mismatch.")
    if source_artifact.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("source artifact model_version mismatch.")
    if source_artifact.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("source artifact dataset_id mismatch.")
    if source_artifact.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("source artifact dataset_version mismatch.")
    if source_artifact.get("split") != EXPECTED_SPLIT:
        raise ValueError("source artifact split must be validation.")
    if source_artifact.get("image_count") != 345:
        raise ValueError("source artifact image_count mismatch.")
    if source_artifact.get("total_bbox_count") != 573:
        raise ValueError("source artifact total_bbox_count mismatch.")
    if source_artifact.get("global_confidence_summary", {}).get("confidence_count") != EXPECTED_CONFIDENCE_COUNT:
        raise ValueError("source artifact confidence_count mismatch.")
    if sum(int(item.get("count", 0)) for item in source_artifact.get("confidence_bins", [])) != EXPECTED_BIN_COUNT_SUM:
        raise ValueError("source artifact confidence_bin_count_sum mismatch.")
    if sum(int(item.get("bbox_count", 0)) for item in source_artifact.get("class_confidence_summary", [])) != EXPECTED_CLASS_BBOX_COUNT_SUM:
        raise ValueError("source artifact class_bbox_count_sum mismatch.")
    if (
        source_artifact.get("global_confidence_summary", {}).get("low_confidence_count", -1)
        + source_artifact.get("global_confidence_summary", {}).get("medium_confidence_count", -1)
        + source_artifact.get("global_confidence_summary", {}).get("high_confidence_count", -1)
        != EXPECTED_BAND_COUNT_SUM
    ):
        raise ValueError("source artifact confidence band counts mismatch.")
    if _sha256_file(artifact_path) != inventory.get("source_confidence_distribution_artifact_sha256"):
        raise ValueError("source artifact hash mismatch with inventory.")
    if artifact_path.stat().st_size != inventory.get("source_confidence_distribution_artifact_size_bytes"):
        raise ValueError("source artifact size mismatch with inventory.")
    if source_artifact.get("source_bbox_prediction_artifact_hash") != inventory.get("source_bbox_prediction_artifact_hash"):
        raise ValueError("source bbox hash mismatch with inventory.")
    if source_artifact.get("source_per_image_summary_artifact_hash") != inventory.get("source_per_image_summary_artifact_hash"):
        raise ValueError("source per-image hash mismatch with inventory.")

    if _sha256_file(Path(inventory["source_bbox_prediction_artifact_path"]).resolve()) != inventory.get("source_bbox_prediction_artifact_hash"):
        raise ValueError("inventory bbox source hash mismatch with actual bbox artifact.")
    if _sha256_file(Path(inventory["source_per_image_summary_artifact_path"]).resolve()) != inventory.get("source_per_image_summary_artifact_hash"):
        raise ValueError("inventory per-image source hash mismatch with actual per-image artifact.")

    for bin_payload, expected_label in zip(source_artifact.get("confidence_bins", []), ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"], strict=True):
        if bin_payload.get("label") != expected_label:
            raise ValueError("source artifact confidence bin label mismatch.")
        if bin_payload.get("count") is None or bin_payload.get("percentage") is None:
            raise ValueError("source artifact confidence bins must have numeric count and percentage.")

    global_summary = source_artifact.get("global_confidence_summary", {})
    for field in ("min_confidence", "max_confidence", "mean_confidence", "median_confidence"):
        if not _is_number(global_summary.get(field)):
            raise ValueError(f"source artifact {field} must be numeric.")
    for field in ("confidence_count", "low_confidence_count", "medium_confidence_count", "high_confidence_count"):
        if not isinstance(global_summary.get(field), int) or isinstance(global_summary.get(field), bool):
            raise ValueError(f"source artifact {field} must be an integer.")

    for row in source_artifact.get("class_confidence_summary", []):
        if not isinstance(row.get("class_id"), int) or isinstance(row.get("class_id"), bool):
            raise ValueError("source artifact class_id must be an integer.")
        if not isinstance(row.get("class_label"), str) or not row.get("class_label").strip():
            raise ValueError("source artifact class_label must be a non-empty string.")
        if not isinstance(row.get("bbox_count"), int) or isinstance(row.get("bbox_count"), bool):
            raise ValueError("source artifact bbox_count must be an integer.")
        for field in ("min_confidence", "max_confidence", "mean_confidence", "median_confidence"):
            if not _is_number(row.get(field)):
                raise ValueError(f"source artifact {field} must be numeric.")
        bin_counts = row.get("bin_counts", [])
        if not isinstance(bin_counts, list) or len(bin_counts) != 4:
            raise ValueError("source artifact class bin_counts must contain four bins.")
        for bin_payload in bin_counts:
            if not isinstance(bin_payload.get("count"), int) or isinstance(bin_payload.get("count"), bool):
                raise ValueError("source artifact class bin count must be an integer.")
            if not _is_number(bin_payload.get("percentage")):
                raise ValueError("source artifact class bin percentage must be numeric.")


def _validate_registry_shape(registry: dict[str, Any]) -> None:
    if not isinstance(registry, dict):
        raise ValueError("artifact registry must be a dictionary.")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifact registry artifacts must be a list.")


def _validate_registry_entry(
    entry: dict[str, Any],
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = registry.get("artifacts", [])
    existing_by_id = next(
        (artifact for artifact in artifacts if artifact.get("artifact_id") == entry["artifact_id"]),
        None,
    )
    if existing_by_id is not None and existing_by_id != entry:
        raise ValueError(
            "artifact_registry already contains artifact_id with different metadata."
        )

    for artifact in artifacts:
        if artifact.get("artifact_path") == entry["artifact_path"] and artifact.get("artifact_id") != entry["artifact_id"]:
            raise ValueError(
                "artifact_registry already contains artifact_path under a different artifact_id."
            )

    checks = [
        _check(
            "artifact_id_duplicate_handling",
            "PASS" if existing_by_id is None or existing_by_id == entry else "FAIL",
            "idempotent no-op is allowed for identical metadata.",
        ),
        _check("artifact_path_collision", "PASS", "no conflicting artifact_path entries detected."),
    ]
    return checks


def _register_entry(
    registry: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    artifacts = list(registry.get("artifacts", []))
    for existing_entry in artifacts:
        if existing_entry.get("artifact_id") == entry["artifact_id"]:
            if existing_entry == entry:
                return registry
            raise ValueError("artifact_registry already contains artifact_id with different metadata.")

    artifacts.append(entry)
    return {"artifacts": artifacts}


def _validate_post_write(
    registry: dict[str, Any],
    entry: dict[str, Any],
    inventory: dict[str, Any],
) -> None:
    matches = [artifact for artifact in registry.get("artifacts", []) if artifact.get("artifact_id") == entry["artifact_id"]]
    if len(matches) != 1:
        raise ValueError("artifact registry must contain exactly one matching entry after write.")
    actual = matches[0]
    if actual.get("artifact_hash") != inventory.get("source_confidence_distribution_artifact_sha256"):
        raise ValueError("artifact registry hash mismatch after write.")
    if actual.get("artifact_size_bytes") != inventory.get("source_confidence_distribution_artifact_size_bytes"):
        raise ValueError("artifact registry size mismatch after write.")


def _check(name: str, status: str, details: str) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False, indent=2)
    tmp_path.replace(path)


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a YAML object: {_repo_relative(path)}")
    return payload


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
