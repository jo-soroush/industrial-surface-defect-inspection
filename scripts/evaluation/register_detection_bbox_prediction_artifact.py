"""Register the governed YOLO bbox prediction export artifact."""

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
    "track_detection_bbox_prediction_artifact_inventory__yolo_train_v0_2_0__validation.json"
)
DEFAULT_ARTIFACT_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
)
EXPECTED_ARTIFACT_ID = "track_detection__yolo_train_v0_2_0__bbox_predictions_validation"
EXPECTED_ARTIFACT_TYPE = "detection_yolo_bbox_predictions"
EXPECTED_RUN_ID = "yolo_train_v0_2_0"
EXPECTED_CONFIG_ID = "yolo_train_v0_2_0"
EXPECTED_MODEL_NAME = "yolo"
EXPECTED_MODEL_VERSION = "0.2.0"
EXPECTED_DATASET_ID = "gc10det_detection"
EXPECTED_DATASET_VERSION = "gc10det_1.0"
EXPECTED_STATUS = "active"
EXPECTED_STORAGE_BACKEND = "local"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register the governed YOLO bbox prediction export artifact."
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to artifact_registry.yaml.",
    )
    parser.add_argument(
        "--inventory-path",
        default=str(DEFAULT_INVENTORY_PATH),
        help="Path to the prediction artifact inventory JSON.",
    )
    parser.add_argument(
        "--artifact-path",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Path to the bbox prediction export JSON.",
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
        source_artifact = _load_json(artifact_path, "source bbox prediction artifact")
        inventory = _load_json(inventory_path, "prediction artifact inventory")
        registry = _load_yaml(registry_path, "artifact registry")

        _validate_inventory(inventory)
        _validate_source_artifact(source_artifact, inventory, artifact_path)
        _validate_registry_shape(registry)

        entry = _build_registry_entry(inventory, artifact_path)
        validation_checks = _validate_registry_entry(entry, registry)

        registry_status = "not performed"
        if publish:
            updated_registry = _register_entry(registry, entry)
            _write_yaml_atomic(registry_path, updated_registry)
            reloaded = _load_yaml(registry_path, "artifact registry")
            _validate_registry_shape(reloaded)
            _validate_post_write(reloaded, entry, inventory)
            registry_status = "pass"

        print("# Detection BBox Prediction Artifact Registry Update")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(artifact_path)}")
        print(f"- exists: PASS")
        print(f"- valid_json: PASS")
        print(f"- size_bytes: {artifact_path.stat().st_size}")
        print(f"- sha256: {inventory['source_prediction_artifact_sha256']}")
        print()
        print("## Inventory")
        print(f"- path: {_repo_relative(inventory_path)}")
        print(f"- inventory_status: {inventory.get('inventory_status')}")
        print(f"- image_count: {inventory.get('image_count')}")
        print(f"- prediction_count: {inventory.get('prediction_count')}")
        print(f"- bbox_count: {inventory.get('bbox_count')}")
        print(f"- row_count: {inventory.get('row_count')}")
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
        print(f"- registry update: {'PERFORMED' if publish else 'NOT PERFORMED'}")
        print("- run_registry update: NOT PERFORMED")
        print("- frontend bundle: NOT PERFORMED")
        print("- notebook update: NOT PERFORMED")
        print()
        print("## Final Verdict")
        print("PASS")
        return 0
    except Exception as exc:
        print("# Detection BBox Prediction Artifact Registry Update")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(artifact_path)}")
        print(f"- exists: {'PASS' if artifact_path.is_file() else 'FAIL'}")
        print(f"- valid_json: {'PASS' if artifact_path.is_file() else 'FAIL'}")
        print(f"- size_bytes: {artifact_path.stat().st_size if artifact_path.is_file() else 0}")
        print(
            f"- sha256: {inventory.get('source_prediction_artifact_sha256') if 'inventory' in locals() and isinstance(inventory, dict) else ''}"
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
        "artifact_hash": inventory["source_prediction_artifact_sha256"],
        "artifact_size_bytes": inventory["source_prediction_artifact_size_bytes"],
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
        "run_id",
        "run_config_id",
        "model_name",
        "model_type",
        "model_version",
        "dataset_id",
        "dataset_version",
        "split",
        "source_prediction_artifact_path",
        "source_prediction_artifact_sha256",
        "source_prediction_artifact_size_bytes",
        "source_prediction_artifact_exists",
        "source_prediction_artifact_valid_json",
        "image_count",
        "prediction_count",
        "bbox_count",
        "row_count",
        "source_artifact_paths",
        "prediction_parameters",
        "created_at",
        "validation_checks",
        "known_limitations",
    }
    missing = [field for field in required if field not in inventory]
    if missing:
        raise ValueError(f"inventory missing required fields: {missing}")
    if inventory.get("inventory_status") != "pass":
        raise ValueError("inventory_status must be pass.")
    if inventory.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("inventory run_id mismatch.")
    if inventory.get("run_config_id") != EXPECTED_CONFIG_ID:
        raise ValueError("inventory run_config_id mismatch.")
    if inventory.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("inventory model_name mismatch.")
    if inventory.get("model_type") != EXPECTED_MODEL_NAME:
        raise ValueError("inventory model_type mismatch.")
    if inventory.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError("inventory model_version mismatch.")
    if inventory.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("inventory dataset_id mismatch.")
    if inventory.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("inventory dataset_version mismatch.")
    if inventory.get("split") != "validation":
        raise ValueError("inventory split must be validation.")


def _validate_source_artifact(
    source_artifact: dict[str, Any],
    inventory: dict[str, Any],
    artifact_path: Path,
) -> None:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"source artifact not found: {_repo_relative(artifact_path)}")
    if inventory.get("source_prediction_artifact_exists") is not True:
        raise ValueError("inventory must confirm the source artifact exists.")
    if inventory.get("source_prediction_artifact_valid_json") is not True:
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
        "created_at",
        "source_artifact_paths",
        "prediction_parameters",
        "image_count",
        "prediction_count",
        "bbox_count",
        "prediction_rows",
    }
    missing = [field for field in required_fields if field not in source_artifact]
    if missing:
        raise ValueError(f"source artifact missing required fields: {missing}")
    if source_artifact.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("source artifact run_id mismatch.")
    if source_artifact.get("track_id") != "detection":
        raise ValueError("source artifact track_id must be detection.")
    if source_artifact.get("task_type") != "object_detection":
        raise ValueError("source artifact task_type must be object_detection.")
    if source_artifact.get("split") != "validation":
        raise ValueError("source artifact split must be validation.")
    if len(source_artifact.get("prediction_rows", [])) != source_artifact.get("image_count"):
        raise ValueError("source artifact image_count mismatch.")
    if len(source_artifact.get("prediction_rows", [])) != source_artifact.get("prediction_count"):
        raise ValueError("source artifact prediction_count mismatch.")
    if _sum_boxes(source_artifact.get("prediction_rows", [])) != source_artifact.get("bbox_count"):
        raise ValueError("source artifact bbox_count mismatch.")
    if _sha256_file(artifact_path) != inventory.get("source_prediction_artifact_sha256"):
        raise ValueError("source artifact hash mismatch with inventory.")
    if artifact_path.stat().st_size != inventory.get("source_prediction_artifact_size_bytes"):
        raise ValueError("source artifact size mismatch with inventory.")


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


def _check(name: str, status: str, details: str) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def _validate_post_write(
    registry: dict[str, Any],
    entry: dict[str, Any],
    inventory: dict[str, Any],
) -> None:
    matches = [artifact for artifact in registry.get("artifacts", []) if artifact.get("artifact_id") == entry["artifact_id"]]
    if len(matches) != 1:
        raise ValueError("artifact registry must contain exactly one matching entry after write.")
    actual = matches[0]
    if actual.get("artifact_hash") != inventory.get("source_prediction_artifact_sha256"):
        raise ValueError("artifact registry hash mismatch after write.")
    if actual.get("artifact_size_bytes") != inventory.get("source_prediction_artifact_size_bytes"):
        raise ValueError("artifact registry size mismatch after write.")


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
