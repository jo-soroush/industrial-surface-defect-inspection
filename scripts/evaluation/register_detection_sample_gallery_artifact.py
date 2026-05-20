"""Register the governed YOLO detection sample gallery artifact."""

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
DEFAULT_RUN_REGISTRY_PATH = REPO_ROOT / "artifacts/models/registry/run_registry.yaml"
DEFAULT_INVENTORY_PATH = REPO_ROOT / (
    "artifacts/models/inventory/"
    "track_detection_sample_gallery_artifact_inventory__yolo_train_v0_2_0__validation.json"
)
DEFAULT_ARTIFACT_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_sample_gallery__yolo_train_v0_2_0__validation.json"
)

EXPECTED_ARTIFACT_ID = "track_detection__yolo_train_v0_2_0__sample_gallery_validation"
EXPECTED_ARTIFACT_TYPE = "detection_sample_gallery"
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
EXPECTED_IMAGE_COUNT = 345
EXPECTED_TOTAL_BBOX_COUNT = 573
EXPECTED_GALLERY_CATEGORY_COUNT = 6
EXPECTED_GALLERY_SAMPLE_COUNT = 30
EXPECTED_DUPLICATE_IMAGE_ID_COUNT = 0
EXPECTED_STATUS = "active"
EXPECTED_STORAGE_BACKEND = "local"
EXPECTED_CATEGORY_IDS = {
    "no_detection_examples",
    "multi_detection_examples",
    "high_confidence_examples",
    "medium_confidence_examples",
    "low_confidence_examples",
    "representative_examples",
}
ENTRY_COMPARISON_FIELDS = [
    "artifact_id",
    "artifact_type",
    "artifact_filename",
    "artifact_path",
    "artifact_uri",
    "artifact_format",
    "artifact_hash",
    "artifact_size_bytes",
    "run_id",
    "config_id",
    "model_name",
    "model_version",
    "dataset_id",
    "dataset_version",
    "status",
    "storage_backend",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register the governed YOLO detection sample gallery artifact."
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to artifact_registry.yaml.",
    )
    parser.add_argument(
        "--inventory-path",
        default=str(DEFAULT_INVENTORY_PATH),
        help="Path to the detection sample gallery artifact inventory JSON.",
    )
    parser.add_argument(
        "--artifact-path",
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Path to the detection sample gallery JSON.",
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
    run_registry_hash_before = _sha256_file(DEFAULT_RUN_REGISTRY_PATH) if DEFAULT_RUN_REGISTRY_PATH.is_file() else ""
    frontend_hash_before = _tree_hash(REPO_ROOT / "frontend")
    notebooks_hash_before = _tree_hash(REPO_ROOT / "notebooks")

    try:
        source_artifact = _load_json(artifact_path, "source sample gallery artifact")
        inventory = _load_json(inventory_path, "sample gallery artifact inventory")
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
                registry_status = "performed"

            reloaded = _load_yaml(registry_path, "artifact registry")
            _validate_registry_shape(reloaded)
            _validate_post_write(reloaded, entry, inventory)
            _validate_no_side_effects(run_registry_hash_before, frontend_hash_before, notebooks_hash_before)

        print("# Detection Sample Gallery Artifact Registry Update")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(artifact_path)}")
        print("- exists: PASS")
        print("- valid_json: PASS")
        print(f"- size_bytes: {artifact_path.stat().st_size}")
        print(f"- sha256: {inventory['source_sample_gallery_artifact_sha256']}")
        print()
        print("## Inventory")
        print(f"- path: {_repo_relative(inventory_path)}")
        print(f"- inventory_status: {inventory.get('inventory_status')}")
        print(f"- image_count: {inventory.get('image_count')}")
        print(f"- total_bbox_count: {inventory.get('total_bbox_count')}")
        print(f"- gallery_category_count: {inventory.get('gallery_category_count')}")
        print(f"- gallery_sample_count: {inventory.get('gallery_sample_count')}")
        print(f"- duplicate_image_id_count: {inventory.get('duplicate_image_id_count')}")
        print()
        print("## Registry Checks")
        for check in validation_checks:
            print(f"- {check['name']}: {check['status']} ({check['details']})")
        print()
        print("## Registry Entry")
        print(f"- artifact_id: {entry['artifact_id']}")
        print(f"- artifact_type: {entry['artifact_type']}")
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
        print("- production-ready claim: NOT MADE")
        print()
        print("## Final Verdict")
        print("PASS")
        return 0
    except Exception as exc:
        print("# Detection Sample Gallery Artifact Registry Update")
        print()
        print("## Source Artifact")
        print(f"- path: {_repo_relative(artifact_path)}")
        print(f"- exists: {'PASS' if artifact_path.is_file() else 'FAIL'}")
        print(f"- valid_json: {'PASS' if artifact_path.is_file() else 'FAIL'}")
        print(f"- size_bytes: {artifact_path.stat().st_size if artifact_path.is_file() else 0}")
        print(
            "- sha256: "
            f"{inventory.get('source_sample_gallery_artifact_sha256') if 'inventory' in locals() and isinstance(inventory, dict) else ''}"
        )
        print()
        print("## Inventory")
        print(f"- path: {_repo_relative(inventory_path)}")
        print(
            "- inventory_status: "
            f"{inventory.get('inventory_status') if 'inventory' in locals() and isinstance(inventory, dict) else 'fail'}"
        )
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
        print("- production-ready claim: NOT MADE")
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
        "artifact_hash": inventory["source_sample_gallery_artifact_sha256"],
        "artifact_size_bytes": inventory["source_sample_gallery_artifact_size_bytes"],
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
        "source_sample_gallery_artifact_path",
        "source_sample_gallery_artifact_sha256",
        "source_sample_gallery_artifact_size_bytes",
        "source_sample_gallery_artifact_exists",
        "source_sample_gallery_artifact_valid_json",
        "image_count",
        "total_bbox_count",
        "gallery_category_count",
        "gallery_sample_count",
        "category_ids",
        "category_sample_counts",
        "duplicate_image_id_count",
        "source_bbox_prediction_artifact_path",
        "source_bbox_prediction_artifact_hash",
        "source_per_image_summary_artifact_path",
        "source_per_image_summary_artifact_hash",
        "source_confidence_distribution_artifact_path",
        "source_confidence_distribution_artifact_hash",
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
    if inventory.get("image_count") != EXPECTED_IMAGE_COUNT:
        raise ValueError("inventory image_count mismatch.")
    if inventory.get("total_bbox_count") != EXPECTED_TOTAL_BBOX_COUNT:
        raise ValueError("inventory total_bbox_count mismatch.")
    if inventory.get("gallery_category_count") != EXPECTED_GALLERY_CATEGORY_COUNT:
        raise ValueError("inventory gallery_category_count mismatch.")
    if inventory.get("gallery_sample_count") != EXPECTED_GALLERY_SAMPLE_COUNT:
        raise ValueError("inventory gallery_sample_count mismatch.")
    if inventory.get("duplicate_image_id_count") != EXPECTED_DUPLICATE_IMAGE_ID_COUNT:
        raise ValueError("inventory duplicate_image_id_count mismatch.")
    if set(inventory.get("category_ids", [])) != EXPECTED_CATEGORY_IDS:
        raise ValueError("inventory category_ids mismatch.")
    if not all(count <= 5 for count in inventory.get("category_sample_counts", {}).values()):
        raise ValueError("inventory category_sample_counts must be <= 5.")
    for check in inventory.get("validation_checks", []):
        if isinstance(check, dict) and check.get("status") != "PASS":
            raise ValueError("inventory validation_checks must all pass.")


def _validate_source_artifact(
    source_artifact: dict[str, Any],
    inventory: dict[str, Any],
    artifact_path: Path,
) -> None:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"source artifact not found: {_repo_relative(artifact_path)}")
    if inventory.get("source_sample_gallery_artifact_exists") is not True:
        raise ValueError("inventory must confirm the source artifact exists.")
    if inventory.get("source_sample_gallery_artifact_valid_json") is not True:
        raise ValueError("inventory must confirm the source artifact JSON is valid.")
    if inventory.get("source_sample_gallery_artifact_path") != _repo_relative(artifact_path):
        raise ValueError("inventory source sample gallery path mismatch.")

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
    if source_artifact.get("image_count") != EXPECTED_IMAGE_COUNT:
        raise ValueError("source artifact image_count mismatch.")
    if source_artifact.get("total_bbox_count") != EXPECTED_TOTAL_BBOX_COUNT:
        raise ValueError("source artifact total_bbox_count mismatch.")
    if source_artifact.get("gallery_category_count") != EXPECTED_GALLERY_CATEGORY_COUNT:
        raise ValueError("source artifact gallery_category_count mismatch.")
    if source_artifact.get("gallery_sample_count") != EXPECTED_GALLERY_SAMPLE_COUNT:
        raise ValueError("source artifact gallery_sample_count mismatch.")

    categories = source_artifact.get("categories")
    if not isinstance(categories, list):
        raise ValueError("source artifact categories must be a list.")
    category_ids = [category.get("category_id") for category in categories if isinstance(category, dict)]
    if set(category_ids) != EXPECTED_CATEGORY_IDS:
        raise ValueError("source artifact category_ids mismatch.")
    if len(categories) != EXPECTED_GALLERY_CATEGORY_COUNT:
        raise ValueError("source artifact category count mismatch.")
    if sum(int(category.get("sample_count", 0)) for category in categories) != EXPECTED_GALLERY_SAMPLE_COUNT:
        raise ValueError("source artifact gallery_sample_count does not match categories.")

    if _duplicate_image_id_count(categories) != EXPECTED_DUPLICATE_IMAGE_ID_COUNT:
        raise ValueError("source artifact duplicate image_id count mismatch.")
    if _sha256_file(artifact_path) != inventory.get("source_sample_gallery_artifact_sha256"):
        raise ValueError("source artifact hash mismatch with inventory.")
    if artifact_path.stat().st_size != inventory.get("source_sample_gallery_artifact_size_bytes"):
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
    if existing_by_id is not None and not _entries_match(existing_by_id, entry):
        raise ValueError("artifact_registry already contains artifact_id with different metadata.")

    for artifact in artifacts:
        if artifact.get("artifact_path") == entry["artifact_path"] and artifact.get("artifact_id") != entry["artifact_id"]:
            raise ValueError("artifact_registry already contains artifact_path under a different artifact_id.")

    return [
        _check(
            "artifact_id_duplicate_handling",
            "PASS",
            "no existing artifact_id found; identical existing metadata would be an idempotent no-op."
            if existing_by_id is None
            else "existing artifact_id has identical registry metadata.",
        ),
        _check("artifact_path_collision", "PASS", "no conflicting artifact_path entries detected."),
        _check("source_inventory_status", "PASS", "inventory_status is pass."),
        _check("source_hash_and_size", "PASS", "inventory hash and size match the source artifact."),
        _check("production_ready_claim_absent", "PASS", "registry entry does not add production readiness fields."),
    ]


def _register_entry(
    registry: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    artifacts = list(registry.get("artifacts", []))
    for existing_entry in artifacts:
        if existing_entry.get("artifact_id") == entry["artifact_id"]:
            if _entries_match(existing_entry, entry):
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
    if actual.get("artifact_hash") != inventory.get("source_sample_gallery_artifact_sha256"):
        raise ValueError("artifact registry hash mismatch after write.")
    if actual.get("artifact_size_bytes") != inventory.get("source_sample_gallery_artifact_size_bytes"):
        raise ValueError("artifact registry size mismatch after write.")
    if "production_ready" in actual or "deployment_candidate" in actual:
        raise ValueError("artifact registry entry must not claim production readiness.")


def _validate_no_side_effects(
    run_registry_hash_before: str,
    frontend_hash_before: str,
    notebooks_hash_before: str,
) -> None:
    run_registry_hash_after = _sha256_file(DEFAULT_RUN_REGISTRY_PATH) if DEFAULT_RUN_REGISTRY_PATH.is_file() else ""
    if run_registry_hash_after != run_registry_hash_before:
        raise ValueError("run_registry.yaml changed unexpectedly.")
    if _tree_hash(REPO_ROOT / "frontend") != frontend_hash_before:
        raise ValueError("frontend files changed unexpectedly.")
    if _tree_hash(REPO_ROOT / "notebooks") != notebooks_hash_before:
        raise ValueError("notebook files changed unexpectedly.")


def _check(name: str, status: str, details: str) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def _duplicate_image_id_count(categories: list[Any]) -> int:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for category in categories:
        if not isinstance(category, dict):
            continue
        samples = category.get("samples", [])
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            image_id = sample.get("image_id")
            if not isinstance(image_id, str):
                continue
            if image_id in seen:
                duplicates.add(image_id)
            seen.add(image_id)
    return len(duplicates)


def _entries_match(existing_entry: dict[str, Any], expected_entry: dict[str, Any]) -> bool:
    return all(existing_entry.get(field) == expected_entry.get(field) for field in ENTRY_COMPARISON_FIELDS)


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


def _tree_hash(path: Path) -> str:
    if not path.exists():
        return ""
    if path.is_file():
        return _sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(_repo_relative(child).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
        digest.update(b"\0")
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
