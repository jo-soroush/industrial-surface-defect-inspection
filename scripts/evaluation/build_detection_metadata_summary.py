"""Build a governed Detection/YOLO metadata summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "yolo_train_v0_1_0"
MODEL_CONFIG_PATH = REPO_ROOT / "configs/models/yolo.yaml"
EXPORT_MANIFEST_PATH = REPO_ROOT / "data/processed/gc10det_yolo/export_manifest.yaml"
DATASET_YAML_PATH = REPO_ROOT / "data/processed/gc10det_yolo/dataset.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a governed Detection/YOLO metadata summary JSON."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--training-result", default=None)
    parser.add_argument("--artifact-inventory", default=None)
    parser.add_argument("--run-config", default=None)
    parser.add_argument("--output-path", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id
    run_dir = Path(args.run_dir or REPO_ROOT / "artifacts/detection/yolo/runs" / run_id)
    training_result_path = Path(
        args.training_result
        or REPO_ROOT / "artifacts/models/analysis" / f"training_result__{run_id}.json"
    )
    inventory_path = Path(
        args.artifact_inventory
        or REPO_ROOT / "artifacts/models/inventory" / f"track_detection_artifact_inventory__{run_id}.json"
    )
    run_config_path = Path(args.run_config or REPO_ROOT / "configs/runs" / f"{run_id}.yaml")
    output_path = Path(
        args.output_path
        or REPO_ROOT
        / "artifacts/models/metadata"
        / f"track_detection_yolo_metadata_summary__{run_id}.json"
    )

    training_result = _load_json_file(training_result_path, "training result summary")
    inventory = _load_json_file(inventory_path, "artifact inventory")
    run_config = _load_yaml_file(run_config_path, "run config")
    model_config = _load_yaml_file(MODEL_CONFIG_PATH, "model config")
    export_manifest = _load_yaml_file(EXPORT_MANIFEST_PATH, "export manifest")
    dataset_yaml = _load_yaml_file(DATASET_YAML_PATH, "dataset yaml")

    _validate_inputs(
        run_id,
        run_dir,
        training_result_path,
        inventory_path,
        run_config_path,
        training_result,
        inventory,
        run_config,
        model_config,
        export_manifest,
        dataset_yaml,
    )

    summary = {
        "metadata_type": "track_detection_yolo_metadata_summary",
        "run_id": run_id,
        "track_id": "detection",
        "task_type": "object_detection",
        "run_status": training_result["training_status"],
        "execution_environment": training_result["execution_environment"],
        "model_identity": {
            "model_name": training_result["model"]["model_name"],
            "model_type": training_result["model"]["model_type"],
            "model_source": training_result["model"]["model_source"],
            "backend": training_result["model"]["backend"],
            "pretrained": training_result["model"]["pretrained"],
        },
        "dataset_identity": {
            "dataset_id": training_result["dataset"]["dataset_id"],
            "dataset_version": training_result["dataset"]["dataset_version"],
            "dataset_yaml_runtime_path": training_result["dataset"]["dataset_yaml_runtime_path"],
            "local_dataset_yaml_path": training_result["dataset"]["local_dataset_yaml_path"],
            "export_manifest_path": training_result["dataset"]["export_manifest_path"],
            "split_counts": training_result["dataset"]["split_counts"],
            "class_count": training_result["dataset"]["class_count"],
        },
        "config_identity": {
            "run_config_path": training_result["config"]["run_config_path"],
            "model_config_path": training_result["config"]["model_config_path"],
            "config_id": training_result["config"]["config_id"],
        },
        "training_parameters": training_result["training_parameters"],
        "planned_config_parameters": training_result["planned_config_parameters"],
        "metrics": training_result["metrics"],
        "artifact_linkage": {
            "training_result_path": _repo_relative(training_result_path),
            "artifact_inventory_path": _repo_relative(inventory_path),
            "run_directory": _repo_relative(run_dir),
            "best_checkpoint_path": training_result["artifacts"]["best_checkpoint_path"],
            "last_checkpoint_path": training_result["artifacts"]["last_checkpoint_path"],
            "results_csv_path": training_result["artifacts"]["results_csv_path"],
            "args_yaml_path": training_result["artifacts"]["args_yaml_path"],
        },
        "artifact_integrity": {
            "training_result_sha256": _sha256(training_result_path),
            "artifact_inventory_sha256": _sha256(inventory_path),
            "best_checkpoint_sha256": training_result["artifacts"]["best_checkpoint_sha256"],
            "last_checkpoint_sha256": training_result["artifacts"]["last_checkpoint_sha256"],
        },
        "governance_status": {
            "training_result_created": True,
            "artifact_inventory_created": True,
            "metadata_summary_created": True,
            "posthoc_log_created": False,
            "registry_updated": False,
            "evaluation_summary_created": False,
        },
        "known_limitations": [
            "This metadata summary describes a 1-epoch YOLO training run created in Colab.",
            "Metrics are not final production-quality performance.",
            "Posthoc log, registry updates, evaluation summary, and re-audit are handled in later steps.",
        ],
        "created_at": _utc_now_iso(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)

    print(f"output_path={output_path}")
    print(f"run_status={summary['run_status']}")
    print(f"mAP50={summary['metrics']['mAP50']}")
    print(f"mAP50_95={summary['metrics']['mAP50_95']}")
    print("metadata_summary_created=true")
    return 0


def _validate_inputs(
    run_id: str,
    run_dir: Path,
    training_result_path: Path,
    inventory_path: Path,
    run_config_path: Path,
    training_result: dict[str, Any],
    inventory: dict[str, Any],
    run_config: dict[str, Any],
    model_config: dict[str, Any],
    export_manifest: dict[str, Any],
    dataset_yaml: dict[str, Any],
) -> None:
    if training_result.get("result_type") != "detection_yolo_training_result":
        raise ValueError("training result summary result_type mismatch.")
    if training_result.get("training_status") != "success":
        raise ValueError("training result summary training_status must be success.")
    if training_result.get("run_id") != run_id:
        raise ValueError("training result summary run_id mismatch.")
    if training_result.get("track_id") != "detection":
        raise ValueError("training result summary track_id must be detection.")
    if training_result.get("task_type") != "object_detection":
        raise ValueError("training result summary task_type must be object_detection.")
    if training_result.get("model", {}).get("model_name") != "yolo":
        raise ValueError("training result model_name must be yolo.")
    if training_result.get("model", {}).get("model_type") != "yolo":
        raise ValueError("training result model_type must be yolo.")
    if training_result.get("dataset", {}).get("dataset_id") != "gc10det_detection":
        raise ValueError("training result dataset_id must be gc10det_detection.")
    if training_result.get("dataset", {}).get("dataset_version") != "gc10det_1.0":
        raise ValueError("training result dataset_version must be gc10det_1.0.")

    if inventory.get("inventory_type") != "track_detection_yolo_artifact_inventory":
        raise ValueError("inventory type mismatch.")
    if inventory.get("inventory_status") != "pass":
        raise ValueError("inventory_status must be pass.")
    if inventory.get("run_id") != run_id:
        raise ValueError("inventory run_id mismatch.")

    if not run_dir.is_dir():
        raise FileNotFoundError(f"YOLO run directory not found: {_repo_relative(run_dir)}")
    for required_path in (training_result_path, inventory_path, run_config_path, MODEL_CONFIG_PATH, EXPORT_MANIFEST_PATH, DATASET_YAML_PATH):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required file not found: {_repo_relative(required_path)}")

    _require_dict(run_config.get("identity"), "run_config.identity")
    _require_dict(run_config.get("model_identity"), "run_config.model_identity")
    _require_dict(run_config.get("dataset_binding"), "run_config.dataset_binding")

    if run_config["identity"].get("task_type") != "object_detection":
        raise ValueError("run config task_type must be object_detection.")
    if run_config["identity"].get("track_id") != "detection":
        raise ValueError("run config track_id must be detection.")
    if run_config["model_identity"].get("model_name") != "yolo":
        raise ValueError("run config model_name must be yolo.")
    if run_config["model_identity"].get("model_type") != "yolo":
        raise ValueError("run config model_type must be yolo.")

    model_source = (
        run_config.get("training_model_source")
        or model_config.get("training_model_source")
    )
    if not isinstance(model_source, str) or not model_source.strip():
        raise ValueError("a governed YOLO training model source must be declared.")
    if model_config.get("backend") != "ultralytics":
        raise ValueError("model config backend must be ultralytics.")

    if export_manifest.get("manifest_type") != "yolo_dataset_export_manifest":
        raise ValueError("export manifest type mismatch.")
    if export_manifest.get("dataset_id") != "gc10det_detection":
        raise ValueError("export manifest dataset_id mismatch.")
    if export_manifest.get("dataset_version") != "gc10det_1.0":
        raise ValueError("export manifest dataset_version mismatch.")
    if dataset_yaml.get("nc") != 10:
        raise ValueError("dataset yaml nc must be 10.")


def _load_yaml_file(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} YAML is invalid: {_repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} YAML must parse to a dictionary: {_repo_relative(path)}")
    return payload


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON is invalid: {_repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object: {_repo_relative(path)}")
    return payload


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
