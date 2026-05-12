"""Build a governed Detection/YOLO training result summary."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "yolo_train_v0_1_0"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "configs/models/yolo.yaml"
DEFAULT_EXPORT_MANIFEST_PATH = REPO_ROOT / "data/processed/gc10det_yolo/export_manifest.yaml"
DEFAULT_DATASET_YAML_PATH = REPO_ROOT / "data/processed/gc10det_yolo/dataset.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a governed Detection/YOLO training result summary JSON."
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Governed Detection/YOLO run id.",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help="Path to the governed YOLO run config.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to the YOLO training run directory.",
    )
    parser.add_argument(
        "--inventory-path",
        default=None,
        help="Path to the governed Detection inventory JSON.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to the training result summary JSON to write.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    run_id = args.run_id
    run_dir = Path(args.run_dir or REPO_ROOT / "artifacts/detection/yolo/runs" / run_id)
    inventory_path = Path(
        args.inventory_path
        or REPO_ROOT / "artifacts/models/inventory" / f"track_detection_artifact_inventory__{run_id}.json"
    )
    output_path = Path(
        args.output_path
        or REPO_ROOT / "artifacts/models/analysis" / f"training_result__{run_id}.json"
    )
    run_config_path = Path(args.run_config or REPO_ROOT / "configs/runs" / f"{run_id}.yaml")

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"YOLO run directory not found: {_repo_relative(run_dir)}"
        )

    run_config = _load_yaml_file(run_config_path, "run config")
    model_config = _load_yaml_file(DEFAULT_MODEL_CONFIG_PATH, "model config")
    export_manifest = _load_yaml_file(DEFAULT_EXPORT_MANIFEST_PATH, "export manifest")
    dataset_yaml = _load_yaml_file(DEFAULT_DATASET_YAML_PATH, "dataset yaml")
    inventory = _load_json_file(inventory_path, "detection inventory")

    args_yaml_path = _require_file(run_dir / "args.yaml", "args.yaml")
    results_csv_path = _require_file(run_dir / "results.csv", "results.csv")
    best_checkpoint_path = _require_file(run_dir / "weights" / "best.pt", "weights/best.pt")
    last_checkpoint_path = _require_file(run_dir / "weights" / "last.pt", "weights/last.pt")

    _validate_config(run_config, model_config, export_manifest, dataset_yaml, inventory)
    model_source = (
        run_config.get("training_model_source")
        or model_config.get("training_model_source")
    )

    metrics_summary = _build_metrics_summary(results_csv_path)
    run_args = _load_yaml_file(args_yaml_path, "YOLO args")
    run_args_summary = _build_run_args_summary(args_yaml_path, run_args)
    training_parameters = _build_training_parameters(run_args)
    artifact_details = _build_artifact_details(
        run_dir=run_dir,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        results_csv_path=results_csv_path,
        args_yaml_path=args_yaml_path,
        inventory_path=inventory_path,
    )

    summary = {
        "result_type": "detection_yolo_training_result",
        "run_id": run_id,
        "track_id": "detection",
        "task_type": "object_detection",
        "training_status": "success",
        "execution_environment": "colab",
        "model": {
            "model_name": "yolo",
            "model_type": "yolo",
            "model_source": model_source,
            "backend": model_config["backend"],
            "pretrained": model_config.get("pretrained"),
        },
        "dataset": {
            "dataset_id": export_manifest["dataset_id"],
            "dataset_version": export_manifest["dataset_version"],
            "dataset_yaml_runtime_path": run_args.get("data"),
            "local_dataset_yaml_path": _repo_relative(DEFAULT_DATASET_YAML_PATH),
            "export_manifest_path": _repo_relative(DEFAULT_EXPORT_MANIFEST_PATH),
            "split_counts": export_manifest["split_counts"],
            "class_count": dataset_yaml["nc"],
        },
        "config": {
            "run_config_path": _repo_relative(run_config_path),
            "model_config_path": _repo_relative(DEFAULT_MODEL_CONFIG_PATH),
            "config_id": run_config["identity"]["run_config_id"],
        },
        "training_parameters": training_parameters,
        "planned_config_parameters": _build_planned_config_parameters(run_config),
        "run_args_summary": run_args_summary,
        "metrics": metrics_summary,
        "artifacts": {
            "run_directory": _repo_relative(run_dir),
            "best_checkpoint_path": _repo_relative(best_checkpoint_path),
            "best_checkpoint_sha256": _sha256(best_checkpoint_path),
            "best_checkpoint_size_bytes": best_checkpoint_path.stat().st_size,
            "last_checkpoint_path": _repo_relative(last_checkpoint_path),
            "last_checkpoint_sha256": _sha256(last_checkpoint_path),
            "last_checkpoint_size_bytes": last_checkpoint_path.stat().st_size,
            "results_csv_path": _repo_relative(results_csv_path),
            "results_csv_sha256": _sha256(results_csv_path),
            "results_csv_size_bytes": results_csv_path.stat().st_size,
            "args_yaml_path": _repo_relative(args_yaml_path),
            "args_yaml_sha256": _sha256(args_yaml_path),
            "args_yaml_size_bytes": args_yaml_path.stat().st_size,
            "inventory_path": _repo_relative(inventory_path),
            "inventory_sha256": _sha256(inventory_path),
            "inventory_size_bytes": inventory_path.stat().st_size,
            "inventory_status": inventory.get("inventory_status"),
            "inventory_artifact_count": inventory.get("artifact_count"),
            "files": artifact_details,
        },
        "governance": {
            "artifact_inventory_created": True,
            "registry_updated": False,
            "metadata_summary_created": False,
            "posthoc_log_created": False,
            "evaluation_summary_created": False,
        },
        "known_limitations": [
            "This is a 1-epoch governed YOLO execution intended to prove the training pipeline and produce first detection artifacts.",
            "Metrics are not final production-quality model performance.",
            "Execution was performed in Colab using the governed dataset export and equivalent config values.",
            "Registry updates and metadata summaries are handled in later governance steps.",
        ],
        "created_at": _utc_now_iso(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)

    print(f"output_path={output_path}")
    print("training_status=success")
    print(f"mAP50={metrics_summary['mAP50']}")
    print(f"mAP50_95={metrics_summary['mAP50_95']}")
    print(f"inventory_status={inventory.get('inventory_status')}")
    return 0


def _validate_config(
    run_config: dict[str, Any],
    model_config: dict[str, Any],
    export_manifest: dict[str, Any],
    dataset_yaml: dict[str, Any],
    inventory: dict[str, Any],
) -> None:
    identity = _require_dict(run_config.get("identity"), "run_config.identity")
    model_identity = _require_dict(run_config.get("model_identity"), "run_config.model_identity")
    dataset_binding = _require_dict(run_config.get("dataset_binding"), "run_config.dataset_binding")

    if identity.get("task_type") != "object_detection":
        raise ValueError("Run config identity.task_type must be object_detection.")
    if identity.get("track_id") != "detection":
        raise ValueError("Run config identity.track_id must be detection.")
    if model_identity.get("model_name") != "yolo":
        raise ValueError("Run config model_identity.model_name must be yolo.")
    if model_identity.get("model_type") != "yolo":
        raise ValueError("Run config model_identity.model_type must be yolo.")
    if dataset_binding.get("dataset_id") != "gc10det_detection":
        raise ValueError("Run config dataset_binding.dataset_id must be gc10det_detection.")
    if dataset_binding.get("dataset_version") != "gc10det_1.0":
        raise ValueError("Run config dataset_binding.dataset_version must be gc10det_1.0.")
    if dataset_binding.get("split_manifest_path") != "data/manifests/split_gc10det_detection.yaml":
        raise ValueError("Run config split_manifest_path must point to the governed GC10-DET split manifest.")

    if model_config.get("backend") != "ultralytics":
        raise ValueError("YOLO model config backend must be ultralytics.")
    if model_config.get("backend_package") != "ultralytics":
        raise ValueError("YOLO model config backend_package must be ultralytics.")
    model_source = (
        run_config.get("training_model_source")
        or model_config.get("training_model_source")
    )
    if not isinstance(model_source, str) or not model_source.strip():
        raise ValueError("a governed YOLO training model source must be declared.")
    if model_config.get("dataset_id") != "gc10det_detection":
        raise ValueError("YOLO model config dataset_id must be gc10det_detection.")
    if model_config.get("dataset_version") != "gc10det_1.0":
        raise ValueError("YOLO model config dataset_version must be gc10det_1.0.")

    if export_manifest.get("manifest_type") != "yolo_dataset_export_manifest":
        raise ValueError("Export manifest type must be yolo_dataset_export_manifest.")
    if export_manifest.get("dataset_id") != "gc10det_detection":
        raise ValueError("Export manifest dataset_id must be gc10det_detection.")
    if export_manifest.get("dataset_version") != "gc10det_1.0":
        raise ValueError("Export manifest dataset_version must be gc10det_1.0.")

    if dataset_yaml.get("nc") != 10:
        raise ValueError("Dataset YAML nc must be 10.")
    names = dataset_yaml.get("names")
    if not isinstance(names, list) or len(names) != 10:
        raise ValueError("Dataset YAML names must contain 10 class labels.")

    if inventory.get("inventory_type") != "track_detection_yolo_artifact_inventory":
        raise ValueError("Detection inventory type mismatch.")
    if inventory.get("inventory_status") != "pass":
        raise ValueError("Detection inventory_status must be pass.")


def _build_metrics_summary(results_csv_path: Path) -> dict[str, Any]:
    rows = _load_csv_rows(results_csv_path)
    if not rows:
        raise ValueError(
            f"results.csv does not contain any data rows: {_repo_relative(results_csv_path)}"
        )
    final_row = rows[-1]
    required_columns = [
        "epoch",
        "time",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    missing = [column for column in required_columns if column not in final_row]
    if missing:
        raise ValueError(f"results.csv is missing required columns: {missing}")

    return {
        "source_file": _repo_relative(results_csv_path),
        "row_count": len(rows),
        "epoch": _coerce_int(final_row["epoch"], "epoch"),
        "time": _coerce_float(final_row["time"], "time"),
        "precision": _coerce_float(final_row["metrics/precision(B)"], "metrics/precision(B)"),
        "recall": _coerce_float(final_row["metrics/recall(B)"], "metrics/recall(B)"),
        "mAP50": _coerce_float(final_row["metrics/mAP50(B)"], "metrics/mAP50(B)"),
        "mAP50_95": _coerce_float(final_row["metrics/mAP50-95(B)"], "metrics/mAP50-95(B)"),
        "train_box_loss": _coerce_float(final_row["train/box_loss"], "train/box_loss"),
        "train_cls_loss": _coerce_float(final_row["train/cls_loss"], "train/cls_loss"),
        "train_dfl_loss": _coerce_float(final_row["train/dfl_loss"], "train/dfl_loss"),
        "val_box_loss": _coerce_float(final_row["val/box_loss"], "val/box_loss"),
        "val_cls_loss": _coerce_float(final_row["val/cls_loss"], "val/cls_loss"),
        "val_dfl_loss": _coerce_float(final_row["val/dfl_loss"], "val/dfl_loss"),
    }


def _build_run_args_summary(args_yaml_path: Path, run_args: dict[str, Any]) -> dict[str, Any]:
    required_fields = [
        "task",
        "mode",
        "model",
        "data",
        "epochs",
        "batch",
        "imgsz",
        "device",
        "project",
        "name",
        "pretrained",
        "seed",
        "deterministic",
        "optimizer",
        "save_dir",
    ]
    missing = [field for field in required_fields if field not in run_args]
    if missing:
        raise ValueError(f"args.yaml is missing required keys: {missing}")

    return {
        "source_file": _repo_relative(args_yaml_path),
        "task": run_args["task"],
        "mode": run_args["mode"],
        "model": run_args["model"],
        "data": run_args["data"],
        "epochs": run_args["epochs"],
        "batch": run_args["batch"],
        "imgsz": run_args["imgsz"],
        "device": run_args["device"],
        "project": run_args["project"],
        "name": run_args["name"],
        "pretrained": run_args["pretrained"],
        "seed": run_args["seed"],
        "deterministic": run_args["deterministic"],
        "optimizer": run_args["optimizer"],
        "save_dir": run_args["save_dir"],
    }


def _build_training_parameters(
    run_args: dict[str, Any],
) -> dict[str, Any]:
    required_fields = [
        "epochs",
        "batch",
        "imgsz",
        "seed",
        "device",
        "optimizer",
        "deterministic",
    ]
    missing = [field for field in required_fields if field not in run_args]
    if missing:
        raise ValueError(f"args.yaml is missing required keys for training_parameters: {missing}")
    return {
        "epochs": run_args["epochs"],
        "batch": run_args["batch"],
        "imgsz": run_args.get("imgsz"),
        "seed": run_args["seed"],
        "device": run_args["device"],
        "optimizer": run_args["optimizer"],
        "deterministic": run_args["deterministic"],
    }


def _build_planned_config_parameters(run_config: dict[str, Any]) -> dict[str, Any]:
    runtime = _require_dict(run_config.get("training_runtime"), "run_config.training_runtime")
    return {
        "epochs": runtime.get("epochs"),
        "batch_size": runtime.get("batch_size"),
        "learning_rate": runtime.get("learning_rate"),
        "optimizer": runtime.get("optimizer"),
        "loss_function": runtime.get("loss_function"),
        "seed": runtime.get("seed"),
        "device": runtime.get("device"),
    }


def _build_artifact_details(
    run_dir: Path,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    results_csv_path: Path,
    args_yaml_path: Path,
    inventory_path: Path,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = [
        _artifact_entry("best_model_checkpoint", best_checkpoint_path, True, False),
        _artifact_entry("last_model_checkpoint", last_checkpoint_path, True, False),
        _artifact_entry("training_metrics_csv", results_csv_path, True, True),
        _artifact_entry("training_args_yaml", args_yaml_path, True, True),
        _artifact_entry("detection_inventory", inventory_path, True, True),
    ]

    optional_files = [
        ("results.png", "training_results_plot"),
        ("confusion_matrix.png", "confusion_matrix_plot"),
        ("confusion_matrix_normalized.png", "normalized_confusion_matrix_plot"),
        ("BoxPR_curve.png", "precision_recall_curve_plot"),
        ("BoxF1_curve.png", "f1_curve_plot"),
        ("BoxP_curve.png", "precision_curve_plot"),
        ("BoxR_curve.png", "recall_curve_plot"),
        ("labels.jpg", "label_distribution_visualization"),
    ]
    for filename, role in optional_files:
        path = run_dir / filename
        if path.is_file():
            artifacts.append(_artifact_entry(role, path, False, True))

    for path in sorted(run_dir.glob("train_batch*.jpg")):
        artifacts.append(_artifact_entry("training_batch_visualization", path, False, True))
    for path in sorted(run_dir.glob("val_batch*_labels.jpg")):
        artifacts.append(_artifact_entry("validation_label_visualization", path, False, True))
    for path in sorted(run_dir.glob("val_batch*_pred.jpg")):
        artifacts.append(_artifact_entry("validation_prediction_visualization", path, False, True))

    return artifacts


def _artifact_entry(
    artifact_role: str,
    path: Path,
    required: bool,
    frontend_ready: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {_repo_relative(path)}")
    return {
        "artifact_role": artifact_role,
        "path": _repo_relative(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "required": required,
        "frontend_ready": frontend_ready,
    }


def _load_yaml_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
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


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    return path


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if numeric.is_integer():
        return int(numeric)
    raise ValueError(f"{field_name} must be an integer-like value.")


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
