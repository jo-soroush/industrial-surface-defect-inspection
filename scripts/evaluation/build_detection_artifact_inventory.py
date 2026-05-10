"""Build a governed Detection/YOLO artifact inventory."""

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
DEFAULT_RUN_DIR = REPO_ROOT / "artifacts/detection/yolo/runs" / DEFAULT_RUN_ID
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "artifacts/models/inventory"
    / f"track_detection_artifact_inventory__{DEFAULT_RUN_ID}.json"
)
DEFAULT_RUN_CONFIG_PATH = REPO_ROOT / "configs/runs/yolo_train_v0_1_0.yaml"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "configs/models/yolo.yaml"

REQUIRED_ARTIFACTS: list[tuple[str, str, bool, bool]] = [
    ("weights/best.pt", "best_model_checkpoint", True, False),
    ("weights/last.pt", "last_model_checkpoint", True, False),
    ("results.csv", "training_metrics_csv", True, True),
    ("args.yaml", "training_args_yaml", True, True),
]

OPTIONAL_ARTIFACT_PATTERNS: list[tuple[str, str, bool]] = [
    ("results.png", "training_results_plot", True),
    ("confusion_matrix.png", "confusion_matrix_plot", True),
    ("confusion_matrix_normalized.png", "normalized_confusion_matrix_plot", True),
    ("BoxPR_curve.png", "precision_recall_curve_plot", True),
    ("BoxF1_curve.png", "f1_curve_plot", True),
    ("BoxP_curve.png", "precision_curve_plot", True),
    ("BoxR_curve.png", "recall_curve_plot", True),
    ("labels.jpg", "label_distribution_visualization", True),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a governed Detection/YOLO artifact inventory JSON."
    )
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Path to the YOLO training run directory.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the inventory JSON to write.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output_path)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"YOLO run directory not found: {_repo_relative(run_dir)}")

    run_config = _load_yaml_file(DEFAULT_RUN_CONFIG_PATH, "run config")
    model_config = _load_yaml_file(DEFAULT_MODEL_CONFIG_PATH, "model config")
    args_yaml_path = _require_file(run_dir / "args.yaml", "args.yaml")
    results_csv_path = _require_file(run_dir / "results.csv", "results.csv")
    _require_file(run_dir / "weights" / "best.pt", "weights/best.pt")
    _require_file(run_dir / "weights" / "last.pt", "weights/last.pt")

    _validate_detection_config(run_config, model_config)
    run_args = _load_yaml_file(args_yaml_path, "YOLO args")
    metrics_rows = _load_csv_rows(results_csv_path)

    if not metrics_rows:
        raise ValueError(f"results.csv does not contain any data rows: {_repo_relative(results_csv_path)}")

    metrics_summary = _build_metrics_summary(results_csv_path, metrics_rows)
    run_args_summary = _build_run_args_summary(args_yaml_path, run_args)
    artifacts = _collect_artifacts(run_dir)

    inventory = {
        "inventory_type": "track_detection_yolo_artifact_inventory",
        "track_id": "detection",
        "task_type": "object_detection",
        "run_id": DEFAULT_RUN_ID,
        "model_name": "yolo",
        "model_type": "yolo",
        "model_source": model_config["training_model_source"],
        "backend": model_config["backend"],
        "dataset_id": run_config["dataset_binding"]["dataset_id"],
        "dataset_version": run_config["dataset_binding"]["dataset_version"],
        "config_id": run_config["identity"]["run_config_id"],
        "config_paths": {
            "run_config_path": _repo_relative(DEFAULT_RUN_CONFIG_PATH),
            "model_config_path": _repo_relative(DEFAULT_MODEL_CONFIG_PATH),
            "split_manifest_path": _repo_relative(
                REPO_ROOT / run_config["dataset_binding"]["split_manifest_path"]
            ),
        },
        "source_run_directory": _repo_relative(run_dir),
        "dataset_yaml_path": _require_string(
            run_args.get("data"), "args.yaml.data"
        ),
        "training_command": "python scripts/detection/train_yolo_detection.py --run-training",
        "runtime_environment": {
            "task": run_args.get("task"),
            "mode": run_args.get("mode"),
            "device": run_args.get("device"),
            "project": run_args.get("project"),
            "name": run_args.get("name"),
            "save_dir": run_args.get("save_dir"),
            "pretrained": run_args.get("pretrained"),
            "optimizer": run_args.get("optimizer"),
            "seed": run_args.get("seed"),
            "deterministic": run_args.get("deterministic"),
        },
        "required_files_status": "pass",
        "metrics_summary": metrics_summary,
        "run_args_summary": run_args_summary,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "inventory_status": "pass",
        "created_at": _utc_now_iso(),
        "known_limitations": [
            "This inventory summarizes the runtime outputs of a single YOLO detection run.",
            "Artifact registration and governed metadata summaries are handled separately.",
            "Only files present in the run directory are inventoried; missing optional diagnostics are excluded.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2, sort_keys=False)

    print(f"inventory_path={output_path}")
    print(f"inventory_status={inventory['inventory_status']}")
    print(f"artifact_count={inventory['artifact_count']}")
    return 0


def _validate_detection_config(run_config: dict[str, Any], model_config: dict[str, Any]) -> None:
    identity = _require_dict(run_config.get("identity"), "run_config.identity")
    model_identity = _require_dict(run_config.get("model_identity"), "run_config.model_identity")
    dataset_binding = _require_dict(run_config.get("dataset_binding"), "run_config.dataset_binding")

    if identity.get("task_type") != "object_detection":
        raise ValueError("Run config identity.task_type must be object_detection.")
    if model_identity.get("model_name") != "yolo":
        raise ValueError("Run config model_identity.model_name must be yolo.")
    if model_identity.get("model_type") != "yolo":
        raise ValueError("Run config model_identity.model_type must be yolo.")
    if dataset_binding.get("dataset_id") != "gc10det_detection":
        raise ValueError("Run config dataset_binding.dataset_id must be gc10det_detection.")
    if dataset_binding.get("dataset_version") != "gc10det_1.0":
        raise ValueError("Run config dataset_binding.dataset_version must be gc10det_1.0.")

    if model_config.get("backend") != "ultralytics":
        raise ValueError("YOLO model config backend must be ultralytics.")
    if model_config.get("backend_package") != "ultralytics":
        raise ValueError("YOLO model config backend_package must be ultralytics.")
    if model_config.get("training_model_source") != "yolov8n.pt":
        raise ValueError("YOLO model config training_model_source must be yolov8n.pt.")


def _build_metrics_summary(results_csv_path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
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
        "lr/pg0",
        "lr/pg1",
        "lr/pg2",
    ]
    missing = [column for column in required_columns if column not in final_row]
    if missing:
        raise ValueError(f"results.csv is missing required columns: {missing}")

    return {
        "source_file": _repo_relative(results_csv_path),
        "row_count": len(rows),
        "final_row": {
            "epoch": _coerce_int(final_row["epoch"], "epoch"),
            "time": _coerce_float(final_row["time"], "time"),
            "train/box_loss": _coerce_float(final_row["train/box_loss"], "train/box_loss"),
            "train/cls_loss": _coerce_float(final_row["train/cls_loss"], "train/cls_loss"),
            "train/dfl_loss": _coerce_float(final_row["train/dfl_loss"], "train/dfl_loss"),
            "metrics/precision(B)": _coerce_float(
                final_row["metrics/precision(B)"], "metrics/precision(B)"
            ),
            "metrics/recall(B)": _coerce_float(final_row["metrics/recall(B)"], "metrics/recall(B)"),
            "metrics/mAP50(B)": _coerce_float(final_row["metrics/mAP50(B)"], "metrics/mAP50(B)"),
            "metrics/mAP50-95(B)": _coerce_float(
                final_row["metrics/mAP50-95(B)"], "metrics/mAP50-95(B)"
            ),
            "val/box_loss": _coerce_float(final_row["val/box_loss"], "val/box_loss"),
            "val/cls_loss": _coerce_float(final_row["val/cls_loss"], "val/cls_loss"),
            "val/dfl_loss": _coerce_float(final_row["val/dfl_loss"], "val/dfl_loss"),
            "lr/pg0": _coerce_float(final_row["lr/pg0"], "lr/pg0"),
            "lr/pg1": _coerce_float(final_row["lr/pg1"], "lr/pg1"),
            "lr/pg2": _coerce_float(final_row["lr/pg2"], "lr/pg2"),
        },
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


def _collect_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []

    for relative_path, artifact_role, required, frontend_ready in REQUIRED_ARTIFACTS:
        artifact_path = run_dir / relative_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Required YOLO artifact missing: {_repo_relative(artifact_path)}")
        artifacts.append(
            _artifact_entry(
                artifact_role=artifact_role,
                path=artifact_path,
                required=required,
                frontend_ready=frontend_ready,
            )
        )

    for filename, artifact_role, frontend_ready in OPTIONAL_ARTIFACT_PATTERNS:
        artifact_path = run_dir / filename
        if artifact_path.is_file():
            artifacts.append(
                _artifact_entry(
                    artifact_role=artifact_role,
                    path=artifact_path,
                    required=False,
                    frontend_ready=frontend_ready,
                )
            )

    for artifact_path in sorted(run_dir.glob("train_batch*.jpg")):
        artifacts.append(
            _artifact_entry(
                artifact_role="training_batch_visualization",
                path=artifact_path,
                required=False,
                frontend_ready=True,
            )
        )

    for artifact_path in sorted(run_dir.glob("val_batch*_labels.jpg")):
        artifacts.append(
            _artifact_entry(
                artifact_role="validation_label_visualization",
                path=artifact_path,
                required=False,
                frontend_ready=True,
            )
        )

    for artifact_path in sorted(run_dir.glob("val_batch*_pred.jpg")):
        artifacts.append(
            _artifact_entry(
                artifact_role="validation_prediction_visualization",
                path=artifact_path,
                required=False,
                frontend_ready=True,
            )
        )

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
        raise FileNotFoundError(f"{label} file not found: {_repo_relative(path)}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} YAML is invalid: {_repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} YAML must contain an object: {_repo_relative(path)}")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    return path


def _coerce_int(value: str, field_name: str) -> int:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if numeric.is_integer():
        return int(numeric)
    raise ValueError(f"{field_name} must be an integer-like value.")


def _coerce_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
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


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
