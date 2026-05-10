"""Build a governed Detection/YOLO posthoc run log."""

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
RUN_ID = "yolo_train_v0_1_0"
TRACK_ID = "detection"
TASK_TYPE = "object_detection"
MODEL_NAME = "yolo"
MODEL_TYPE = "yolo"
DATASET_ID = "gc10det_detection"
DATASET_VERSION = "gc10det_1.0"
RUN_DIR = REPO_ROOT / "artifacts/detection/yolo/runs" / RUN_ID
TRAINING_RESULT_PATH = REPO_ROOT / "artifacts/models/analysis" / f"training_result__{RUN_ID}.json"
INVENTORY_PATH = (
    REPO_ROOT / "artifacts/models/inventory" / f"track_detection_artifact_inventory__{RUN_ID}.json"
)
METADATA_SUMMARY_PATH = (
    REPO_ROOT
    / "artifacts/models/metadata"
    / f"track_detection_yolo_metadata_summary__{RUN_ID}.json"
)
RESULTS_CSV_PATH = RUN_DIR / "results.csv"
ARGS_YAML_PATH = RUN_DIR / "args.yaml"
BEST_CHECKPOINT_PATH = RUN_DIR / "weights" / "best.pt"
LAST_CHECKPOINT_PATH = RUN_DIR / "weights" / "last.pt"
OUTPUT_PATH = (
    REPO_ROOT
    / "artifacts/models/logs"
    / f"track_detection_yolo_posthoc_run_log__{RUN_ID}.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a governed Detection/YOLO posthoc run log JSON."
    )
    return parser


def main() -> int:
    build_parser().parse_args()

    training_result = _load_json_file(TRAINING_RESULT_PATH, "training result summary")
    inventory = _load_json_file(INVENTORY_PATH, "artifact inventory")
    metadata_summary = _load_json_file(METADATA_SUMMARY_PATH, "metadata summary")

    results_csv_rows = _load_csv_rows(RESULTS_CSV_PATH)
    if not results_csv_rows:
        raise ValueError(f"results.csv does not contain any data rows: {_repo_relative(RESULTS_CSV_PATH)}")
    final_metrics_row = results_csv_rows[-1]
    run_args = _load_yaml_file(ARGS_YAML_PATH, "YOLO args")

    _validate_required_files(
        (
            TRAINING_RESULT_PATH,
            INVENTORY_PATH,
            METADATA_SUMMARY_PATH,
            RESULTS_CSV_PATH,
            ARGS_YAML_PATH,
            BEST_CHECKPOINT_PATH,
            LAST_CHECKPOINT_PATH,
        )
    )
    _validate_payloads(training_result, inventory, metadata_summary, final_metrics_row, run_args)

    runtime_summary = {
        "model_source": training_result["model"]["model_source"],
        "backend": training_result["model"]["backend"],
        "epochs": training_result["training_parameters"]["epochs"],
        "batch": training_result["training_parameters"]["batch"],
        "imgsz": training_result["training_parameters"]["imgsz"],
        "device": training_result["training_parameters"]["device"],
        "optimizer": training_result["training_parameters"]["optimizer"],
        "seed": training_result["training_parameters"]["seed"],
        "deterministic": training_result["training_parameters"]["deterministic"],
    }

    metrics_summary = {
        "precision": _coerce_float(final_metrics_row["metrics/precision(B)"], "metrics/precision(B)"),
        "recall": _coerce_float(final_metrics_row["metrics/recall(B)"], "metrics/recall(B)"),
        "mAP50": _coerce_float(final_metrics_row["metrics/mAP50(B)"], "metrics/mAP50(B)"),
        "mAP50_95": _coerce_float(final_metrics_row["metrics/mAP50-95(B)"], "metrics/mAP50-95(B)"),
    }

    posthoc_log = {
        "log_type": "track_detection_yolo_posthoc_run_log",
        "run_id": RUN_ID,
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "run_status": training_result["training_status"],
        "execution_environment": metadata_summary["execution_environment"],
        "timeline": [
            {
                "event": "governed_yolo_dataset_export_prepared",
                "status": "completed",
                "details": "GC10-DET was exported to YOLO format using the governed dataset export boundary.",
            },
            {
                "event": "colab_training_executed",
                "status": "completed",
                "details": "YOLO training was executed in Colab using the governed exported dataset and equivalent config values.",
            },
            {
                "event": "training_outputs_returned_to_local",
                "status": "completed",
                "details": "The Colab YOLO run directory was archived, downloaded, and restored under artifacts/detection/yolo/runs/yolo_train_v0_1_0.",
            },
            {
                "event": "artifact_inventory_created",
                "status": "completed",
                "details": "A governed artifact inventory was created with hashes, file sizes, required flags, and frontend-ready flags.",
            },
            {
                "event": "training_result_summary_created",
                "status": "completed",
                "details": "A governed training result summary was created from results.csv, args.yaml, inventory, and run outputs.",
            },
            {
                "event": "metadata_summary_created",
                "status": "completed",
                "details": "A governed metadata summary was created and linked to training result, inventory, metrics, checkpoints, and config identities.",
            },
            {
                "event": "registry_update",
                "status": "pending",
                "details": "Run and artifact registries have not been updated yet.",
            },
            {
                "event": "evaluation_summary",
                "status": "pending",
                "details": "Detection evaluation summary has not been created yet.",
            },
            {
                "event": "detection_reaudit",
                "status": "pending",
                "details": "Detection re-audit has not been performed yet.",
            },
        ],
        "runtime_summary": runtime_summary,
        "metrics_summary": metrics_summary,
        "artifact_references": {
            "training_result_path": _repo_relative(TRAINING_RESULT_PATH),
            "artifact_inventory_path": _repo_relative(INVENTORY_PATH),
            "metadata_summary_path": _repo_relative(METADATA_SUMMARY_PATH),
            "run_directory": _repo_relative(RUN_DIR),
            "best_checkpoint_path": _repo_relative(BEST_CHECKPOINT_PATH),
            "last_checkpoint_path": _repo_relative(LAST_CHECKPOINT_PATH),
            "results_csv_path": _repo_relative(RESULTS_CSV_PATH),
            "args_yaml_path": _repo_relative(ARGS_YAML_PATH),
        },
        "artifact_integrity": {
            "training_result_sha256": _sha256(TRAINING_RESULT_PATH),
            "artifact_inventory_sha256": _sha256(INVENTORY_PATH),
            "metadata_summary_sha256": _sha256(METADATA_SUMMARY_PATH),
            "best_checkpoint_sha256": _sha256(BEST_CHECKPOINT_PATH),
            "last_checkpoint_sha256": _sha256(LAST_CHECKPOINT_PATH),
            "results_csv_sha256": _sha256(RESULTS_CSV_PATH),
            "args_yaml_sha256": _sha256(ARGS_YAML_PATH),
        },
        "governance_status": {
            "training_completed": True,
            "artifact_inventory_created": True,
            "training_result_created": True,
            "metadata_summary_created": True,
            "posthoc_log_created": True,
            "registry_updated": False,
            "evaluation_summary_created": False,
            "detection_reaudit_completed": False,
        },
        "known_limitations": [
            "This posthoc log documents a 1-epoch YOLO training run executed in Colab.",
            "Metrics are not final production-quality model performance.",
            "Registry updates, evaluation summary, validation script, and Detection re-audit are handled in later steps.",
        ],
        "created_at": _utc_now_iso(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(posthoc_log, handle, indent=2, sort_keys=False)

    print(f"output_path={OUTPUT_PATH}")
    print(f"run_status={posthoc_log['run_status']}")
    print(f"mAP50={metrics_summary['mAP50']}")
    print(f"mAP50_95={metrics_summary['mAP50_95']}")
    print("posthoc_log_created=true")
    return 0


def _validate_payloads(
    training_result: dict[str, Any],
    inventory: dict[str, Any],
    metadata_summary: dict[str, Any],
    final_metrics_row: dict[str, str],
    run_args: dict[str, Any],
) -> None:
    if training_result.get("result_type") != "detection_yolo_training_result":
        raise ValueError("training result summary result_type mismatch.")
    if training_result.get("training_status") != "success":
        raise ValueError("training result summary training_status must be success.")
    if training_result.get("run_id") != RUN_ID:
        raise ValueError("training result summary run_id mismatch.")
    if training_result.get("track_id") != TRACK_ID:
        raise ValueError("training result summary track_id mismatch.")
    if training_result.get("task_type") != TASK_TYPE:
        raise ValueError("training result summary task_type mismatch.")
    if training_result.get("model", {}).get("model_name") != MODEL_NAME:
        raise ValueError("training result summary model_name must be yolo.")
    if training_result.get("model", {}).get("model_type") != MODEL_TYPE:
        raise ValueError("training result summary model_type must be yolo.")
    if training_result.get("dataset", {}).get("dataset_id") != DATASET_ID:
        raise ValueError("training result summary dataset_id mismatch.")
    if training_result.get("dataset", {}).get("dataset_version") != DATASET_VERSION:
        raise ValueError("training result summary dataset_version mismatch.")

    if inventory.get("inventory_type") != "track_detection_yolo_artifact_inventory":
        raise ValueError("artifact inventory type mismatch.")
    if inventory.get("inventory_status") != "pass":
        raise ValueError("artifact inventory_status must be pass.")
    if inventory.get("run_id") != RUN_ID:
        raise ValueError("artifact inventory run_id mismatch.")
    if inventory.get("track_id") != TRACK_ID:
        raise ValueError("artifact inventory track_id mismatch.")
    if inventory.get("task_type") != TASK_TYPE:
        raise ValueError("artifact inventory task_type mismatch.")
    if inventory.get("model_name") != MODEL_NAME:
        raise ValueError("artifact inventory model_name mismatch.")
    if inventory.get("model_type") != MODEL_TYPE:
        raise ValueError("artifact inventory model_type mismatch.")
    if inventory.get("dataset_id") != DATASET_ID:
        raise ValueError("artifact inventory dataset_id mismatch.")
    if inventory.get("dataset_version") != DATASET_VERSION:
        raise ValueError("artifact inventory dataset_version mismatch.")

    if metadata_summary.get("metadata_type") != "track_detection_yolo_metadata_summary":
        raise ValueError("metadata summary type mismatch.")
    if metadata_summary.get("run_status") != "success":
        raise ValueError("metadata summary run_status must be success.")
    if metadata_summary.get("run_id") != RUN_ID:
        raise ValueError("metadata summary run_id mismatch.")
    if metadata_summary.get("track_id") != TRACK_ID:
        raise ValueError("metadata summary track_id mismatch.")
    if metadata_summary.get("task_type") != TASK_TYPE:
        raise ValueError("metadata summary task_type mismatch.")
    if metadata_summary.get("execution_environment") != "colab":
        raise ValueError("metadata summary execution_environment must be colab.")

    governance = _require_dict(metadata_summary.get("governance_status"), "metadata_summary.governance_status")
    if governance.get("metadata_summary_created") is not True:
        raise ValueError("metadata summary governance flag metadata_summary_created must be true.")
    if governance.get("registry_updated") is not False:
        raise ValueError("metadata summary governance flag registry_updated must be false.")
    if governance.get("evaluation_summary_created") is not False:
        raise ValueError("metadata summary governance flag evaluation_summary_created must be false.")

    if not RUN_DIR.is_dir():
        raise FileNotFoundError(f"YOLO run directory not found: {_repo_relative(RUN_DIR)}")
    _validate_required_files(
        (
            TRAINING_RESULT_PATH,
            INVENTORY_PATH,
            METADATA_SUMMARY_PATH,
            RESULTS_CSV_PATH,
            ARGS_YAML_PATH,
            BEST_CHECKPOINT_PATH,
            LAST_CHECKPOINT_PATH,
        )
    )

    training_metrics = _require_dict(training_result.get("metrics"), "training_result.metrics")
    metadata_metrics = _require_dict(metadata_summary.get("metrics"), "metadata_summary.metrics")
    self_consistent_metrics = {
        "epoch": _coerce_int(final_metrics_row["epoch"], "epoch"),
        "time": _coerce_float(final_metrics_row["time"], "time"),
        "precision": _coerce_float(final_metrics_row["metrics/precision(B)"], "metrics/precision(B)"),
        "recall": _coerce_float(final_metrics_row["metrics/recall(B)"], "metrics/recall(B)"),
        "mAP50": _coerce_float(final_metrics_row["metrics/mAP50(B)"], "metrics/mAP50(B)"),
        "mAP50_95": _coerce_float(final_metrics_row["metrics/mAP50-95(B)"], "metrics/mAP50-95(B)"),
        "train_box_loss": _coerce_float(final_metrics_row["train/box_loss"], "train/box_loss"),
        "train_cls_loss": _coerce_float(final_metrics_row["train/cls_loss"], "train/cls_loss"),
        "train_dfl_loss": _coerce_float(final_metrics_row["train/dfl_loss"], "train/dfl_loss"),
        "val_box_loss": _coerce_float(final_metrics_row["val/box_loss"], "val/box_loss"),
        "val_cls_loss": _coerce_float(final_metrics_row["val/cls_loss"], "val/cls_loss"),
        "val_dfl_loss": _coerce_float(final_metrics_row["val/dfl_loss"], "val/dfl_loss"),
    }
    _assert_metric_subset(training_metrics, self_consistent_metrics, "training result metrics")
    _assert_metric_subset(metadata_metrics, self_consistent_metrics, "metadata summary metrics")

    if run_args.get("task") != "detect":
        raise ValueError("YOLO args task must be detect.")
    if run_args.get("mode") != "train":
        raise ValueError("YOLO args mode must be train.")
    if run_args.get("model") != "yolov8n.pt":
        raise ValueError("YOLO args model must be yolov8n.pt.")
    if run_args.get("data") != metadata_summary["dataset_identity"]["dataset_yaml_runtime_path"]:
        raise ValueError("YOLO args data must match metadata dataset_yaml_runtime_path.")
    if run_args.get("epochs") != metadata_summary["training_parameters"]["epochs"]:
        raise ValueError("YOLO args epochs must match training parameters.")
    if run_args.get("batch") != metadata_summary["training_parameters"]["batch"]:
        raise ValueError("YOLO args batch must match training parameters.")
    if run_args.get("imgsz") != metadata_summary["training_parameters"]["imgsz"]:
        raise ValueError("YOLO args imgsz must match training parameters.")
    if run_args.get("device") != metadata_summary["training_parameters"]["device"]:
        raise ValueError("YOLO args device must match training parameters.")
    if run_args.get("optimizer") != metadata_summary["training_parameters"]["optimizer"]:
        raise ValueError("YOLO args optimizer must match training parameters.")
    if run_args.get("seed") != metadata_summary["training_parameters"]["seed"]:
        raise ValueError("YOLO args seed must match training parameters.")
    if run_args.get("deterministic") != metadata_summary["training_parameters"]["deterministic"]:
        raise ValueError("YOLO args deterministic must match training parameters.")


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


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"results.csv not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _validate_required_files(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {_repo_relative(path)}")


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def _assert_metric_subset(
    payload: dict[str, Any],
    expected: dict[str, Any],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in payload:
            raise ValueError(f"{label} missing required metric field: {key}")
        actual_value = payload[key]
        if isinstance(expected_value, float):
            if abs(_coerce_float(actual_value, key) - expected_value) > 1e-9:
                raise ValueError(f"{label} field {key} does not match results.csv.")
        elif actual_value != expected_value:
            raise ValueError(f"{label} field {key} does not match results.csv.")


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


if __name__ == "__main__":
    raise SystemExit(main())
