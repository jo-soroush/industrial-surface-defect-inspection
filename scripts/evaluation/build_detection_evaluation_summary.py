"""Build a governed Detection/YOLO validation evaluation summary."""

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
POSTHOC_LOG_PATH = (
    REPO_ROOT
    / "artifacts/models/logs"
    / f"track_detection_yolo_posthoc_run_log__{RUN_ID}.json"
)
RESULTS_CSV_PATH = RUN_DIR / "results.csv"
ARGS_YAML_PATH = RUN_DIR / "args.yaml"
EXPORT_MANIFEST_PATH = REPO_ROOT / "data/processed/gc10det_yolo/export_manifest.yaml"
DATASET_YAML_PATH = REPO_ROOT / "data/processed/gc10det_yolo/dataset.yaml"
OUTPUT_PATH = (
    REPO_ROOT
    / "artifacts/models/metrics"
    / f"detection_evaluation__{RUN_ID}__validation.json"
)
EVIDENCE_FILE_SPECS = [
    ("training_metrics_csv", RESULTS_CSV_PATH, True),
    ("training_args_yaml", ARGS_YAML_PATH, True),
    ("confusion_matrix_plot", RUN_DIR / "confusion_matrix.png", True),
    (
        "normalized_confusion_matrix_plot",
        RUN_DIR / "confusion_matrix_normalized.png",
        True,
    ),
    ("precision_recall_curve_plot", RUN_DIR / "BoxPR_curve.png", True),
    ("f1_curve_plot", RUN_DIR / "BoxF1_curve.png", True),
    ("precision_curve_plot", RUN_DIR / "BoxP_curve.png", True),
    ("recall_curve_plot", RUN_DIR / "BoxR_curve.png", True),
    ("validation_prediction_visualization", RUN_DIR / "val_batch0_pred.jpg", False),
    ("validation_prediction_visualization", RUN_DIR / "val_batch1_pred.jpg", False),
    ("validation_prediction_visualization", RUN_DIR / "val_batch2_pred.jpg", False),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a governed Detection/YOLO validation evaluation summary JSON."
    )
    return parser


def main() -> int:
    build_parser().parse_args()

    training_result = _load_json_file(TRAINING_RESULT_PATH, "training result summary")
    inventory = _load_json_file(INVENTORY_PATH, "artifact inventory")
    metadata_summary = _load_json_file(METADATA_SUMMARY_PATH, "metadata summary")
    posthoc_log = _load_json_file(POSTHOC_LOG_PATH, "posthoc run log")
    run_config = _load_yaml_file(REPO_ROOT / "configs/runs/yolo_train_v0_1_0.yaml", "run config")
    model_config = _load_yaml_file(REPO_ROOT / "configs/models/yolo.yaml", "model config")
    export_manifest = _load_yaml_file(EXPORT_MANIFEST_PATH, "export manifest")
    dataset_yaml = _load_yaml_file(DATASET_YAML_PATH, "dataset yaml")

    results_csv_rows = _load_csv_rows(RESULTS_CSV_PATH)
    if not results_csv_rows:
        raise ValueError(f"results.csv does not contain any data rows: {_repo_relative(RESULTS_CSV_PATH)}")
    final_row = results_csv_rows[-1]

    _validate_inputs(
        training_result=training_result,
        inventory=inventory,
        metadata_summary=metadata_summary,
        posthoc_log=posthoc_log,
        run_config=run_config,
        model_config=model_config,
        export_manifest=export_manifest,
        dataset_yaml=dataset_yaml,
        final_row=final_row,
    )

    evidence_files = _build_evidence_files()
    evaluation = {
        "evaluation_type": "detection_yolo_validation_evaluation",
        "run_id": RUN_ID,
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "evaluation_split": "validation",
        "evaluation_status": "completed",
        "model_identity": {
            "model_name": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "model_source": model_config["training_model_source"],
            "backend": model_config["backend"],
        },
        "dataset_identity": {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "validation_sample_count": export_manifest["split_counts"]["validation"],
            "validation_bbox_count": export_manifest["bbox_counts_by_split"]["validation"],
            "class_count": dataset_yaml["nc"],
        },
        "metrics": {
            "precision": _coerce_float(final_row["metrics/precision(B)"], "metrics/precision(B)"),
            "recall": _coerce_float(final_row["metrics/recall(B)"], "metrics/recall(B)"),
            "mAP50": _coerce_float(final_row["metrics/mAP50(B)"], "metrics/mAP50(B)"),
            "mAP50_95": _coerce_float(final_row["metrics/mAP50-95(B)"], "metrics/mAP50-95(B)"),
            "val_box_loss": _coerce_float(final_row["val/box_loss"], "val/box_loss"),
            "val_cls_loss": _coerce_float(final_row["val/cls_loss"], "val/cls_loss"),
            "val_dfl_loss": _coerce_float(final_row["val/dfl_loss"], "val/dfl_loss"),
        },
        "metric_interpretation": {
            "performance_level": "low_initial_baseline",
            "production_readiness": "not_ready",
            "summary": (
                "This 1-epoch YOLO run proves the governed training and artifact pipeline, "
                "but the validation metrics are too weak for production deployment."
            ),
            "recommendation_note": (
                "Use this run as first governed Detection evidence only. Do not mark "
                "Detection PASS or recommend it for deployment yet."
            ),
        },
        "evidence_files": evidence_files,
        "governance_references": {
            "training_result_path": _repo_relative(TRAINING_RESULT_PATH),
            "artifact_inventory_path": _repo_relative(INVENTORY_PATH),
            "metadata_summary_path": _repo_relative(METADATA_SUMMARY_PATH),
            "posthoc_log_path": _repo_relative(POSTHOC_LOG_PATH),
        },
        "governance_status": {
            "training_completed": True,
            "artifact_inventory_created": True,
            "training_result_created": True,
            "metadata_summary_created": True,
            "posthoc_log_created": True,
            "evaluation_summary_created": True,
            "registry_updated": False,
            "detection_reaudit_completed": False,
        },
        "known_limitations": [
            "This evaluation summarizes a 1-epoch YOLO baseline.",
            "Metrics are not production-quality.",
            "This file is evaluation evidence only and does not mark Detection PASS.",
            "Registry updates and Detection re-audit are handled later.",
        ],
        "created_at": _utc_now_iso(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, sort_keys=False)

    print(f"output_path={OUTPUT_PATH}")
    print(f"evaluation_status={evaluation['evaluation_status']}")
    print(f"precision={evaluation['metrics']['precision']}")
    print(f"recall={evaluation['metrics']['recall']}")
    print(f"mAP50={evaluation['metrics']['mAP50']}")
    print(f"mAP50_95={evaluation['metrics']['mAP50_95']}")
    print(f"production_readiness={evaluation['metric_interpretation']['production_readiness']}")
    return 0


def _validate_inputs(
    training_result: dict[str, Any],
    inventory: dict[str, Any],
    metadata_summary: dict[str, Any],
    posthoc_log: dict[str, Any],
    run_config: dict[str, Any],
    model_config: dict[str, Any],
    export_manifest: dict[str, Any],
    dataset_yaml: dict[str, Any],
    final_row: dict[str, str],
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

    governance = _require_dict(metadata_summary.get("governance_status"), "metadata_summary.governance_status")
    if governance.get("metadata_summary_created") is not True:
        raise ValueError("metadata summary governance flag metadata_summary_created must be true.")
    if governance.get("registry_updated") is not False:
        raise ValueError("metadata summary governance flag registry_updated must be false.")
    if governance.get("evaluation_summary_created") is not False:
        raise ValueError("metadata summary governance flag evaluation_summary_created must be false.")

    if posthoc_log.get("log_type") != "track_detection_yolo_posthoc_run_log":
        raise ValueError("posthoc log type mismatch.")
    if posthoc_log.get("run_id") != RUN_ID:
        raise ValueError("posthoc log run_id mismatch.")
    if posthoc_log.get("track_id") != TRACK_ID:
        raise ValueError("posthoc log track_id mismatch.")
    if posthoc_log.get("task_type") != TASK_TYPE:
        raise ValueError("posthoc log task_type mismatch.")
    if posthoc_log.get("run_status") != "success":
        raise ValueError("posthoc log run_status must be success.")
    posthoc_governance = _require_dict(posthoc_log.get("governance_status"), "posthoc_log.governance_status")
    if posthoc_governance.get("posthoc_log_created") is not True:
        raise ValueError("posthoc log governance flag posthoc_log_created must be true.")
    if posthoc_governance.get("registry_updated") is not False:
        raise ValueError("posthoc log governance flag registry_updated must be false.")
    if posthoc_governance.get("detection_reaudit_completed") is not False:
        raise ValueError("posthoc log governance flag detection_reaudit_completed must be false.")

    if run_config.get("identity", {}).get("task_type") != TASK_TYPE:
        raise ValueError("run config task_type mismatch.")
    if run_config.get("identity", {}).get("track_id") != TRACK_ID:
        raise ValueError("run config track_id mismatch.")
    if model_config.get("backend") != "ultralytics":
        raise ValueError("model config backend must be ultralytics.")
    if model_config.get("training_model_source") != "yolov8n.pt":
        raise ValueError("model config training_model_source must be yolov8n.pt.")
    if export_manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("export manifest dataset_id mismatch.")
    if export_manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError("export manifest dataset_version mismatch.")
    if dataset_yaml.get("nc") != 10:
        raise ValueError("dataset yaml nc must be 10.")

    if training_result.get("dataset", {}).get("split_counts", {}).get("validation") != export_manifest["split_counts"]["validation"]:
        raise ValueError("training result validation sample count mismatch.")
    if export_manifest["split_counts"]["validation"] != 345:
        raise ValueError("validation sample count must be 345.")
    if export_manifest["bbox_counts_by_split"]["validation"] != 607:
        raise ValueError("validation bbox count must be 607.")

    metrics = _require_dict(final_row, "results.csv final row")
    required_columns = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    for column in required_columns:
        if column not in metrics:
            raise ValueError(f"results.csv is missing required column: {column}")

    if metadata_summary.get("metrics", {}) != training_result.get("metrics"):
        raise ValueError("metadata summary metrics must match training result metrics.")
    if posthoc_log.get("metrics_summary", {}) != {
        "precision": _coerce_float(metrics["metrics/precision(B)"], "metrics/precision(B)"),
        "recall": _coerce_float(metrics["metrics/recall(B)"], "metrics/recall(B)"),
        "mAP50": _coerce_float(metrics["metrics/mAP50(B)"], "metrics/mAP50(B)"),
        "mAP50_95": _coerce_float(metrics["metrics/mAP50-95(B)"], "metrics/mAP50-95(B)"),
    }:
        raise ValueError("posthoc log metrics_summary must match results.csv metrics.")


def _build_evidence_files() -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for artifact_role, path, frontend_ready in EVIDENCE_FILE_SPECS:
        if not path.is_file():
            if frontend_ready:
                raise FileNotFoundError(f"Required evidence file not found: {_repo_relative(path)}")
            continue
        evidence.append(
            {
                "artifact_role": artifact_role,
                "path": _repo_relative(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "frontend_ready": frontend_ready,
            }
        )
    return evidence


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


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


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
