"""Validate governed Detection/YOLO artifacts end-to-end."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


DEFAULT_RUN_ID = "yolo_train_v0_1_0"
TRACK_ID = "detection"
TASK_TYPE = "object_detection"
MODEL_NAME = "yolo"
MODEL_VERSION = "0.1.0"
DATASET_ID = "gc10det_detection"
DATASET_VERSION = "gc10det_1.0"


@dataclass(frozen=True)
class ValidationContext:
    run_id: str
    config_id: str
    model_version: str
    run_registry_path: Path
    artifact_registry_path: Path
    training_result_path: Path
    artifact_inventory_path: Path
    metadata_summary_path: Path
    posthoc_log_path: Path
    evaluation_summary_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    results_csv_path: Path
    args_yaml_path: Path
    expected_artifacts: dict[str, dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate governed Detection/YOLO artifacts end-to-end."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument(
        "--run-registry",
        default="artifacts/models/registry/run_registry.yaml",
    )
    parser.add_argument(
        "--artifact-registry",
        default="artifacts/models/registry/artifact_registry.yaml",
    )
    parser.add_argument(
        "--detection-runs-root",
        default="artifacts/detection/yolo/runs",
    )
    parser.add_argument(
        "--training-result",
        default=None,
        help="Override path to the Detection/YOLO training result summary.",
    )
    parser.add_argument(
        "--artifact-inventory",
        default=None,
        help="Override path to the Detection/YOLO artifact inventory.",
    )
    parser.add_argument(
        "--metadata-summary",
        default=None,
        help="Override path to the Detection/YOLO metadata summary.",
    )
    parser.add_argument(
        "--posthoc-log",
        default=None,
        help="Override path to the Detection/YOLO post-hoc log.",
    )
    parser.add_argument(
        "--evaluation-summary",
        default=None,
        help="Override path to the Detection/YOLO validation evaluation summary.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ctx = _build_context(args)

    _validate_required_files(ctx)

    run_registry = _validate_run_registry(ctx)
    artifact_registry_entries = _validate_artifact_registry(ctx)
    training_result = _validate_training_result(ctx)
    artifact_inventory = _validate_artifact_inventory(ctx)
    metadata_summary = _validate_metadata_summary(ctx)
    posthoc_log = _validate_posthoc_log(ctx)
    evaluation_summary = _validate_evaluation_summary(ctx)
    _validate_cross_file_consistency(
        ctx,
        run_registry,
        artifact_registry_entries,
        training_result,
        artifact_inventory,
        metadata_summary,
        posthoc_log,
        evaluation_summary,
    )

    metrics = _require_dict(evaluation_summary.get("metrics"), "evaluation_summary.metrics")
    interpretation = _require_dict(
        evaluation_summary.get("metric_interpretation"),
        "evaluation_summary.metric_interpretation",
    )

    print("validation_status=pass")
    print("run_registry_status=pass")
    print("artifact_registry_status=pass")
    print("training_result_status=pass")
    print("artifact_inventory_status=pass")
    print("metadata_summary_status=pass")
    print("posthoc_log_status=pass")
    print("evaluation_summary_status=pass")
    print("cross_file_consistency_status=pass")
    print(f"run_id={ctx.run_id}")
    print(f"mAP50={metrics.get('mAP50')}")
    print(f"mAP50_95={metrics.get('mAP50_95')}")
    print(f"production_readiness={interpretation.get('production_readiness')}")
    return 0


def _build_context(args: argparse.Namespace) -> ValidationContext:
    run_id = _require_non_empty_string(args.run_id, "run_id")
    config_id = args.config_id or run_id
    run_dir = Path(args.detection_runs_root) / run_id
    training_result_path = Path(
        args.training_result
        or f"artifacts/models/analysis/training_result__{run_id}.json"
    )
    artifact_inventory_path = Path(
        args.artifact_inventory
        or f"artifacts/models/inventory/track_detection_artifact_inventory__{run_id}.json"
    )
    metadata_summary_path = Path(
        args.metadata_summary
        or f"artifacts/models/metadata/track_detection_yolo_metadata_summary__{run_id}.json"
    )
    posthoc_log_path = Path(
        args.posthoc_log
        or f"artifacts/models/logs/track_detection_yolo_posthoc_run_log__{run_id}.json"
    )
    evaluation_summary_path = Path(
        args.evaluation_summary
        or f"artifacts/models/metrics/detection_evaluation__{run_id}__validation.json"
    )
    best_checkpoint_path = run_dir / "weights" / "best.pt"
    last_checkpoint_path = run_dir / "weights" / "last.pt"
    results_csv_path = run_dir / "results.csv"
    args_yaml_path = run_dir / "args.yaml"

    expected_artifacts = {
        f"track_detection__{run_id}__training_result": {
            "path": training_result_path,
            "type": "detection_yolo_training_result",
            "format": "json",
        },
        f"track_detection__{run_id}__validation_evaluation": {
            "path": evaluation_summary_path,
            "type": "detection_yolo_validation_evaluation",
            "format": "json",
        },
        f"track_detection__{run_id}__artifact_inventory": {
            "path": artifact_inventory_path,
            "type": "track_detection_yolo_artifact_inventory",
            "format": "json",
        },
        f"track_detection__{run_id}__metadata_summary": {
            "path": metadata_summary_path,
            "type": "track_detection_yolo_metadata_summary",
            "format": "json",
        },
        f"track_detection__{run_id}__posthoc_log": {
            "path": posthoc_log_path,
            "type": "track_detection_yolo_posthoc_run_log",
            "format": "json",
        },
        f"track_detection__{run_id}__best_checkpoint": {
            "path": best_checkpoint_path,
            "type": "yolo_best_checkpoint",
            "format": "pt",
        },
        f"track_detection__{run_id}__last_checkpoint": {
            "path": last_checkpoint_path,
            "type": "yolo_last_checkpoint",
            "format": "pt",
        },
        f"track_detection__{run_id}__training_metrics_csv": {
            "path": results_csv_path,
            "type": "yolo_training_metrics_csv",
            "format": "csv",
        },
        f"track_detection__{run_id}__training_args": {
            "path": args_yaml_path,
            "type": "yolo_training_args",
            "format": "yaml",
        },
    }

    return ValidationContext(
        run_id=run_id,
        config_id=config_id,
        model_version=args.model_version,
        run_registry_path=Path(args.run_registry),
        artifact_registry_path=Path(args.artifact_registry),
        training_result_path=training_result_path,
        artifact_inventory_path=artifact_inventory_path,
        metadata_summary_path=metadata_summary_path,
        posthoc_log_path=posthoc_log_path,
        evaluation_summary_path=evaluation_summary_path,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        results_csv_path=results_csv_path,
        args_yaml_path=args_yaml_path,
        expected_artifacts=expected_artifacts,
    )


def _validate_required_files(ctx: ValidationContext) -> None:
    required_files = {
        "run_registry": ctx.run_registry_path,
        "artifact_registry": ctx.artifact_registry_path,
        "training_result": ctx.training_result_path,
        "artifact_inventory": ctx.artifact_inventory_path,
        "metadata_summary": ctx.metadata_summary_path,
        "posthoc_log": ctx.posthoc_log_path,
        "evaluation_summary": ctx.evaluation_summary_path,
        "best_checkpoint": ctx.best_checkpoint_path,
        "last_checkpoint": ctx.last_checkpoint_path,
        "results_csv": ctx.results_csv_path,
        "args_yaml": ctx.args_yaml_path,
    }
    for name, path in required_files.items():
        _require_file(path, name)


def _validate_run_registry(ctx: ValidationContext) -> dict[str, Any]:
    registry = _load_yaml_file(ctx.run_registry_path, "run registry")
    runs = _require_list(registry.get("runs"), "run_registry.runs")
    matches = [
        run for run in runs if isinstance(run, dict) and run.get("run_id") == ctx.run_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"run_registry must contain exactly one run_id {ctx.run_id}; found {len(matches)}."
        )
    run = matches[0]
    _require_fields(
        run,
        "run_registry.run",
        {
            "model_name": MODEL_NAME,
            "model_version": ctx.model_version,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "task_type": TASK_TYPE,
            "config_id": ctx.config_id,
            "run_status": "success",
            "artifact_created": True,
        },
    )
    expected_references = {
        "metadata_path": ctx.training_result_path,
        "metrics_path": ctx.evaluation_summary_path,
        "log_path": ctx.posthoc_log_path,
        "checkpoint_path": ctx.best_checkpoint_path,
        "inventory_path": ctx.artifact_inventory_path,
        "metadata_summary_path": ctx.metadata_summary_path,
    }
    for key, expected_path in expected_references.items():
        actual = Path(_require_string(run.get(key), f"run_registry.run.{key}"))
        if actual != expected_path:
            raise ValueError(
                f"run_registry.run.{key} must be {expected_path}; found {actual}."
            )
        _require_file(actual, f"run_registry.run.{key}")
    return run


def _validate_artifact_registry(ctx: ValidationContext) -> dict[str, dict[str, Any]]:
    registry = _load_yaml_file(ctx.artifact_registry_path, "artifact registry")
    artifacts = _require_list(registry.get("artifacts"), "artifact_registry.artifacts")
    validated_entries: dict[str, dict[str, Any]] = {}

    for artifact_id, expected in ctx.expected_artifacts.items():
        matches = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict) and artifact.get("artifact_id") == artifact_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"artifact_registry must contain exactly one {artifact_id}; found {len(matches)}."
            )
        entry = matches[0]
        expected_path = _require_path(expected["path"], f"{artifact_id}.expected.path")
        _require_fields(
            entry,
            f"artifact_registry.{artifact_id}",
            {
                "run_id": ctx.run_id,
                "model_name": MODEL_NAME,
                "model_version": ctx.model_version,
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "config_id": ctx.config_id,
                "status": "active",
                "artifact_path": str(expected_path),
                "artifact_type": expected["type"],
                "artifact_format": expected["format"],
                "artifact_uri": None,
                "storage_backend": "local",
            },
        )
        _validate_file_integrity(
            expected_path,
            entry.get("artifact_size_bytes"),
            entry.get("artifact_hash"),
            f"artifact_registry.{artifact_id}",
        )
        validated_entries[artifact_id] = entry
    return validated_entries


def _validate_training_result(ctx: ValidationContext) -> dict[str, Any]:
    payload = _load_json_file(ctx.training_result_path, "training result")
    _require_fields(
        payload,
        "training_result",
        {
            "result_type": "detection_yolo_training_result",
            "run_id": ctx.run_id,
            "track_id": TRACK_ID,
            "task_type": TASK_TYPE,
            "training_status": "success",
            "execution_environment": "colab",
        },
    )
    metrics = _require_dict(payload.get("metrics"), "training_result.metrics")
    _require_required_metrics(metrics, "training_result.metrics")
    artifacts = _require_dict(payload.get("artifacts"), "training_result.artifacts")
    _require_equal(
        artifacts.get("inventory_status"),
        "pass",
        "training_result.artifacts.inventory_status",
    )
    governance = _require_dict(payload.get("governance"), "training_result.governance")
    _require_equal(
        governance.get("registry_updated"),
        False,
        "training_result.governance.registry_updated",
    )
    _require_equal(
        governance.get("evaluation_summary_created"),
        False,
        "training_result.governance.evaluation_summary_created",
    )
    return payload


def _validate_artifact_inventory(ctx: ValidationContext) -> dict[str, Any]:
    payload = _load_json_file(ctx.artifact_inventory_path, "artifact inventory")
    _require_fields(
        payload,
        "artifact_inventory",
        {
            "inventory_type": "track_detection_yolo_artifact_inventory",
            "run_id": ctx.run_id,
            "track_id": TRACK_ID,
            "task_type": TASK_TYPE,
            "inventory_status": "pass",
            "required_files_status": "pass",
        },
    )
    artifact_count = payload.get("artifact_count")
    if isinstance(artifact_count, bool) or not isinstance(artifact_count, int):
        raise ValueError("artifact_inventory.artifact_count must be a positive integer.")
    if artifact_count <= 0:
        raise ValueError("artifact_inventory.artifact_count must be a positive integer.")
    artifacts = _require_list(payload.get("artifacts"), "artifact_inventory.artifacts")
    if len(artifacts) != artifact_count:
        raise ValueError("artifact_inventory.artifacts length must match artifact_count.")
    for index, artifact in enumerate(artifacts):
        entry = _require_dict(artifact, f"artifact_inventory.artifacts[{index}]")
        if entry.get("exists") is not True:
            raise ValueError(
                f"artifact_inventory.artifacts[{index}].exists must be true."
            )
        path = Path(
            _require_string(entry.get("path"), f"artifact_inventory.artifacts[{index}].path")
        )
        _validate_file_integrity(
            path,
            entry.get("size_bytes"),
            entry.get("sha256"),
            f"artifact_inventory.artifacts[{index}]",
        )
    return payload


def _validate_metadata_summary(ctx: ValidationContext) -> dict[str, Any]:
    payload = _load_json_file(ctx.metadata_summary_path, "metadata summary")
    _require_fields(
        payload,
        "metadata_summary",
        {
            "metadata_type": "track_detection_yolo_metadata_summary",
            "run_id": ctx.run_id,
            "track_id": TRACK_ID,
            "task_type": TASK_TYPE,
            "run_status": "success",
            "execution_environment": "colab",
        },
    )
    governance = _require_dict(
        payload.get("governance_status"), "metadata_summary.governance_status"
    )
    _require_fields(
        governance,
        "metadata_summary.governance_status",
        {
            "metadata_summary_created": True,
            "registry_updated": False,
            "evaluation_summary_created": False,
        },
    )
    return payload


def _validate_posthoc_log(ctx: ValidationContext) -> dict[str, Any]:
    payload = _load_json_file(ctx.posthoc_log_path, "posthoc log")
    _require_fields(
        payload,
        "posthoc_log",
        {
            "log_type": "track_detection_yolo_posthoc_run_log",
            "run_id": ctx.run_id,
            "track_id": TRACK_ID,
            "task_type": TASK_TYPE,
            "run_status": "success",
            "execution_environment": "colab",
        },
    )
    timeline = _require_list(payload.get("timeline"), "posthoc_log.timeline")
    if len(timeline) != 9:
        raise ValueError(f"posthoc_log.timeline length must be 9; found {len(timeline)}.")
    governance = _require_dict(
        payload.get("governance_status"), "posthoc_log.governance_status"
    )
    _require_fields(
        governance,
        "posthoc_log.governance_status",
        {
            "posthoc_log_created": True,
            "registry_updated": False,
            "evaluation_summary_created": False,
            "detection_reaudit_completed": False,
        },
    )
    return payload


def _validate_evaluation_summary(ctx: ValidationContext) -> dict[str, Any]:
    payload = _load_json_file(ctx.evaluation_summary_path, "evaluation summary")
    _require_fields(
        payload,
        "evaluation_summary",
        {
            "evaluation_type": "detection_yolo_validation_evaluation",
            "run_id": ctx.run_id,
            "track_id": TRACK_ID,
            "task_type": TASK_TYPE,
            "evaluation_split": "validation",
            "evaluation_status": "completed",
        },
    )
    metrics = _require_dict(payload.get("metrics"), "evaluation_summary.metrics")
    _require_required_metrics(metrics, "evaluation_summary.metrics")
    interpretation = _require_dict(
        payload.get("metric_interpretation"),
        "evaluation_summary.metric_interpretation",
    )
    _require_string(
        interpretation.get("production_readiness"),
        "evaluation_summary.metric_interpretation.production_readiness",
    )
    evidence_files = _require_list(
        payload.get("evidence_files"), "evaluation_summary.evidence_files"
    )
    if len(evidence_files) != 11:
        raise ValueError(
            f"evaluation_summary.evidence_files count must be 11; found {len(evidence_files)}."
        )
    for index, evidence_file in enumerate(evidence_files):
        entry = _require_dict(evidence_file, f"evaluation_summary.evidence_files[{index}]")
        if entry.get("exists") is not True:
            raise ValueError(
                f"evaluation_summary.evidence_files[{index}].exists must be true."
            )
        path = Path(
            _require_string(
                entry.get("path"), f"evaluation_summary.evidence_files[{index}].path"
            )
        )
        _validate_file_integrity(
            path,
            entry.get("size_bytes"),
            entry.get("sha256"),
            f"evaluation_summary.evidence_files[{index}]",
        )
    governance = _require_dict(
        payload.get("governance_status"), "evaluation_summary.governance_status"
    )
    _require_fields(
        governance,
        "evaluation_summary.governance_status",
        {
            "evaluation_summary_created": True,
            "registry_updated": False,
            "detection_reaudit_completed": False,
        },
    )
    return payload


def _validate_cross_file_consistency(
    ctx: ValidationContext,
    run_registry: dict[str, Any],
    artifact_registry_entries: dict[str, dict[str, Any]],
    training_result: dict[str, Any],
    artifact_inventory: dict[str, Any],
    metadata_summary: dict[str, Any],
    posthoc_log: dict[str, Any],
    evaluation_summary: dict[str, Any],
) -> None:
    evidence_files = {
        "training_result": training_result,
        "artifact_inventory": artifact_inventory,
        "metadata_summary": metadata_summary,
        "posthoc_log": posthoc_log,
        "evaluation_summary": evaluation_summary,
    }
    for name, payload in evidence_files.items():
        _require_equal(payload.get("run_id"), ctx.run_id, f"{name}.run_id")
        if "track_id" in payload:
            _require_equal(payload.get("track_id"), TRACK_ID, f"{name}.track_id")
        if "task_type" in payload:
            _require_equal(payload.get("task_type"), TASK_TYPE, f"{name}.task_type")

    _require_equal(run_registry.get("run_id"), ctx.run_id, "run_registry.run_id")
    for artifact_id, entry in artifact_registry_entries.items():
        _require_equal(
            entry.get("run_id"), ctx.run_id, f"artifact_registry.{artifact_id}.run_id"
        )

    training_metrics = _require_dict(training_result.get("metrics"), "training_result.metrics")
    metadata_metrics = _require_dict(metadata_summary.get("metrics"), "metadata_summary.metrics")
    posthoc_metrics = _require_dict(posthoc_log.get("metrics_summary"), "posthoc_log.metrics_summary")
    evaluation_metrics = _require_dict(evaluation_summary.get("metrics"), "evaluation_summary.metrics")
    for metric_name in ("precision", "recall", "mAP50", "mAP50_95"):
        baseline = training_metrics.get(metric_name)
        for source_name, metrics in (
            ("metadata_summary", metadata_metrics),
            ("posthoc_log", posthoc_metrics),
            ("evaluation_summary", evaluation_metrics),
        ):
            _require_equal(
                metrics.get(metric_name),
                baseline,
                f"{source_name}.metrics.{metric_name}",
            )

    _validate_integrity_hash(
        metadata_summary,
        "metadata_summary.artifact_integrity.best_checkpoint_sha256",
        ("artifact_integrity", "best_checkpoint_sha256"),
        ctx.best_checkpoint_path,
    )
    _validate_integrity_hash(
        metadata_summary,
        "metadata_summary.artifact_integrity.last_checkpoint_sha256",
        ("artifact_integrity", "last_checkpoint_sha256"),
        ctx.last_checkpoint_path,
    )
    _validate_integrity_hash(
        posthoc_log,
        "posthoc_log.artifact_integrity.best_checkpoint_sha256",
        ("artifact_integrity", "best_checkpoint_sha256"),
        ctx.best_checkpoint_path,
    )
    _validate_integrity_hash(
        posthoc_log,
        "posthoc_log.artifact_integrity.last_checkpoint_sha256",
        ("artifact_integrity", "last_checkpoint_sha256"),
        ctx.last_checkpoint_path,
    )

    _require_artifact_reference(
        posthoc_log,
        ("artifact_references", "best_checkpoint_path"),
        ctx.best_checkpoint_path,
        "posthoc_log.artifact_references.best_checkpoint_path",
    )
    _require_artifact_reference(
        posthoc_log,
        ("artifact_references", "last_checkpoint_path"),
        ctx.last_checkpoint_path,
        "posthoc_log.artifact_references.last_checkpoint_path",
    )

    registry_paths = {
        Path(entry["artifact_path"]) for entry in artifact_registry_entries.values()
    }
    expected_paths = {config["path"] for config in ctx.expected_artifacts.values()}
    if registry_paths != expected_paths:
        raise ValueError("artifact_registry governed paths do not match expected paths.")


def _validate_integrity_hash(
    payload: dict[str, Any],
    field_name: str,
    keys: tuple[str, str],
    path: Path,
) -> None:
    parent = _require_dict(payload.get(keys[0]), keys[0])
    _require_equal(parent.get(keys[1]), _sha256(path), field_name)


def _require_artifact_reference(
    payload: dict[str, Any],
    keys: tuple[str, str],
    expected_path: Path,
    field_name: str,
) -> None:
    parent = _require_dict(payload.get(keys[0]), keys[0])
    actual_path = Path(_require_string(parent.get(keys[1]), field_name))
    if actual_path != expected_path:
        raise ValueError(f"{field_name} must be {expected_path}; found {actual_path}.")
    _require_file(actual_path, field_name)


def _validate_file_integrity(
    path: Path,
    expected_size: Any,
    expected_sha256: Any,
    field_name: str,
) -> None:
    _require_file(path, field_name)
    _require_equal(expected_size, path.stat().st_size, f"{field_name}.size_bytes")
    _require_equal(expected_sha256, _sha256(path), f"{field_name}.sha256")


def _require_required_metrics(metrics: dict[str, Any], field_name: str) -> None:
    for metric_name in ("precision", "recall", "mAP50", "mAP50_95"):
        if metric_name not in metrics:
            raise ValueError(f"{field_name}.{metric_name} is required.")
        if not isinstance(metrics[metric_name], (int, float)):
            raise ValueError(f"{field_name}.{metric_name} must be numeric.")


def _load_json_file(path: Path, artifact_name: str) -> dict[str, Any]:
    _require_file(path, artifact_name)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact_name} JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON must contain an object.")
    return payload


def _load_yaml_file(path: Path, artifact_name: str) -> dict[str, Any]:
    _require_file(path, artifact_name)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} YAML must contain a mapping.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(
    payload: dict[str, Any],
    scope: str,
    expected_fields: dict[str, Any],
) -> None:
    for key, expected_value in expected_fields.items():
        _require_equal(payload.get(key), expected_value, f"{scope}.{key}")


def _require_equal(actual: Any, expected: Any, field_name: str) -> None:
    if actual != expected:
        raise ValueError(f"{field_name} must be {expected!r}; found {actual!r}.")


def _require_file(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} file not found: {path}")


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_non_empty_string(value: Any, field_name: str) -> str:
    return _require_string(value, field_name).strip()


def _require_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise ValueError(f"{field_name} must be a Path.")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
