"""Build the final governed Detection/YOLO re-audit report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


DEFAULT_RUN_ID = "yolo_train_v0_1_0"
TRACK_ID = "detection"
TASK_TYPE = "object_detection"

VALIDATOR_SCRIPT = Path("scripts/validation/validate_detection_artifacts.py")
RUN_REGISTRY_PATH = Path("artifacts/models/registry/run_registry.yaml")
ARTIFACT_REGISTRY_PATH = Path("artifacts/models/registry/artifact_registry.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the final governed Detection/YOLO re-audit report."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--training-result", default=None)
    parser.add_argument("--artifact-inventory", default=None)
    parser.add_argument("--metadata-summary", default=None)
    parser.add_argument("--posthoc-log", default=None)
    parser.add_argument("--evaluation-summary", default=None)
    parser.add_argument("--run-registry", default=str(RUN_REGISTRY_PATH))
    parser.add_argument("--artifact-registry", default=str(ARTIFACT_REGISTRY_PATH))
    parser.add_argument("--output-path", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id
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
    run_registry_path = Path(args.run_registry)
    artifact_registry_path = Path(args.artifact_registry)
    output_path = Path(
        args.output_path
        or f"artifacts/reports/audits/detection_yolo_reaudit__{run_id}.json"
    )

    validator_status = _run_validator(run_id)

    evaluation_summary = _load_json_file(evaluation_summary_path, "evaluation summary")
    training_result = _load_json_file(training_result_path, "training result")
    artifact_inventory = _load_json_file(artifact_inventory_path, "artifact inventory")
    metadata_summary = _load_json_file(metadata_summary_path, "metadata summary")
    posthoc_log = _load_json_file(posthoc_log_path, "posthoc log")
    run_registry = _load_yaml_file(run_registry_path, "run registry")
    artifact_registry = _load_yaml_file(artifact_registry_path, "artifact registry")

    _validate_loaded_evidence(
        run_id,
        validator_status,
        evaluation_summary,
        training_result,
        artifact_inventory,
        metadata_summary,
        posthoc_log,
        run_registry,
        artifact_registry,
    )

    metrics = _require_dict(evaluation_summary.get("metrics"), "evaluation_summary.metrics")
    interpretation = _require_dict(
        evaluation_summary.get("metric_interpretation"),
        "evaluation_summary.metric_interpretation",
    )

    report = {
        "audit_type": "detection_yolo_final_reaudit",
        "run_id": run_id,
        "track_id": TRACK_ID,
        "task_type": TASK_TYPE,
        "audit_status": "governance_pass_model_not_ready",
        "governance_status": {
            "required_files_present": True,
            "run_registry_entry_valid": True,
            "artifact_registry_entries_valid": True,
            "training_result_valid": True,
            "artifact_inventory_valid": True,
            "metadata_summary_valid": True,
            "posthoc_log_valid": True,
            "evaluation_summary_valid": True,
            "hashes_and_sizes_valid": True,
            "cross_file_consistency_valid": True,
            "validation_script_status": validator_status["validation_status"],
        },
        "model_performance_status": {
            "production_readiness": interpretation["production_readiness"],
            "performance_level": interpretation["performance_level"],
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "reason": (
                "The governed YOLO pipeline is valid, but the 1-epoch validation "
                "metrics are too weak for production deployment."
            ),
        },
        "decision": {
            "governance_pipeline_pass": True,
            "model_production_ready": False,
            "detection_track_final_pass": False,
            "recommended_next_step": (
                "Use this as governed first-run evidence only. Improve training "
                "duration, hyperparameters, data strategy, and evaluation before "
                "marking Detection as model-ready."
            ),
        },
        "evidence_references": {
            "validator_script": str(VALIDATOR_SCRIPT),
            "training_result_path": str(training_result_path),
            "artifact_inventory_path": str(artifact_inventory_path),
            "metadata_summary_path": str(metadata_summary_path),
            "posthoc_log_path": str(posthoc_log_path),
            "evaluation_summary_path": str(evaluation_summary_path),
            "run_registry_path": str(run_registry_path),
            "artifact_registry_path": str(artifact_registry_path),
        },
        "known_limitations": [
            "The audit confirms governance and evidence consistency, not production model quality.",
            "The YOLO model was trained for only 1 epoch.",
            "The validation metrics indicate a low initial baseline.",
            "Further training and evaluation are required before Detection can be considered model-ready.",
        ],
        "created_at": _utc_timestamp(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"output_path={output_path}")
    print(f"audit_status={report['audit_status']}")
    print(
        "validation_script_status="
        f"{report['governance_status']['validation_script_status']}"
    )
    print(
        "governance_pipeline_pass="
        f"{str(report['decision']['governance_pipeline_pass']).lower()}"
    )
    print(
        "model_production_ready="
        f"{str(report['decision']['model_production_ready']).lower()}"
    )
    print(
        "detection_track_final_pass="
        f"{str(report['decision']['detection_track_final_pass']).lower()}"
    )
    print(f"mAP50={report['model_performance_status']['mAP50']}")
    print(f"mAP50_95={report['model_performance_status']['mAP50_95']}")
    print(
        "production_readiness="
        f"{report['model_performance_status']['production_readiness']}"
    )
    return 0


def _run_validator(run_id: str) -> dict[str, str]:
    _require_file(VALIDATOR_SCRIPT, "validator_script")
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR_SCRIPT), "--run-id", run_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Detection artifact validation failed before re-audit report build. "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )

    status = _parse_key_value_output(completed.stdout)
    expected_validator_values = {
        "validation_status": "pass",
        "run_registry_status": "pass",
        "artifact_registry_status": "pass",
        "training_result_status": "pass",
        "artifact_inventory_status": "pass",
        "metadata_summary_status": "pass",
        "posthoc_log_status": "pass",
        "evaluation_summary_status": "pass",
        "cross_file_consistency_status": "pass",
        "run_id": run_id,
    }
    for key, expected_value in expected_validator_values.items():
        actual_value = status.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"validator output {key} must be {expected_value!r}; found {actual_value!r}."
            )
    return status


def _validate_loaded_evidence(
    run_id: str,
    validator_status: dict[str, str],
    evaluation_summary: dict[str, Any],
    training_result: dict[str, Any],
    artifact_inventory: dict[str, Any],
    metadata_summary: dict[str, Any],
    posthoc_log: dict[str, Any],
    run_registry: dict[str, Any],
    artifact_registry: dict[str, Any],
) -> None:
    _require_equal(validator_status.get("validation_status"), "pass", "validation_status")

    for name, payload in (
        ("evaluation_summary", evaluation_summary),
        ("training_result", training_result),
        ("artifact_inventory", artifact_inventory),
        ("metadata_summary", metadata_summary),
        ("posthoc_log", posthoc_log),
    ):
        _require_equal(payload.get("run_id"), run_id, f"{name}.run_id")

    _require_equal(
        evaluation_summary.get("evaluation_status"),
        "completed",
        "evaluation_summary.evaluation_status",
    )
    metrics = _require_dict(evaluation_summary.get("metrics"), "evaluation_summary.metrics")
    _require_number(metrics.get("mAP50"), "evaluation_summary.metrics.mAP50")
    _require_number(metrics.get("mAP50_95"), "evaluation_summary.metrics.mAP50_95")

    interpretation = _require_dict(
        evaluation_summary.get("metric_interpretation"),
        "evaluation_summary.metric_interpretation",
    )
    _require_equal(
        interpretation.get("production_readiness"),
        "not_ready",
        "evaluation_summary.metric_interpretation.production_readiness",
    )
    _require_equal(
        interpretation.get("performance_level"),
        "low_initial_baseline",
        "evaluation_summary.metric_interpretation.performance_level",
    )

    runs = _require_list(run_registry.get("runs"), "run_registry.runs")
    run_matches = [run for run in runs if isinstance(run, dict) and run.get("run_id") == run_id]
    if len(run_matches) != 1:
        raise ValueError(f"run_registry must contain exactly one {run_id}.")

    artifacts = _require_list(artifact_registry.get("artifacts"), "artifact_registry.artifacts")
    detection_entries = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("run_id") == run_id
    ]
    if len(detection_entries) < 9:
        raise ValueError("artifact_registry must contain the governed Detection entries.")


def _parse_key_value_output(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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


def _require_equal(actual: Any, expected: Any, field_name: str) -> None:
    if actual != expected:
        raise ValueError(f"{field_name} must be {expected!r}; found {actual!r}.")


def _require_number(value: Any, field_name: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric; found {value!r}.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
