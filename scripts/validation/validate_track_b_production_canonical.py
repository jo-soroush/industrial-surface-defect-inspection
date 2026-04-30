"""Validate Track B production-canonical summary and inventory artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


TRACK_ID = "track_b"
TASK_TYPE = "anomaly_detection"
CANONICAL_STATUS = "production-canonical"
EXPECTED_SUMMARY_TYPE = "track_b_production_canonical_summary"
EXPECTED_INVENTORY_TYPE = "track_b_artifact_inventory"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Track B production-canonical summary and inventory."
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to track_b_production_canonical_summary__<run_id>.json.",
    )
    parser.add_argument(
        "--inventory",
        required=True,
        help="Path to track_b_artifact_inventory__<run_id>.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = validate_production_canonical(
        summary_path=Path(args.summary),
        inventory_path=Path(args.inventory),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["validation_result"] == "pass" else 1


def validate_production_canonical(
    summary_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    warnings: list[str] = []
    validated_artifacts: list[str] = []

    try:
        summary = _load_json_file(summary_path, "Track B production summary")
        inventory = _load_json_file(inventory_path, "Track B inventory")

        _validate_summary_header(summary)
        _validate_inventory_header(inventory)
        run_id = _require_string(summary.get("run_id"), "summary.run_id")
        if inventory.get("run_id") != run_id:
            raise ValueError("Inventory run_id must match summary run_id.")
        if inventory.get("canonical_run_id") != run_id:
            raise ValueError("Inventory canonical_run_id must match summary run_id.")

        summary_artifacts = _require_dict(summary.get("artifacts"), "summary.artifacts")
        inventory_artifacts = _require_dict(
            inventory.get("artifacts"), "inventory.artifacts"
        )

        expected_artifacts = (
            "training_result",
            "model_checkpoint",
            "anomaly_evaluation",
            "inventory",
        )
        for artifact_name in expected_artifacts:
            summary_entry = _require_dict(
                summary_artifacts.get(artifact_name),
                f"summary.artifacts.{artifact_name}",
            )
            artifact_path = Path(
                _require_string(
                    summary_entry.get("path"),
                    f"summary.artifacts.{artifact_name}.path",
                )
            )
            _validate_file_hash_and_size(
                path=artifact_path,
                expected_sha=_require_string(
                    summary_entry.get("sha256"),
                    f"summary.artifacts.{artifact_name}.sha256",
                ),
                expected_size=summary_entry.get("size_bytes"),
                artifact_name=artifact_name,
            )
            validated_artifacts.append(str(artifact_path))

            if artifact_name != "inventory":
                inventory_entry = _require_dict(
                    inventory_artifacts.get(artifact_name),
                    f"inventory.artifacts.{artifact_name}",
                )
                _validate_inventory_entry(
                    inventory_entry=inventory_entry,
                    summary_entry=summary_entry,
                    artifact_name=artifact_name,
                )

        _validate_linkage(summary=summary, inventory=inventory)
        _validate_training_result(
            Path(summary_artifacts["training_result"]["path"]),
            run_id=run_id,
        )
        _validate_evaluation(
            Path(summary_artifacts["anomaly_evaluation"]["path"]),
            run_id=run_id,
        )

        return {
            "validation_result": "pass",
            "run_id": run_id,
            "canonical_status": CANONICAL_STATUS,
            "validated_at": _utc_now_iso(),
            "validated_artifacts": validated_artifacts,
            "warnings": warnings,
        }
    except Exception as exc:
        return {
            "validation_result": "fail",
            "run_id": _safe_run_id(summary_path),
            "canonical_status": None,
            "validated_at": _utc_now_iso(),
            "validated_artifacts": validated_artifacts,
            "warnings": warnings,
            "error": str(exc),
        }


def _validate_summary_header(summary: dict[str, Any]) -> None:
    if summary.get("artifact_type") != EXPECTED_SUMMARY_TYPE:
        raise ValueError(
            f"Summary artifact_type must be {EXPECTED_SUMMARY_TYPE}."
        )
    if summary.get("task_type") != TASK_TYPE:
        raise ValueError("Summary task_type must be anomaly_detection.")
    if summary.get("track_id") != TRACK_ID:
        raise ValueError("Summary track_id must be track_b.")
    if summary.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("Summary canonical_status must be production-canonical.")
    validation = _require_dict(summary.get("validation"), "summary.validation")
    if validation.get("status") != "pass":
        raise ValueError("Summary embedded validation.status must be pass.")


def _validate_inventory_header(inventory: dict[str, Any]) -> None:
    if inventory.get("artifact_type") != EXPECTED_INVENTORY_TYPE:
        raise ValueError(
            f"Inventory artifact_type must be {EXPECTED_INVENTORY_TYPE}."
        )
    if inventory.get("task_type") != TASK_TYPE:
        raise ValueError("Inventory task_type must be anomaly_detection.")
    if inventory.get("track_id") != TRACK_ID:
        raise ValueError("Inventory track_id must be track_b.")
    if inventory.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("Inventory canonical_status must be production-canonical.")


def _validate_file_hash_and_size(
    path: Path,
    expected_sha: str,
    expected_size: Any,
    artifact_name: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} file not found: {path}")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{artifact_name} sha256 mismatch: expected {expected_sha}, got {actual_sha}."
        )
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise ValueError(f"{artifact_name} size_bytes must be an integer.")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{artifact_name} size mismatch: expected {expected_size}, got {actual_size}."
        )


def _validate_inventory_entry(
    inventory_entry: dict[str, Any],
    summary_entry: dict[str, Any],
    artifact_name: str,
) -> None:
    if inventory_entry.get("exists") is not True:
        raise ValueError(f"Inventory {artifact_name}.exists must be true.")
    for field in ("path", "sha256", "size_bytes"):
        if inventory_entry.get(field) != summary_entry.get(field):
            raise ValueError(
                f"Inventory {artifact_name}.{field} must match summary."
            )


def _validate_linkage(summary: dict[str, Any], inventory: dict[str, Any]) -> None:
    summary_artifacts = _require_dict(summary.get("artifacts"), "summary.artifacts")
    linkage = _require_dict(inventory.get("linkage"), "inventory.linkage")

    expected_training_result = summary_artifacts["training_result"]["path"]
    expected_checkpoint = summary_artifacts["model_checkpoint"]["path"]
    expected_evaluation = summary_artifacts["anomaly_evaluation"]["path"]

    if linkage.get("training_result") != expected_training_result:
        raise ValueError("Inventory linkage.training_result mismatch.")
    if linkage.get("model_checkpoint") != expected_checkpoint:
        raise ValueError("Inventory linkage.model_checkpoint mismatch.")
    if linkage.get("anomaly_evaluation") != expected_evaluation:
        raise ValueError("Inventory linkage.anomaly_evaluation mismatch.")
    if linkage.get("evaluation_source_training_result") != expected_training_result:
        raise ValueError("Inventory evaluation source TrainingResult mismatch.")
    if linkage.get("evaluation_source_model_checkpoint") != expected_checkpoint:
        raise ValueError("Inventory evaluation source checkpoint mismatch.")


def _validate_training_result(path: Path, run_id: str) -> None:
    payload = _load_json_file(path, "TrainingResult")
    identity = _require_dict(payload.get("identity"), "training_result.identity")
    metadata = _require_dict(payload.get("metadata"), "training_result.metadata")
    artifacts = _require_dict(payload.get("artifacts"), "training_result.artifacts")

    if identity.get("run_id") != run_id:
        raise ValueError("TrainingResult run_id mismatch.")
    if identity.get("task_type") != TASK_TYPE:
        raise ValueError("TrainingResult task_type must be anomaly_detection.")
    if identity.get("is_experiment") is not False:
        raise ValueError("TrainingResult is_experiment must be false.")
    if metadata.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("TrainingResult canonical_status must be production-canonical.")
    model_artifact = _require_dict(
        artifacts.get("model_artifact"), "training_result.artifacts.model_artifact"
    )
    if model_artifact.get("type") != "pytorch_state_dict":
        raise ValueError("TrainingResult model_artifact.type must be pytorch_state_dict.")
    checkpoint_path = Path(
        _require_string(model_artifact.get("path"), "model_artifact.path")
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TrainingResult checkpoint missing: {checkpoint_path}")


def _validate_evaluation(path: Path, run_id: str) -> None:
    payload = _load_json_file(path, "anomaly evaluation")
    if payload.get("artifact_type") != "anomaly_detection_evaluation":
        raise ValueError("Evaluation artifact_type must be anomaly_detection_evaluation.")
    if payload.get("run_id") != run_id:
        raise ValueError("Evaluation run_id mismatch.")
    if payload.get("task_type") != TASK_TYPE:
        raise ValueError("Evaluation task_type must be anomaly_detection.")
    counts = _require_dict(payload.get("counts"), "evaluation.counts")
    test_count = counts.get("test_score_count")
    samples = payload.get("samples")
    if isinstance(test_count, bool) or not isinstance(test_count, int) or test_count <= 0:
        raise ValueError("Evaluation counts.test_score_count must be a positive integer.")
    if not isinstance(samples, list) or len(samples) != test_count:
        raise ValueError("Evaluation samples count must match test_score_count.")


def _load_json_file(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact_name} JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON must contain an object.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _safe_run_id(summary_path: Path) -> str | None:
    try:
        summary = _load_json_file(summary_path, "Track B production summary")
    except Exception:
        return None
    run_id = summary.get("run_id")
    return run_id if isinstance(run_id, str) else None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
