"""Build a governed Track B anomaly detection artifact inventory."""

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
DATASET_ID = "mvtec_anomaly"
MODEL_TYPE = "autoencoder"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Track B anomaly detection artifact inventory JSON."
    )
    parser.add_argument(
        "--training-result",
        required=True,
        help="Path to the required Track B TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--evaluation",
        required=True,
        help="Path to the anomaly_detection_evaluation JSON artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/inventory",
        help="Directory where the inventory JSON will be written.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    training_result_path = Path(args.training_result)
    evaluation_path = Path(args.evaluation)

    training_result = _load_json_file(training_result_path, "TrainingResult")
    evaluation = _load_json_file(evaluation_path, "anomaly evaluation")
    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")
    artifacts = _require_dict(training_result.get("artifacts"), "artifacts")

    run_id = _require_string(identity.get("run_id"), "identity.run_id")
    _validate_training_result(training_result)
    _validate_evaluation(evaluation, run_id, metadata)

    checkpoint_path = _model_checkpoint_path(artifacts)
    config_id = _require_string(
        identity.get("run_config_id") or metadata.get("training_config_id"),
        "config_id",
    )
    model_id = _require_string(metadata.get("model_name"), "metadata.model_name")
    dataset_id = _require_string(metadata.get("dataset_id"), "metadata.dataset_id")

    inventory = {
        "artifact_type": "track_b_artifact_inventory",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_run_id": run_id,
        "canonical_status": (
            "development_canonical"
            if identity.get("is_experiment") is True
            else "canonical"
        ),
        "canonical_run_policy": {
            "policy": (
                "Use the inventory canonical_run_id for Track B evaluation and "
                "frontend integration until a replacement run is explicitly "
                "promoted by regenerating this inventory."
            ),
            "promotion_requirement": (
                "Production canonical runs should be generated with "
                "identity.is_experiment=false; development canonical runs must "
                "declare canonical_status=development_canonical."
            ),
            "is_experiment": identity.get("is_experiment"),
        },
        "model_id": model_id,
        "model_type": MODEL_TYPE,
        "dataset_id": dataset_id,
        "config_id": config_id,
        "created_at": _utc_now_iso(),
        "source_training_result": str(training_result_path),
        "source_model_checkpoint": str(checkpoint_path),
        "source_evaluation": str(evaluation_path),
        "counts": {
            "train_sample_count": metadata.get("train_sample_count"),
            "validation_sample_count": metadata.get("validation_sample_count"),
            "test_sample_count": metadata.get("test_sample_count"),
            "train_score_count": _require_dict(evaluation.get("counts"), "evaluation.counts").get(
                "train_score_count"
            ),
            "test_score_count": _require_dict(evaluation.get("counts"), "evaluation.counts").get(
                "test_score_count"
            ),
            "normal_test_count": _require_dict(evaluation.get("counts"), "evaluation.counts").get(
                "normal_test_count"
            ),
            "anomaly_test_count": _require_dict(evaluation.get("counts"), "evaluation.counts").get(
                "anomaly_test_count"
            ),
        },
        "linkage": {
            "evaluation_source_training_result": evaluation.get("source_training_result"),
            "evaluation_source_model_checkpoint": evaluation.get("source_model_checkpoint"),
            "training_result_model_checkpoint": str(checkpoint_path),
        },
        "artifacts": {
            "training_result": _artifact_entry(
                path=training_result_path,
                artifact_type="TrainingResult",
                purpose="Source of truth for the Track B autoencoder run.",
                frontend_ready=True,
                required_for_frontend=True,
            ),
            "model_checkpoint": _artifact_entry(
                path=checkpoint_path,
                artifact_type="model_checkpoint",
                purpose="Autoencoder checkpoint referenced by the TrainingResult.",
                frontend_ready=False,
                required_for_frontend=True,
            ),
            "anomaly_evaluation": _artifact_entry(
                path=evaluation_path,
                artifact_type="anomaly_detection_evaluation",
                purpose="Image-level Track B anomaly scores, predictions, and metrics.",
                frontend_ready=True,
                required_for_frontend=True,
            ),
        },
    }
    _validate_inventory(inventory)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"track_b_artifact_inventory__{run_id}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2)

    print(f"track_b_artifact_inventory_path={output_path}")
    print(f"canonical_run_id={run_id}")
    print(f"canonical_status={inventory['canonical_status']}")
    return 0


def _validate_training_result(training_result: dict[str, Any]) -> None:
    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")
    artifacts = _require_dict(training_result.get("artifacts"), "artifacts")
    metrics = _require_dict(training_result.get("metrics"), "metrics")
    learning_curves = _require_dict(
        training_result.get("learning_curves"), "learning_curves"
    )

    if identity.get("task_type") != TASK_TYPE:
        raise ValueError("TrainingResult identity.task_type must be anomaly_detection.")
    if identity.get("model_type") != MODEL_TYPE:
        raise ValueError("TrainingResult identity.model_type must be autoencoder.")
    if metadata.get("dataset_id") != DATASET_ID:
        raise ValueError("TrainingResult metadata.dataset_id must be mvtec_anomaly.")
    if metadata.get("model_name") != MODEL_TYPE:
        raise ValueError("TrainingResult metadata.model_name must be autoencoder.")
    if "reconstruction_loss" not in metrics:
        raise ValueError("TrainingResult metrics must include reconstruction_loss.")
    if "train_loss" not in learning_curves:
        raise ValueError("TrainingResult learning_curves must include train_loss.")
    if "val_loss" not in learning_curves:
        raise ValueError("TrainingResult learning_curves must include val_loss.")
    _model_checkpoint_path(artifacts)


def _validate_evaluation(
    evaluation: dict[str, Any], run_id: str, metadata: dict[str, Any]
) -> None:
    if evaluation.get("artifact_type") != "anomaly_detection_evaluation":
        raise ValueError("Evaluation artifact_type must be anomaly_detection_evaluation.")
    if evaluation.get("task_type") != TASK_TYPE:
        raise ValueError("Evaluation task_type must be anomaly_detection.")
    if evaluation.get("track_id") != TRACK_ID:
        raise ValueError("Evaluation track_id must be track_b.")
    if evaluation.get("run_id") != run_id:
        raise ValueError("Evaluation run_id must match TrainingResult run_id.")
    if evaluation.get("model_type") != MODEL_TYPE:
        raise ValueError("Evaluation model_type must be autoencoder.")
    if evaluation.get("dataset_id") != metadata.get("dataset_id"):
        raise ValueError("Evaluation dataset_id must match TrainingResult metadata.")

    for section in ("metrics", "train_score_summary", "test_score_summary", "counts"):
        _require_dict(evaluation.get(section), f"evaluation.{section}")
    samples = evaluation.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Evaluation samples must be a non-empty list.")


def _validate_inventory(inventory: dict[str, Any]) -> None:
    artifacts = _require_dict(inventory.get("artifacts"), "inventory.artifacts")
    for name in ("training_result", "model_checkpoint", "anomaly_evaluation"):
        artifact = _require_dict(artifacts.get(name), f"inventory.artifacts.{name}")
        if artifact.get("exists") is not True:
            raise ValueError(f"Inventory artifact {name} must exist.")
        _require_string(artifact.get("path"), f"inventory.artifacts.{name}.path")
        _require_string(artifact.get("sha256"), f"inventory.artifacts.{name}.sha256")


def _model_checkpoint_path(artifacts: dict[str, Any]) -> Path:
    model_artifact = _require_dict(
        artifacts.get("model_artifact"), "artifacts.model_artifact"
    )
    path = Path(_require_string(model_artifact.get("path"), "model_artifact.path"))
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return path


def _artifact_entry(
    path: Path,
    artifact_type: str,
    purpose: str,
    frontend_ready: bool,
    required_for_frontend: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return {
        "path": str(path),
        "exists": True,
        "artifact_type": artifact_type,
        "purpose": purpose,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "frontend_ready": frontend_ready,
        "required_for_frontend": required_for_frontend,
    }


def _load_json_file(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} JSON not found: {path}")
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
