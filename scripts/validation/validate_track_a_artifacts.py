"""Validate the canonical Track A artifact set for frontend and CI readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


TRACK_A_DATASET_ID = "mvtec_classification_supervised"
REQUIRED_MODEL_TYPES = {"mlp", "cnn", "resnet18"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate canonical Track A classification artifacts."
    )
    parser.add_argument("--mlp-training-result", required=True)
    parser.add_argument("--cnn-training-result", required=True)
    parser.add_argument("--resnet18-training-result", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--inventory", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    training_results = {
        "mlp": _validate_training_result(Path(args.mlp_training_result), "mlp"),
        "cnn": _validate_training_result(Path(args.cnn_training_result), "cnn"),
        "resnet18": _validate_training_result(
            Path(args.resnet18_training_result), "resnet18"
        ),
    }
    for model_type, training_result in training_results.items():
        _validate_evaluation_artifact(training_result, model_type)

    _validate_comparison(Path(args.comparison), training_results)
    _validate_inventory(Path(args.inventory), training_results["cnn"])

    print("track_a_artifact_contract=pass")
    return 0


def _validate_training_result(path: Path, expected_model_type: str) -> dict[str, Any]:
    payload = _load_json_file(path, f"{expected_model_type} TrainingResult")
    identity = _require_dict(payload.get("identity"), f"{expected_model_type}.identity")
    metadata = _require_dict(payload.get("metadata"), f"{expected_model_type}.metadata")
    artifacts = _require_dict(
        payload.get("artifacts"), f"{expected_model_type}.artifacts"
    )

    if identity.get("task_type") != "classification":
        raise ValueError(f"{expected_model_type} TrainingResult task_type must be classification.")
    if identity.get("model_type") != expected_model_type:
        raise ValueError(
            f"{expected_model_type} TrainingResult model_type must be {expected_model_type}."
        )
    if metadata.get("dataset_id") != TRACK_A_DATASET_ID:
        raise ValueError(
            f"{expected_model_type} TrainingResult dataset_id must be {TRACK_A_DATASET_ID}."
        )

    model_artifact = _require_dict(
        artifacts.get("model_artifact"),
        f"{expected_model_type}.artifacts.model_artifact",
    )
    model_artifact_path = Path(
        _require_string(
            model_artifact.get("path"), f"{expected_model_type}.model_artifact.path"
        )
    )
    if not model_artifact_path.is_file():
        raise FileNotFoundError(
            f"{expected_model_type} model artifact not found: {model_artifact_path}"
        )

    evaluation_path = Path(
        _require_string(
            metadata.get("validation_evaluation_path"),
            f"{expected_model_type}.metadata.validation_evaluation_path",
        )
    )
    if not evaluation_path.is_file():
        raise FileNotFoundError(
            f"{expected_model_type} validation evaluation not found: {evaluation_path}"
        )

    return payload


def _validate_evaluation_artifact(
    training_result: dict[str, Any],
    model_type: str,
) -> None:
    identity = _require_dict(training_result.get("identity"), f"{model_type}.identity")
    metadata = _require_dict(training_result.get("metadata"), f"{model_type}.metadata")
    evaluation_path = Path(
        _require_string(
            metadata.get("validation_evaluation_path"),
            f"{model_type}.metadata.validation_evaluation_path",
        )
    )
    evaluation = _load_json_file(evaluation_path, f"{model_type} evaluation artifact")

    if evaluation.get("artifact_type") != "classification_validation_evaluation":
        raise ValueError(
            f"{model_type} evaluation artifact_type must be classification_validation_evaluation."
        )
    if evaluation.get("run_id") != identity.get("run_id"):
        raise ValueError(f"{model_type} evaluation run_id must match TrainingResult.")
    if evaluation.get("dataset_id") != metadata.get("dataset_id"):
        raise ValueError(f"{model_type} evaluation dataset_id must match TrainingResult.")
    if "confusion_matrix" not in evaluation:
        raise ValueError(f"{model_type} evaluation must include confusion_matrix.")
    if "macro_metrics" not in evaluation:
        raise ValueError(f"{model_type} evaluation must include macro_metrics.")


def _validate_comparison(
    path: Path,
    training_results: dict[str, dict[str, Any]],
) -> None:
    comparison = _load_json_file(path, "Track A comparison")
    if comparison.get("artifact_type") != "track_a_comparison":
        raise ValueError("Comparison artifact_type must be track_a_comparison.")
    if comparison.get("candidate_count") != 3:
        raise ValueError("Comparison candidate_count must be 3.")
    candidates = comparison.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Comparison candidates must be a list.")

    candidate_model_types = {
        candidate.get("model_type") for candidate in candidates if isinstance(candidate, dict)
    }
    if candidate_model_types != REQUIRED_MODEL_TYPES:
        raise ValueError(
            f"Comparison candidate model types must be {sorted(REQUIRED_MODEL_TYPES)}."
        )

    expected_run_ids = {
        _require_dict(result.get("identity"), f"{model_type}.identity").get("run_id")
        for model_type, result in training_results.items()
    }
    candidate_run_ids = {
        candidate.get("run_id") for candidate in candidates if isinstance(candidate, dict)
    }
    if candidate_run_ids != expected_run_ids:
        raise ValueError("Comparison candidate run_id values must match provided TrainingResults.")

    if not isinstance(comparison.get("decision_policy"), dict):
        raise ValueError("Comparison must include decision_policy.")
    if not isinstance(comparison.get("recommended_candidate"), dict):
        raise ValueError("Comparison must include recommended_candidate.")


def _validate_inventory(path: Path, cnn_training_result: dict[str, Any]) -> None:
    inventory = _load_json_file(path, "Track A artifact inventory")
    if inventory.get("artifact_type") != "track_a_artifact_inventory":
        raise ValueError("Inventory artifact_type must be track_a_artifact_inventory.")

    cnn_identity = _require_dict(cnn_training_result.get("identity"), "cnn.identity")
    if inventory.get("run_id") != cnn_identity.get("run_id"):
        raise ValueError("Inventory run_id must match CNN TrainingResult run_id.")

    artifacts = _require_dict(inventory.get("artifacts"), "inventory.artifacts")
    _require_inventory_exists(artifacts, "training_result")
    _require_inventory_exists(artifacts, "model_checkpoint")
    _require_inventory_exists(artifacts, "sample_predictions")
    _require_inventory_exists(artifacts, "classification_heatmaps")

    evaluation_metrics = _require_dict(
        artifacts.get("evaluation_metrics"), "inventory.artifacts.evaluation_metrics"
    )
    if evaluation_metrics.get("artifact_type") != "classification_validation_evaluation":
        raise ValueError(
            "Inventory evaluation_metrics artifact_type must be classification_validation_evaluation."
        )

    comparison = _require_dict(
        artifacts.get("comparison"), "inventory.artifacts.comparison"
    )
    if comparison.get("artifact_type") != "track_a_comparison":
        raise ValueError("Inventory comparison artifact_type must be track_a_comparison.")


def _require_inventory_exists(artifacts: dict[str, Any], name: str) -> None:
    entry = _require_dict(artifacts.get(name), f"inventory.artifacts.{name}")
    if entry.get("exists") is not True:
        raise ValueError(f"Inventory artifact {name} must have exists=true.")


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


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
