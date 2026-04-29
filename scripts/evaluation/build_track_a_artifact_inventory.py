"""Build a frontend-ready Track A classification artifact inventory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Track A classification artifact inventory JSON."
    )
    parser.add_argument(
        "--training-result",
        required=True,
        help="Path to the required TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--sample-predictions",
        help="Optional path to a sample_predictions JSON artifact.",
    )
    parser.add_argument(
        "--heatmaps",
        help="Optional path to a classification_heatmaps JSON artifact.",
    )
    parser.add_argument(
        "--comparison",
        help="Optional path to a Track A comparison JSON artifact.",
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
    training_result = _load_json_file(training_result_path, "TrainingResult")
    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")
    artifacts = _require_dict(training_result.get("artifacts"), "artifacts")

    run_id = _require_string(identity.get("run_id"), "identity.run_id")
    task_type = _require_string(identity.get("task_type"), "identity.task_type")
    if task_type != "classification":
        raise ValueError("Track A artifact inventory only supports classification.")

    dataset_id = metadata.get("dataset_id")
    model_id = metadata.get("model_name")
    config_id = identity.get("run_config_id") or metadata.get("training_config_id")

    inventory = {
        "artifact_type": "track_a_artifact_inventory",
        "track_id": "track_a",
        "task_type": "classification",
        "run_id": run_id,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "config_id": config_id,
        "created_at": _utc_now_iso(),
        "source_training_result": str(training_result_path),
        "artifacts": {
            "training_result": _present_entry(
                path=training_result_path,
                artifact_type="TrainingResult",
                purpose="Source of truth for the Track A classification run.",
                frontend_ready=True,
                required_for_frontend=True,
            ),
            "model_checkpoint": _model_checkpoint_entry(artifacts),
            "sample_predictions": _optional_artifact_entry(
                path_value=args.sample_predictions,
                artifact_name="sample_predictions",
                expected_artifact_type="sample_predictions",
                purpose="Frontend sample-level prediction examples.",
                run_id=run_id,
                dataset_id=dataset_id,
                frontend_ready=True,
                required_for_frontend=False,
            ),
            "classification_heatmaps": _optional_artifact_entry(
                path_value=args.heatmaps,
                artifact_name="classification_heatmaps",
                expected_artifact_type="classification_heatmaps",
                purpose="Frontend Grad-CAM heatmap and overlay references.",
                run_id=run_id,
                dataset_id=dataset_id,
                frontend_ready=True,
                required_for_frontend=False,
            ),
            "comparison": _comparison_entry(
                path_value=args.comparison,
                run_id=run_id,
                dataset_id=dataset_id,
            ),
            "evaluation_metrics": _evaluation_metrics_entry(
                metadata=metadata,
                run_id=run_id,
                dataset_id=dataset_id,
            ),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"track_a_artifact_inventory__{run_id}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2)

    print(f"track_a_artifact_inventory_path={output_path}")
    return 0


def _model_checkpoint_entry(artifacts: dict[str, Any]) -> dict[str, Any]:
    model_artifact = artifacts.get("model_artifact")
    if model_artifact is None:
        return _not_provided_entry(
            artifact_type="model_checkpoint",
            purpose="Model checkpoint referenced by the TrainingResult.",
            required_for_frontend=True,
        )
    path_value = (
        model_artifact.get("path") if isinstance(model_artifact, dict) else model_artifact
    )
    path = Path(_require_string(path_value, "artifacts.model_artifact.path"))
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return _present_entry(
        path=path,
        artifact_type="model_checkpoint",
        purpose="Model checkpoint referenced by the TrainingResult.",
        frontend_ready=False,
        required_for_frontend=True,
    )


def _optional_artifact_entry(
    path_value: str | None,
    artifact_name: str,
    expected_artifact_type: str,
    purpose: str,
    run_id: str,
    dataset_id: Any,
    frontend_ready: bool,
    required_for_frontend: bool,
) -> dict[str, Any]:
    if path_value is None:
        return _not_provided_entry(
            artifact_type=None,
            purpose=purpose,
            required_for_frontend=required_for_frontend,
        )

    path = Path(_require_string(path_value, artifact_name))
    payload = _load_json_file(path, artifact_name)
    if payload.get("artifact_type") != expected_artifact_type:
        raise ValueError(
            f"{artifact_name} artifact_type must be {expected_artifact_type}."
        )
    _validate_run_and_dataset(payload, artifact_name, run_id, dataset_id)
    return _present_entry(
        path=path,
        artifact_type=expected_artifact_type,
        purpose=purpose,
        frontend_ready=frontend_ready,
        required_for_frontend=required_for_frontend,
    )


def _comparison_entry(
    path_value: str | None,
    run_id: str,
    dataset_id: Any,
) -> dict[str, Any]:
    purpose = "Track A model comparison and recommendation context."
    if path_value is None:
        return _not_provided_entry(
            artifact_type=None,
            purpose=purpose,
            required_for_frontend=False,
        )

    path = Path(_require_string(path_value, "comparison"))
    payload = _load_json_file(path, "comparison")
    _validate_comparison(payload, run_id, dataset_id)
    return _present_entry(
        path=path,
        artifact_type=payload.get("artifact_type"),
        purpose=purpose,
        frontend_ready=True,
        required_for_frontend=False,
    )


def _evaluation_metrics_entry(
    metadata: dict[str, Any],
    run_id: str,
    dataset_id: Any,
) -> dict[str, Any]:
    purpose = "Validation evaluation metrics referenced by the TrainingResult."
    path_value = metadata.get("validation_evaluation_path")
    if path_value is None:
        return _not_provided_entry(
            artifact_type=None,
            purpose=purpose,
            required_for_frontend=False,
        )

    path = Path(_require_string(path_value, "metadata.validation_evaluation_path"))
    payload = _load_json_file(path, "evaluation_metrics")
    _validate_run_and_dataset(payload, "evaluation_metrics", run_id, dataset_id)
    return _present_entry(
        path=path,
        artifact_type=payload.get("artifact_type"),
        purpose=purpose,
        frontend_ready=True,
        required_for_frontend=False,
    )


def _validate_run_and_dataset(
    payload: dict[str, Any],
    artifact_name: str,
    run_id: str,
    dataset_id: Any,
) -> None:
    artifact_run_id = payload.get("run_id")
    if artifact_run_id is not None and artifact_run_id != run_id:
        raise ValueError(f"{artifact_name} run_id must match TrainingResult run_id.")
    if dataset_id and payload.get("dataset_id") != dataset_id:
        raise ValueError(
            f"{artifact_name} dataset_id must match TrainingResult dataset_id."
        )


def _validate_comparison(
    payload: dict[str, Any],
    run_id: str,
    dataset_id: Any,
) -> None:
    _validate_run_and_dataset(payload, "comparison", run_id, dataset_id)
    if payload.get("run_id") is None:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(
                "comparison must contain run_id or a candidates list with run_id values."
            )
        candidate_run_ids = [
            candidate.get("run_id")
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        if run_id not in candidate_run_ids:
            raise ValueError("comparison candidates must include TrainingResult run_id.")


def _present_entry(
    path: Path,
    artifact_type: Any,
    purpose: str,
    frontend_ready: bool,
    required_for_frontend: bool,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "artifact_type": artifact_type,
        "purpose": purpose,
        "frontend_ready": frontend_ready,
        "required_for_frontend": required_for_frontend,
    }


def _not_provided_entry(
    artifact_type: Any,
    purpose: str,
    required_for_frontend: bool,
) -> dict[str, Any]:
    return {
        "path": None,
        "exists": False,
        "artifact_type": artifact_type,
        "purpose": purpose,
        "frontend_ready": False,
        "required_for_frontend": required_for_frontend,
        "status": "not_provided",
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
