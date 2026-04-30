"""Run a governed production-canonical Track B anomaly detection pipeline.

This script intentionally supports only the implemented Track B autoencoder
anomaly path. Classification belongs to Track A, and object detection is not a
real trainable path in this repository yet.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from numbers import Real
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import torch
import yaml

from inspection_ai.evaluation.anomaly_evaluation import (
    compute_anomaly_metrics,
    compute_threshold,
    generate_predictions,
    run_anomaly_inference,
)
from inspection_ai.models.factory import create_model
from inspection_ai.training.checkpointing import (
    resolve_model_checkpoint_path,
    save_checkpoint,
)
from inspection_ai.training.data_loading import build_data_loaders
from inspection_ai.training.result_persistence import persist_training_result
from inspection_ai.training.result_validation import validate_training_result
from inspection_ai.training.train_loop import run_training_loop


TRACK_ID = "track_b"
TASK_TYPE = "anomaly_detection"
DATASET_ID = "mvtec_anomaly"
MODEL_TYPE = "autoencoder"
CANONICAL_STATUS = "production-canonical"
MODEL_ARTIFACT_TYPE = "pytorch_state_dict"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the production-canonical Track B autoencoder pipeline."
    )
    parser.add_argument(
        "--run-config",
        default="configs/runs/autoencoder_train_v0_1_0.yaml",
        help="Governed Track B autoencoder run config.",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=("mean", "percentile95"),
        default="percentile95",
        help="Threshold strategy computed from train scores only.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/models",
        help="Root directory for governed model artifacts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts_root = Path(args.artifacts_root)
    config = _load_run_config(Path(args.run_config))
    config = _production_canonical_config(config)
    _validate_supported_track_b_config(config)
    _set_reproducibility(config)

    data_loaders = build_data_loaders(config)
    _validate_loader_contract(data_loaders)

    model = create_model(config)
    result = run_training_loop(config=config, model=model, data_loader=data_loaders)
    result.add_metadata("canonical_status", CANONICAL_STATUS)
    result.add_metadata("canonical_track_id", TRACK_ID)
    result.add_metadata("canonical_policy", "production_canonical_track_b")

    checkpoint_path = resolve_model_checkpoint_path(result.identity["run_id"])
    save_checkpoint(model.state_dict(), checkpoint_path)
    result.add_artifact(
        "model_artifact",
        {
            "path": str(checkpoint_path),
            "type": MODEL_ARTIFACT_TYPE,
        },
    )
    validate_training_result(result)

    training_result_path = persist_training_result(
        result=result,
        output_dir=artifacts_root / "analysis" / "training_results",
    )

    evaluation_path = _write_anomaly_evaluation(
        config=config,
        training_result_path=training_result_path,
        checkpoint_path=checkpoint_path,
        data_loaders=data_loaders,
        threshold_strategy=args.threshold_strategy,
        output_dir=artifacts_root / "metrics",
    )
    inventory_path = _write_inventory(
        training_result_path=training_result_path,
        checkpoint_path=checkpoint_path,
        evaluation_path=evaluation_path,
        output_dir=artifacts_root / "inventory",
    )
    validation = _validate_artifacts_read_only(
        training_result_path=training_result_path,
        checkpoint_path=checkpoint_path,
        evaluation_path=evaluation_path,
        inventory_path=inventory_path,
    )
    summary_path = _write_summary(
        run_id=result.identity["run_id"],
        validation=validation,
        training_result_path=training_result_path,
        checkpoint_path=checkpoint_path,
        evaluation_path=evaluation_path,
        inventory_path=inventory_path,
        output_dir=artifacts_root / "metadata",
    )

    print(f"run_id={result.identity['run_id']}")
    print(f"canonical_status={CANONICAL_STATUS}")
    print(f"training_result={training_result_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"evaluation={evaluation_path}")
    print(f"inventory={inventory_path}")
    print(f"summary={summary_path}")
    print(f"validation_result={validation['status']}")
    return 0


def _load_run_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Run config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Run config YAML is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Run config must contain a YAML object.")
    return payload


def _production_canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    production_config = deepcopy(config)
    identity = _require_dict(production_config.get("identity"), "identity")
    identity["is_experiment"] = False
    identity["track_id"] = TRACK_ID
    return production_config


def _validate_supported_track_b_config(config: dict[str, Any]) -> None:
    identity = _require_dict(config.get("identity"), "identity")
    model_identity = _require_dict(config.get("model_identity"), "model_identity")
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    if identity.get("task_type") != TASK_TYPE:
        raise ValueError("Production Track B script only supports anomaly_detection.")
    if identity.get("track_id") != TRACK_ID:
        raise ValueError("Production Track B config identity.track_id must be track_b.")
    if identity.get("is_experiment") is not False:
        raise ValueError("Production Track B runs require identity.is_experiment=false.")
    if model_identity.get("model_type") != MODEL_TYPE:
        raise ValueError("Production Track B script only supports autoencoder.")
    if dataset_binding.get("dataset_id") != DATASET_ID:
        raise ValueError("Production Track B dataset_id must be mvtec_anomaly.")
    split_manifest_path = Path(
        _require_string(dataset_binding.get("split_manifest_path"), "split_manifest_path")
    )
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")


def _set_reproducibility(config: dict[str, Any]) -> None:
    runtime = _require_dict(config.get("training_runtime"), "training_runtime")
    seed = runtime.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training_runtime.seed must be an integer.")
    random.seed(seed)
    torch.manual_seed(seed)


def _validate_loader_contract(data_loaders: dict[str, Any]) -> None:
    if data_loaders.get("task_type") != TASK_TYPE:
        raise ValueError("Data loader task_type must be anomaly_detection.")
    for split_name in ("train", "test"):
        entries = data_loaders.get(split_name)
        loader = data_loaders.get(f"{split_name}_loader")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{split_name} split must be non-empty.")
        if loader is None:
            raise ValueError(f"{split_name}_loader must exist.")
    validation_entries = data_loaders.get("validation")
    if not isinstance(validation_entries, list):
        raise ValueError("validation split must be a list.")


def _write_anomaly_evaluation(
    config: dict[str, Any],
    training_result_path: Path,
    checkpoint_path: Path,
    data_loaders: dict[str, Any],
    threshold_strategy: str,
    output_dir: Path,
) -> Path:
    training_result = _load_json_file(training_result_path, "TrainingResult")
    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")

    model = create_model(config)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state_dict dictionary.")
    model.load_state_dict(payload)
    model.eval()

    train_inference = run_anomaly_inference(model, data_loaders["train_loader"])
    threshold = compute_threshold(
        train_inference["scores"], {"threshold_strategy": threshold_strategy}
    )
    test_inference = run_anomaly_inference(model, data_loaders["test_loader"])
    labels = _require_binary_label_mix(test_inference["labels"])
    predictions = generate_predictions(test_inference["scores"], threshold)
    metrics = compute_anomaly_metrics(
        labels=labels,
        scores=test_inference["scores"],
        predictions=predictions,
    )
    test_entries = _require_entries(data_loaders, "test")
    samples = _build_samples(test_entries, test_inference, predictions)
    counts = _build_counts(train_inference["scores"], labels, predictions, samples)

    evaluation = {
        "artifact_type": "anomaly_detection_evaluation",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": identity["run_id"],
        "model_id": metadata.get("model_name"),
        "model_type": MODEL_TYPE,
        "dataset_id": DATASET_ID,
        "config_id": identity.get("run_config_id") or metadata.get("training_config_id"),
        "source_training_result": str(training_result_path),
        "source_model_checkpoint": str(checkpoint_path),
        "split_manifest_path": metadata.get("split_manifest_path"),
        "preprocessing_policy_path": _preprocessing_policy_path(config),
        "created_at": _utc_now_iso(),
        "score_definition": "mean_squared_reconstruction_error_per_image",
        "threshold_strategy": threshold_strategy,
        "threshold": float(threshold),
        "metrics": metrics,
        "train_score_summary": _score_summary(train_inference["scores"]),
        "test_score_summary": _score_summary(test_inference["scores"]),
        "counts": counts,
        "samples": samples,
    }
    _validate_evaluation_payload(evaluation)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"anomaly_detection_evaluation__{identity['run_id']}__test.json"
    _write_json(output_path, evaluation)
    return output_path


def _write_inventory(
    training_result_path: Path,
    checkpoint_path: Path,
    evaluation_path: Path,
    output_dir: Path,
) -> Path:
    training_result = _load_json_file(training_result_path, "TrainingResult")
    evaluation = _load_json_file(evaluation_path, "anomaly evaluation")
    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")

    run_id = _require_string(identity.get("run_id"), "identity.run_id")
    inventory = {
        "artifact_type": "track_b_artifact_inventory",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "model_id": metadata.get("model_name"),
        "model_type": MODEL_TYPE,
        "dataset_id": metadata.get("dataset_id"),
        "config_id": identity.get("run_config_id") or metadata.get("training_config_id"),
        "created_at": _utc_now_iso(),
        "counts": {
            "train_sample_count": metadata.get("train_sample_count"),
            "validation_sample_count": metadata.get("validation_sample_count"),
            "test_sample_count": metadata.get("test_sample_count"),
            "train_score_count": _require_dict(evaluation.get("counts"), "evaluation.counts").get("train_score_count"),
            "test_score_count": _require_dict(evaluation.get("counts"), "evaluation.counts").get("test_score_count"),
            "normal_test_count": _require_dict(evaluation.get("counts"), "evaluation.counts").get("normal_test_count"),
            "anomaly_test_count": _require_dict(evaluation.get("counts"), "evaluation.counts").get("anomaly_test_count"),
        },
        "linkage": {
            "training_result": str(training_result_path),
            "model_checkpoint": str(checkpoint_path),
            "anomaly_evaluation": str(evaluation_path),
            "evaluation_source_training_result": evaluation.get("source_training_result"),
            "evaluation_source_model_checkpoint": evaluation.get("source_model_checkpoint"),
        },
        "artifacts": {
            "training_result": _artifact_entry(training_result_path, "TrainingResult"),
            "model_checkpoint": _artifact_entry(checkpoint_path, "model_checkpoint"),
            "anomaly_evaluation": _artifact_entry(
                evaluation_path, "anomaly_detection_evaluation"
            ),
        },
    }
    _validate_inventory_payload(inventory)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"track_b_artifact_inventory__{run_id}.json"
    _write_json(output_path, inventory)
    return output_path


def _validate_artifacts_read_only(
    training_result_path: Path,
    checkpoint_path: Path,
    evaluation_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    artifacts = {
        "training_result": training_result_path,
        "model_checkpoint": checkpoint_path,
        "anomaly_evaluation": evaluation_path,
        "inventory": inventory_path,
    }
    hashes = {}
    for name, path in artifacts.items():
        if not path.is_file():
            return {"status": "fail", "reason": f"{name} missing: {path}", "hashes": hashes}
        hashes[name] = _sha256(path)

    training_result = _load_json_file(training_result_path, "TrainingResult")
    evaluation = _load_json_file(evaluation_path, "anomaly evaluation")
    inventory = _load_json_file(inventory_path, "Track B inventory")
    _validate_training_result_payload(training_result)
    _validate_evaluation_payload(evaluation)
    _validate_inventory_payload(inventory)
    return {"status": "pass", "hashes": hashes}


def _write_summary(
    run_id: str,
    validation: dict[str, Any],
    training_result_path: Path,
    checkpoint_path: Path,
    evaluation_path: Path,
    inventory_path: Path,
    output_dir: Path,
) -> Path:
    summary = {
        "artifact_type": "track_b_production_canonical_summary",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "created_at": _utc_now_iso(),
        "artifacts": {
            "training_result": _summary_artifact(training_result_path),
            "model_checkpoint": _summary_artifact(checkpoint_path),
            "anomaly_evaluation": _summary_artifact(evaluation_path),
            "inventory": _summary_artifact(inventory_path),
        },
        "validation": validation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"track_b_production_canonical_summary__{run_id}.json"
    _write_json(output_path, summary)
    return output_path


def _validate_training_result_payload(payload: dict[str, Any]) -> None:
    identity = _require_dict(payload.get("identity"), "identity")
    metadata = _require_dict(payload.get("metadata"), "metadata")
    metrics = _require_dict(payload.get("metrics"), "metrics")
    artifacts = _require_dict(payload.get("artifacts"), "artifacts")
    if identity.get("task_type") != TASK_TYPE:
        raise ValueError("TrainingResult task_type mismatch.")
    if identity.get("model_type") != MODEL_TYPE:
        raise ValueError("TrainingResult model_type mismatch.")
    if identity.get("is_experiment") is not False:
        raise ValueError("Production TrainingResult requires is_experiment=false.")
    if metadata.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("TrainingResult canonical_status mismatch.")
    if metadata.get("dataset_id") != DATASET_ID:
        raise ValueError("TrainingResult dataset_id mismatch.")
    _require_numeric(metrics.get("reconstruction_loss"), "reconstruction_loss")
    checkpoint_path = Path(
        _require_string(
            _require_dict(artifacts.get("model_artifact"), "model_artifact").get("path"),
            "model_artifact.path",
        )
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Linked checkpoint missing: {checkpoint_path}")


def _validate_evaluation_payload(payload: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "anomaly_detection_evaluation":
        raise ValueError("Evaluation artifact_type mismatch.")
    if payload.get("task_type") != TASK_TYPE:
        raise ValueError("Evaluation task_type mismatch.")
    if payload.get("track_id") != TRACK_ID:
        raise ValueError("Evaluation track_id mismatch.")
    _require_numeric(payload.get("threshold"), "threshold")
    metrics = _require_dict(payload.get("metrics"), "metrics")
    for metric_name in ("roc_auc", "precision", "recall", "f1"):
        _require_numeric(metrics.get(metric_name), f"metrics.{metric_name}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Evaluation samples must be a non-empty list.")


def _validate_inventory_payload(payload: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "track_b_artifact_inventory":
        raise ValueError("Inventory artifact_type mismatch.")
    if payload.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("Inventory canonical_status mismatch.")
    artifacts = _require_dict(payload.get("artifacts"), "artifacts")
    for name in ("training_result", "model_checkpoint", "anomaly_evaluation"):
        entry = _require_dict(artifacts.get(name), f"artifacts.{name}")
        path = Path(_require_string(entry.get("path"), f"artifacts.{name}.path"))
        if not path.is_file():
            raise FileNotFoundError(f"Inventory artifact missing: {path}")
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"Inventory checksum mismatch for {name}.")


def _build_samples(
    test_entries: list[dict[str, Any]],
    inference: dict[str, list[Any]],
    predictions: list[int],
) -> list[dict[str, Any]]:
    scores = _require_list(inference.get("scores"), "scores")
    labels = _require_list(inference.get("labels"), "labels")
    paths = _require_list(inference.get("paths"), "paths")
    mask_paths = _require_list(inference.get("mask_paths"), "mask_paths")
    if not (len(scores) == len(labels) == len(paths) == len(mask_paths) == len(predictions) == len(test_entries)):
        raise ValueError("Evaluation sample source counts must match.")

    samples = []
    for index, entry in enumerate(test_entries):
        label_id = _require_binary_int(labels[index], f"labels[{index}]")
        prediction_id = _require_binary_int(predictions[index], f"predictions[{index}]")
        path = _require_string(paths[index], f"paths[{index}]")
        if path != entry.get("image_path"):
            raise ValueError("Inference order does not match test manifest order.")
        mask_path = mask_paths[index]
        if mask_path == "":
            mask_path = None
        samples.append(
            {
                "sample_id": index,
                "image_path": path,
                "true_label": _label_name(label_id),
                "true_label_id": label_id,
                "defect_type": entry.get("defect_type"),
                "mask_path": mask_path,
                "anomaly_score": float(_require_numeric(scores[index], f"scores[{index}]")),
                "predicted_label": _label_name(prediction_id),
                "predicted_label_id": prediction_id,
                "correct": label_id == prediction_id,
            }
        )
    return samples


def _build_counts(
    train_scores: list[float],
    labels: list[int],
    predictions: list[int],
    samples: list[dict[str, Any]],
) -> dict[str, int]:
    correct_count = sum(1 for sample in samples if sample["correct"])
    return {
        "train_score_count": len(train_scores),
        "test_score_count": len(labels),
        "normal_test_count": sum(1 for label in labels if label == 0),
        "anomaly_test_count": sum(1 for label in labels if label == 1),
        "predicted_normal_count": sum(1 for prediction in predictions if prediction == 0),
        "predicted_anomaly_count": sum(1 for prediction in predictions if prediction == 1),
        "correct_count": correct_count,
        "incorrect_count": len(samples) - correct_count,
    }


def _score_summary(scores: list[float]) -> dict[str, float | int]:
    values = [float(_require_numeric(score, "score")) for score in scores]
    if not values:
        raise ValueError("Score summary requires non-empty scores.")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "percentile_95": _percentile(values, 95.0),
    }


def _percentile(values: list[float], percentile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    return sorted_values[lower_index] + (
        (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def _artifact_entry(path: Path, artifact_type: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "artifact_type": artifact_type,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _summary_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _preprocessing_policy_path(config: dict[str, Any]) -> str | None:
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        return None
    value = preprocessing.get("preprocessing_policy_path")
    if value is None:
        return None
    return _require_string(value, "preprocessing_policy_path")


def _require_entries(data_loaders: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    entries = data_loaders.get(split_name)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{split_name} entries must be a non-empty list.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{split_name} entries must contain dictionaries.")
    return entries


def _require_binary_label_mix(labels: Any) -> list[int]:
    values = [_require_binary_int(value, "label") for value in _require_list(labels, "labels")]
    if set(values) != {0, 1}:
        raise ValueError("Test labels must include both normal and anomaly samples.")
    return values


def _label_name(label_id: int) -> str:
    return "anomaly" if label_id == 1 else "normal"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_json_file(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} must contain a JSON object.")
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


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_numeric(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _require_binary_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise ValueError(f"{field_name} must be 0 or 1.")
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
