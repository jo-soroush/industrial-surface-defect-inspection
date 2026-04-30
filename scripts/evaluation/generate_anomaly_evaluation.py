"""Generate governed Track B anomaly detection evaluation artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from numbers import Real
from pathlib import Path
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
from inspection_ai.models.autoencoder import AutoencoderModel
from inspection_ai.models.factory import create_model
from inspection_ai.training.data_loading import build_data_loaders


SUPPORTED_THRESHOLD_STRATEGIES = {"mean", "percentile95"}
EXPECTED_TASK_TYPE = "anomaly_detection"
EXPECTED_MODEL_TYPE = "autoencoder"
EXPECTED_DATASET_ID = "mvtec_anomaly"
EXPECTED_MODEL_ARTIFACT_TYPE = "pytorch_state_dict"
OUTPUT_FILENAME_TEMPLATE = "anomaly_detection_evaluation__{run_id}__test.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Track B anomaly detection evaluation metrics."
    )
    parser.add_argument(
        "--training-result",
        required=True,
        help="Path to a Track B autoencoder TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=sorted(SUPPORTED_THRESHOLD_STRATEGIES),
        default="percentile95",
        help="Threshold strategy computed from train scores only.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/metrics",
        help="Directory where the anomaly evaluation artifact will be written.",
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
    _validate_training_result_identity(identity=identity, metadata=metadata)
    checkpoint_path = _resolve_checkpoint_path(artifacts)
    split_manifest_path = Path(
        _require_string(metadata.get("split_manifest_path"), "metadata.split_manifest_path")
    )
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")

    config = _load_run_config(identity, metadata)
    data_loaders = build_data_loaders(config)
    train_loader = _require_non_empty_loader(data_loaders, "train")
    test_loader = _require_non_empty_loader(data_loaders, "test")

    model = _build_autoencoder_model(config)
    _load_model_weights(model, checkpoint_path)
    model.eval()

    train_inference = run_anomaly_inference(model, train_loader)
    threshold = compute_threshold(
        train_inference["scores"],
        {"threshold_strategy": args.threshold_strategy},
    )
    _validate_numeric(threshold, "threshold")

    test_inference = run_anomaly_inference(model, test_loader)
    labels = _require_binary_label_mix(test_inference["labels"])
    predictions = generate_predictions(test_inference["scores"], threshold)
    metrics = compute_anomaly_metrics(
        labels=labels,
        scores=test_inference["scores"],
        predictions=predictions,
    )
    _validate_metrics(metrics)

    samples = _build_samples(
        test_entries=_require_split_entries(data_loaders, "test"),
        inference=test_inference,
        predictions=predictions,
    )
    counts = _build_counts(
        train_scores=train_inference["scores"],
        labels=labels,
        predictions=predictions,
        samples=samples,
    )

    _validate_output_counts(
        scores=test_inference["scores"],
        labels=labels,
        predictions=predictions,
        samples=samples,
    )

    payload = {
        "artifact_type": "anomaly_detection_evaluation",
        "task_type": EXPECTED_TASK_TYPE,
        "track_id": "track_b",
        "run_id": run_id,
        "model_id": metadata.get("model_name"),
        "model_type": EXPECTED_MODEL_TYPE,
        "dataset_id": EXPECTED_DATASET_ID,
        "config_id": identity.get("run_config_id") or metadata.get("training_config_id"),
        "source_training_result": str(training_result_path),
        "source_model_checkpoint": str(checkpoint_path),
        "split_manifest_path": str(split_manifest_path),
        "preprocessing_policy_path": _preprocessing_policy_path(config),
        "created_at": _utc_now_iso(),
        "score_definition": "mean_squared_reconstruction_error_per_image",
        "threshold_strategy": args.threshold_strategy,
        "threshold": float(threshold),
        "metrics": metrics,
        "train_score_summary": _score_summary(train_inference["scores"]),
        "test_score_summary": _score_summary(test_inference["scores"]),
        "counts": counts,
        "samples": samples,
    }
    _validate_payload(payload)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME_TEMPLATE.format(run_id=run_id)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"anomaly_detection_evaluation_artifact_path={output_path}")
    print(f"threshold_strategy={args.threshold_strategy}")
    print(f"threshold={float(threshold):.10f}")
    print(f"test_score_count={counts['test_score_count']}")
    return 0


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


def _load_yaml_file(path: Path, config_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{config_name} file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"{config_name} YAML is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{config_name} YAML must contain an object: {path}")
    return payload


def _validate_training_result_identity(
    identity: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    task_type = _require_string(identity.get("task_type"), "identity.task_type")
    if task_type != EXPECTED_TASK_TYPE:
        raise ValueError("TrainingResult identity.task_type must be anomaly_detection.")

    model_type = _require_string(identity.get("model_type"), "identity.model_type")
    if model_type != EXPECTED_MODEL_TYPE:
        raise ValueError("TrainingResult identity.model_type must be autoencoder.")

    dataset_id = _require_string(metadata.get("dataset_id"), "metadata.dataset_id")
    if dataset_id != EXPECTED_DATASET_ID:
        raise ValueError("TrainingResult metadata.dataset_id must be mvtec_anomaly.")

    model_name = _require_string(metadata.get("model_name"), "metadata.model_name")
    if model_name != EXPECTED_MODEL_TYPE:
        raise ValueError("TrainingResult metadata.model_name must be autoencoder.")

    _require_string(identity.get("run_config_id"), "identity.run_config_id")


def _resolve_checkpoint_path(artifacts: dict[str, Any]) -> Path:
    model_artifact = _require_dict(
        artifacts.get("model_artifact"),
        "artifacts.model_artifact",
    )
    artifact_type = _require_string(
        model_artifact.get("type"),
        "artifacts.model_artifact.type",
    )
    if artifact_type != EXPECTED_MODEL_ARTIFACT_TYPE:
        raise ValueError("artifacts.model_artifact.type must be pytorch_state_dict.")

    checkpoint_path = Path(
        _require_string(
            model_artifact.get("path"),
            "artifacts.model_artifact.path",
        )
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _load_run_config(
    identity: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    run_config_id = _require_string(identity.get("run_config_id"), "identity.run_config_id")
    config_path = Path("configs/runs") / f"{run_config_id}.yaml"
    config = _load_yaml_file(config_path, "run config")
    config_identity = _require_dict(config.get("identity"), "config.identity")
    if config_identity.get("run_config_id") != run_config_id:
        raise ValueError("Run config identity.run_config_id does not match TrainingResult.")

    dataset_binding = _require_dict(config.get("dataset_binding"), "config.dataset_binding")
    expected_split_manifest = _require_string(
        metadata.get("split_manifest_path"),
        "metadata.split_manifest_path",
    )
    if dataset_binding.get("split_manifest_path") != expected_split_manifest:
        raise ValueError(
            "Run config dataset_binding.split_manifest_path does not match "
            "TrainingResult metadata.split_manifest_path."
        )
    if dataset_binding.get("dataset_id") != metadata.get("dataset_id"):
        raise ValueError(
            "Run config dataset_binding.dataset_id does not match "
            "TrainingResult metadata.dataset_id."
        )

    return config


def _build_autoencoder_model(config: dict[str, Any]) -> Any:
    try:
        model = create_model(config)
    except Exception as exc:
        raise ValueError("Model factory failed to create autoencoder model.") from exc
    if not isinstance(model, AutoencoderModel):
        raise ValueError("Model factory did not create an AutoencoderModel.")
    return model


def _load_model_weights(model: Any, checkpoint_path: Path) -> None:
    if not hasattr(model, "load_state_dict"):
        raise ValueError("Configured model does not support load_state_dict.")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break

    if not isinstance(state_dict, dict):
        raise ValueError("Model checkpoint does not contain a valid state_dict.")
    model.load_state_dict(state_dict)


def _require_non_empty_loader(data_loaders: dict[str, Any], split_name: str) -> Any:
    loader = data_loaders.get(f"{split_name}_loader")
    if loader is None:
        raise ValueError(f"{split_name}_loader is missing.")
    entries = _require_split_entries(data_loaders, split_name)
    if not entries:
        raise ValueError(f"{split_name}_loader has no governed entries.")
    return loader


def _require_split_entries(
    data_loaders: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    entries = data_loaders.get(split_name)
    if not isinstance(entries, list):
        raise ValueError(f"data_loaders.{split_name} must be a list.")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"data_loaders.{split_name}[{index}] must be a dictionary.")
    return entries


def _require_binary_label_mix(labels: Any) -> list[int]:
    if not isinstance(labels, list) or not labels:
        raise ValueError("Test labels must be a non-empty list.")
    normalized = []
    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, int) or label not in {0, 1}:
            raise ValueError(f"Test labels[{index}] must be 0 or 1.")
        normalized.append(label)
    if set(normalized) != {0, 1}:
        raise ValueError("Test labels must contain both normal and anomaly samples.")
    return normalized


def _build_samples(
    test_entries: list[dict[str, Any]],
    inference: dict[str, list[Any]],
    predictions: list[int],
) -> list[dict[str, Any]]:
    scores = inference.get("scores")
    labels = inference.get("labels")
    paths = inference.get("paths")
    mask_paths = inference.get("mask_paths")
    if not all(isinstance(values, list) for values in (scores, labels, paths, mask_paths)):
        raise ValueError("Anomaly inference output lists are invalid.")
    if not (len(scores) == len(labels) == len(paths) == len(mask_paths) == len(predictions)):
        raise ValueError("Anomaly score, label, path, mask, and prediction counts differ.")
    if len(test_entries) != len(scores):
        raise ValueError("Test manifest entry count must match anomaly score count.")

    samples = []
    for index, entry in enumerate(test_entries):
        true_label_id = _require_int(labels[index], f"labels[{index}]")
        predicted_label_id = _require_int(predictions[index], f"predictions[{index}]")
        expected_path = _require_string(entry.get("image_path"), f"test_entries[{index}].image_path")
        path = _require_string(paths[index], f"paths[{index}]")
        if path != expected_path:
            raise ValueError("Inference path order does not match test manifest order.")

        true_label = _label_name(true_label_id)
        predicted_label = _label_name(predicted_label_id)
        mask_path = mask_paths[index]
        if mask_path == "":
            mask_path = None
        if mask_path is not None and not isinstance(mask_path, str):
            raise ValueError(f"mask_paths[{index}] must be a string or null.")

        samples.append(
            {
                "sample_id": index,
                "image_path": path,
                "true_label": true_label,
                "true_label_id": true_label_id,
                "defect_type": _require_string(
                    entry.get("defect_type"),
                    f"test_entries[{index}].defect_type",
                ),
                "mask_path": mask_path,
                "anomaly_score": float(_require_numeric(scores[index], f"scores[{index}]")),
                "predicted_label": predicted_label,
                "predicted_label_id": predicted_label_id,
                "correct": true_label_id == predicted_label_id,
            }
        )
    return samples


def _build_counts(
    train_scores: list[float],
    labels: list[int],
    predictions: list[int],
    samples: list[dict[str, Any]],
) -> dict[str, int]:
    correct_count = sum(1 for sample in samples if sample.get("correct") is True)
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


def _validate_output_counts(
    scores: list[float],
    labels: list[int],
    predictions: list[int],
    samples: list[dict[str, Any]],
) -> None:
    if not (len(scores) == len(labels) == len(predictions) == len(samples)):
        raise ValueError("Output score, label, prediction, and sample counts differ.")


def _validate_metrics(metrics: dict[str, Any]) -> None:
    for key in ("roc_auc", "precision", "recall", "f1"):
        if key not in metrics:
            raise ValueError(f"Anomaly metrics missing required field: {key}.")
        value = metrics[key]
        if key == "roc_auc" and value is None:
            continue
        _validate_numeric(value, f"metrics.{key}")


def _validate_payload(payload: dict[str, Any]) -> None:
    _validate_numeric(payload.get("threshold"), "threshold")
    _validate_metrics(_require_dict(payload.get("metrics"), "metrics"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Output payload samples must be a non-empty list.")


def _score_summary(scores: list[float]) -> dict[str, float | int]:
    values = [float(_require_numeric(score, "score")) for score in scores]
    if not values:
        raise ValueError("Score summary requires at least one score.")
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


def _preprocessing_policy_path(config: dict[str, Any]) -> str | None:
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        return None
    value = preprocessing.get("preprocessing_policy_path")
    if value is None:
        return None
    return _require_string(value, "config.preprocessing.preprocessing_policy_path")


def _label_name(label_id: int) -> str:
    if label_id == 0:
        return "normal"
    if label_id == 1:
        return "anomaly"
    raise ValueError(f"Unsupported label id: {label_id}")


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _require_numeric(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _validate_numeric(value: Any, field_name: str) -> None:
    _require_numeric(value, field_name)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
