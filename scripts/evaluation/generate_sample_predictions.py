"""Generate governed sample prediction artifacts for Track A classification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

from inspection_ai.models.factory import create_model
from inspection_ai.training.data_loading import build_data_loaders


SUPPORTED_SPLITS = {"validation", "test"}
SUPPORTED_MODEL_TYPES = {"mlp", "cnn", "resnet18"}
MODEL_ARTIFACT_KEYS = (
    "model",
    "model_artifact",
    "model_artifact_path",
    "checkpoint",
    "checkpoint_path",
    "model_state_dict",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate Track A supervised classification sample predictions."
    )
    parser.add_argument(
        "--training-result",
        required=True,
        help="Path to a TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=sorted(SUPPORTED_SPLITS),
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Maximum number of sample predictions to write.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/predictions",
        help="Directory where the sample prediction artifact will be written.",
    )
    return parser


def main() -> int:
    """Run sample prediction artifact generation."""
    args = build_parser().parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1.")

    training_result_path = Path(args.training_result)
    training_result = _load_json_file(training_result_path, "TrainingResult")
    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")
    artifacts = _require_dict(training_result.get("artifacts"), "artifacts")

    run_id = _require_string(identity.get("run_id"), "identity.run_id")
    task_type = _require_string(identity.get("task_type"), "identity.task_type")
    if task_type != "classification":
        raise ValueError("Sample prediction generation only supports classification.")

    model_type = _require_string(identity.get("model_type"), "identity.model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type '{model_type}'. "
            f"Supported values: {sorted(SUPPORTED_MODEL_TYPES)}."
        )

    config = _load_run_config(training_result)
    class_mapping = _load_class_mapping(config)
    index_to_class = _build_index_to_class(class_mapping)
    data_loaders = build_data_loaders(config)
    split_loader = data_loaders.get(f"{args.split}_loader")
    if split_loader is None:
        raise ValueError(f"Dataset split cannot be loaded: {args.split}.")

    model_artifact_path = _resolve_model_artifact_path(artifacts, metadata)
    model = create_model(config)
    _load_model_weights(model, model_artifact_path)
    model.eval()

    samples = _collect_sample_predictions(
        model=model,
        data_loader=split_loader,
        index_to_class=index_to_class,
        num_requested=args.num_samples,
    )

    output = {
        "artifact_type": "sample_predictions",
        "task_type": "classification",
        "track_id": metadata.get("track_id"),
        "dataset_id": metadata.get("dataset_id"),
        "split": args.split,
        "run_id": run_id,
        "model_id": metadata.get("model_name"),
        "config_id": identity.get("run_config_id") or metadata.get("training_config_id"),
        "source_training_result": str(training_result_path),
        "created_at": _utc_now_iso(),
        "num_requested": args.num_samples,
        "num_written": len(samples),
        "samples": samples,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sample_predictions__{run_id}__{args.split}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"sample_predictions_artifact_path={output_path}")
    print(f"num_written={len(samples)}")
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


def _load_run_config(training_result: dict[str, Any]) -> dict[str, Any]:
    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")
    run_config_id = identity.get("run_config_id") or metadata.get("training_config_id")
    run_config_id = _require_string(run_config_id, "run_config_id")

    for path in sorted(Path("configs/runs").glob("*.yaml")):
        config = _load_yaml_file(path, "run config")
        identity_section = config.get("identity")
        if isinstance(identity_section, dict) and identity_section.get("run_config_id") == run_config_id:
            _validate_config_matches_result(config, metadata)
            return config

    raise FileNotFoundError(f"Run config not found for run_config_id: {run_config_id}")


def _validate_config_matches_result(
    config: dict[str, Any], metadata: dict[str, Any]
) -> None:
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    expected_dataset_id = metadata.get("dataset_id")
    if expected_dataset_id and dataset_binding.get("dataset_id") != expected_dataset_id:
        raise ValueError("Run config dataset_id does not match TrainingResult metadata.")

    expected_split_manifest = metadata.get("split_manifest_path")
    if expected_split_manifest and dataset_binding.get("split_manifest_path") != expected_split_manifest:
        raise ValueError(
            "Run config split_manifest_path does not match TrainingResult metadata."
        )


def _load_class_mapping(config: dict[str, Any]) -> dict[str, Any]:
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    class_mapping_path = _require_string(
        dataset_binding.get("class_mapping_path"),
        "dataset_binding.class_mapping_path",
    )
    return _load_yaml_file(Path(class_mapping_path), "class mapping")


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


def _build_index_to_class(class_mapping: dict[str, Any]) -> dict[int, str]:
    raw_mapping = class_mapping.get("index_to_class")
    if not isinstance(raw_mapping, dict):
        raise ValueError("Class mapping is missing index_to_class.")

    index_to_class = {}
    for key, value in raw_mapping.items():
        if isinstance(key, str) and key.isdigit():
            index = int(key)
        elif isinstance(key, int):
            index = key
        else:
            raise ValueError("Class mapping index_to_class keys must be integers.")
        if not isinstance(value, str) or not value:
            raise ValueError("Class mapping index_to_class values must be strings.")
        index_to_class[index] = value

    if 0 not in index_to_class or 1 not in index_to_class:
        raise ValueError("Class mapping must define classes 0 and 1.")
    return index_to_class


def _resolve_model_artifact_path(
    artifacts: dict[str, Any], metadata: dict[str, Any]
) -> Path:
    for key in MODEL_ARTIFACT_KEYS:
        value = artifacts.get(key) or metadata.get(key)
        if value is None:
            continue
        path_value = value.get("path") if isinstance(value, dict) else value
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"Model artifact field {key} must contain a path string.")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact path does not exist: {path}")
        return path

    raise FileNotFoundError(
        "TrainingResult does not reference a saved model artifact. "
        f"Expected one of artifact/metadata keys: {list(MODEL_ARTIFACT_KEYS)}."
    )


def _load_model_weights(model: Any, model_artifact_path: Path) -> None:
    if not hasattr(model, "load_state_dict"):
        raise ValueError("Configured model does not support load_state_dict.")

    payload = torch.load(model_artifact_path, map_location="cpu")
    state_dict = payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break

    if not isinstance(state_dict, dict):
        raise ValueError("Model artifact does not contain a valid state_dict.")
    model.load_state_dict(state_dict)


def _collect_sample_predictions(
    model: Any,
    data_loader: Any,
    index_to_class: dict[int, str],
    num_requested: int,
) -> list[dict[str, Any]]:
    collected = []
    seen_correct = False
    seen_incorrect = False
    stable_index = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch.get("image")
            labels = batch.get("label")
            paths = batch.get("path")
            raw_labels = batch.get("raw_label")
            if not isinstance(images, torch.Tensor):
                raise ValueError("Data loader batch image must be a torch.Tensor.")
            if not isinstance(labels, torch.Tensor):
                raise ValueError("Data loader batch label must be a torch.Tensor.")

            logits = model(images)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Model must output classification logits with shape [B, 2].")
            probabilities = torch.softmax(logits, dim=1)
            predicted_ids = torch.argmax(probabilities, dim=1)

            for batch_index in range(labels.shape[0]):
                sample = _build_sample_payload(
                    stable_index=stable_index,
                    batch_index=batch_index,
                    paths=paths,
                    raw_labels=raw_labels,
                    true_label_id=int(labels[batch_index].item()),
                    predicted_label_id=int(predicted_ids[batch_index].item()),
                    probabilities=probabilities[batch_index],
                    index_to_class=index_to_class,
                )
                stable_index += 1
                is_correct = bool(sample["correct"])

                if len(collected) < num_requested:
                    collected.append(sample)
                    seen_correct = seen_correct or is_correct
                    seen_incorrect = seen_incorrect or not is_correct
                    if len(collected) >= num_requested and seen_correct and seen_incorrect:
                        return collected
                elif is_correct and not seen_correct:
                    collected[-1] = sample
                    return collected
                elif not is_correct and not seen_incorrect:
                    collected[-1] = sample
                    return collected
                else:
                    seen_correct = seen_correct or is_correct
                    seen_incorrect = seen_incorrect or not is_correct
                    if seen_correct and seen_incorrect:
                        return collected

            if len(collected) >= num_requested and seen_correct and seen_incorrect:
                return collected

    return collected[:num_requested]


def _build_sample_payload(
    stable_index: int,
    batch_index: int,
    paths: Any,
    raw_labels: Any,
    true_label_id: int,
    predicted_label_id: int,
    probabilities: torch.Tensor,
    index_to_class: dict[int, str],
) -> dict[str, Any]:
    true_label = index_to_class.get(true_label_id)
    predicted_label = index_to_class.get(predicted_label_id)
    if true_label is None or predicted_label is None:
        raise ValueError("Predicted or true label id is missing from class mapping.")

    probability_values = [float(value) for value in probabilities.tolist()]
    confidence = float(probability_values[predicted_label_id])
    correct = predicted_label_id == true_label_id
    return {
        "sample_id": f"sample_{stable_index:06d}",
        "stable_index": stable_index,
        "image_path": _batch_value(paths, batch_index),
        "input_reference": _batch_value(paths, batch_index),
        "true_label": _batch_value(raw_labels, batch_index) or true_label,
        "true_label_id": true_label_id,
        "predicted_label": predicted_label,
        "predicted_label_id": predicted_label_id,
        "confidence": confidence,
        "probabilities": {
            index_to_class[index]: probability
            for index, probability in enumerate(probability_values)
        },
        "correct": correct,
        "error_type": None if correct else f"{true_label}_predicted_as_{predicted_label}",
    }


def _batch_value(values: Any, index: int) -> Any:
    if isinstance(values, (list, tuple)):
        return values[index]
    return None


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
