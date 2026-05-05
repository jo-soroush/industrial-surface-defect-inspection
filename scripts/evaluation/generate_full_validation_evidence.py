"""Generate governed full-validation evidence for Track A classification runs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

from inspection_ai.models.factory import create_model
from inspection_ai.training.data_loading import build_data_loaders


SUPPORTED_MODEL_TYPES = {"mlp", "cnn", "resnet18"}
MODEL_ARTIFACT_KEYS = (
    "model_artifact",
    "model",
    "model_artifact_path",
    "checkpoint",
    "checkpoint_path",
    "model_state_dict",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate full validation evidence for Track A classification runs."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Governed canonical run id for the classification model.",
    )
    parser.add_argument(
        "--run-config",
        required=True,
        help="Path to the resolved Track A run config YAML.",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the governed model checkpoint artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/metrics",
        help="Directory where full validation artifacts will be written.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=("validation",),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use for inference (auto, cpu, cuda, or a torch device string).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for evaluation batch size.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load inputs and validate the evaluation plan without writing artifacts.",
    )
    return parser


def main() -> int:
    """Run the full-validation evaluator."""
    args = build_parser().parse_args()

    run_config_path = Path(args.run_config)
    checkpoint_path = Path(args.checkpoint_path)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1.")

    config = _load_run_config(run_config_path)
    config = _apply_batch_size_override(config, args.batch_size)
    _validate_supported_config(config, run_config_path)
    _validate_checkpoint_path(checkpoint_path)

    data_loaders = build_data_loaders(config)
    validation_loader = data_loaders.get(f"{args.split}_loader")
    validation_entries = data_loaders.get(args.split)
    if not isinstance(validation_entries, list):
        raise ValueError("validation split must be a list.")
    if validation_loader is None:
        raise ValueError("validation_loader must exist for full validation.")

    expected_validation_sample_count = len(validation_entries)
    class_mapping = _load_class_mapping(config)
    index_to_class = _build_index_to_class(class_mapping)

    model = create_model(config)
    _load_model_weights(model, checkpoint_path)
    device = _resolve_device(args.device)
    model.to(device)
    model.eval()

    evaluation = _evaluate_full_validation(
        model=model,
        validation_loader=validation_loader,
        device=device,
        expected_validation_sample_count=expected_validation_sample_count,
        run_id=args.run_id,
        config=config,
        index_to_class=index_to_class,
        split=args.split,
    )

    output_dir = Path(args.output_dir)
    evaluation_path = output_dir / f"classification_full_validation_evaluation__{args.run_id}.json"
    confusion_matrix_path = (
        output_dir / f"confusion_matrix_full_validation__{args.run_id}__{args.split}.json"
    )

    if args.validate_only:
        print(f"validation_scope={evaluation['validation_scope']}")
        print(f"run_id={args.run_id}")
        print(f"evaluation_path={evaluation_path}")
        print(f"confusion_matrix_path={confusion_matrix_path}")
        print(f"total_samples={evaluation['total_samples']}")
        print(f"expected_validation_sample_count={evaluation['expected_validation_sample_count']}")
        print(f"count_match_status={evaluation['count_match_status']}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(evaluation_path, evaluation)
    _write_json(
        confusion_matrix_path,
        _build_confusion_matrix_artifact(
            evaluation=evaluation,
            source_evaluation_path=evaluation_path,
        ),
    )

    print(f"validation_scope={evaluation['validation_scope']}")
    print(f"run_id={args.run_id}")
    print(f"evaluation_path={evaluation_path}")
    print(f"confusion_matrix_path={confusion_matrix_path}")
    print(f"total_samples={evaluation['total_samples']}")
    print(f"expected_validation_sample_count={evaluation['expected_validation_sample_count']}")
    print(f"count_match_status={evaluation['count_match_status']}")
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


def _apply_batch_size_override(
    config: dict[str, Any], batch_size: int | None
) -> dict[str, Any]:
    if batch_size is None:
        return config
    updated = deepcopy(config)
    runtime = _require_dict(updated.get("training_runtime"), "training_runtime")
    runtime["batch_size"] = batch_size
    return updated


def _validate_supported_config(config: dict[str, Any], run_config_path: Path) -> None:
    identity = _require_dict(config.get("identity"), "identity")
    model_identity = _require_dict(config.get("model_identity"), "model_identity")
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    preprocessing = _require_dict(config.get("preprocessing"), "preprocessing")

    run_config_id = identity.get("run_config_id")
    if not isinstance(run_config_id, str) or not run_config_id:
        raise ValueError("identity.run_config_id must be a non-empty string.")
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {run_config_path}")
    if identity.get("task_type") != "classification":
        raise ValueError("Full validation evaluator only supports classification.")

    model_type = model_identity.get("model_type")
    if not isinstance(model_type, str) or model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "Unsupported model_identity.model_type. "
            f"Allowed values: {sorted(SUPPORTED_MODEL_TYPES)}."
        )

    for field in ("dataset_id", "dataset_version", "split_manifest_path", "class_mapping_path"):
        value = dataset_binding.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"dataset_binding.{field} must be a non-empty string.")

    preprocessing_policy_path = preprocessing.get("preprocessing_policy_path")
    if not isinstance(preprocessing_policy_path, str) or not preprocessing_policy_path:
        raise ValueError(
            "preprocessing.preprocessing_policy_path must be a non-empty string."
        )


def _validate_checkpoint_path(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _load_class_mapping(config: dict[str, Any]) -> dict[str, Any]:
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    class_mapping_path = _require_string(
        dataset_binding.get("class_mapping_path"),
        "dataset_binding.class_mapping_path",
    )
    return _load_yaml_file(Path(class_mapping_path), "class mapping")


def _build_index_to_class(class_mapping: dict[str, Any]) -> dict[int, str]:
    raw_mapping = class_mapping.get("index_to_class")
    if not isinstance(raw_mapping, dict):
        raise ValueError("Class mapping is missing index_to_class.")

    index_to_class: dict[int, str] = {}
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
        raise ValueError("Checkpoint does not contain a valid state_dict.")
    model.load_state_dict(state_dict)


def _evaluate_full_validation(
    *,
    model: Any,
    validation_loader: Any,
    device: torch.device,
    expected_validation_sample_count: int,
    run_id: str,
    config: dict[str, Any],
    index_to_class: dict[int, str],
    split: str,
) -> dict[str, Any]:
    if expected_validation_sample_count <= 0:
        raise ValueError("Expected validation sample count must be greater than 0.")

    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    total_samples = 0

    with torch.no_grad():
        for batch in validation_loader:
            if not isinstance(batch, dict):
                raise ValueError("validation_loader batch must be a dictionary.")
            images = batch.get("image")
            labels = batch.get("label")
            if not isinstance(images, torch.Tensor):
                raise ValueError("validation_loader batch image must be a torch.Tensor.")
            if not isinstance(labels, torch.Tensor):
                raise ValueError("validation_loader batch label must be a torch.Tensor.")

            images = images.to(device)
            labels = labels.to(device).reshape(-1).long()
            logits = model(images)
            if not isinstance(logits, torch.Tensor):
                raise ValueError("Model must return a torch.Tensor.")
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Validation logits must have shape [B, 2].")
            if logits.shape[0] != labels.shape[0]:
                raise ValueError("Validation logits and labels batch sizes must match.")

            predictions = torch.argmax(logits, dim=1)
            if predictions.shape != labels.shape:
                raise ValueError("Predictions and labels must have the same shape.")

            batch_size = labels.shape[0]
            total_samples += batch_size
            total_tn += int(((predictions == 0) & (labels == 0)).sum().item())
            total_fp += int(((predictions == 1) & (labels == 0)).sum().item())
            total_fn += int(((predictions == 0) & (labels == 1)).sum().item())
            total_tp += int(((predictions == 1) & (labels == 1)).sum().item())

    if total_samples == 0:
        raise ValueError("validation_loader must provide at least one sample.")

    count_match_status = (
        "match"
        if total_samples == expected_validation_sample_count
        else "mismatch"
    )
    if count_match_status != "match":
        raise ValueError(
            "Full validation sample count does not match expected validation count. "
            f"observed={total_samples} expected={expected_validation_sample_count}"
        )

    class_0 = _compute_class_metrics(
        true_positive=total_tn,
        false_positive=total_fn,
        false_negative=total_fp,
    )
    class_1 = _compute_class_metrics(
        true_positive=total_tp,
        false_positive=total_fp,
        false_negative=total_fn,
    )
    macro_metrics = {
        "precision": (class_0["precision"] + class_1["precision"]) / 2,
        "recall": (class_0["recall"] + class_1["recall"]) / 2,
        "f1": (class_0["f1"] + class_1["f1"]) / 2,
    }
    accuracy = _safe_ratio(total_tn + total_tp, total_samples)
    class_labels = [index_to_class[index] for index in sorted(index_to_class)]

    identity = _require_dict(config.get("identity"), "identity")
    model_identity = _require_dict(config.get("model_identity"), "model_identity")
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")

    return {
        "artifact_type": "classification_full_validation_evaluation",
        "validation_scope": "full",
        "generated_at": _utc_now_iso(),
        "run_id": run_id,
        "dataset_id": dataset_binding.get("dataset_id"),
        "dataset_version": dataset_binding.get("dataset_version"),
        "config_id": identity.get("run_config_id"),
        "model_name": model_identity.get("model_name"),
        "model_version": model_identity.get("model_version"),
        "split": split,
        "total_samples": total_samples,
        "expected_validation_sample_count": expected_validation_sample_count,
        "count_match_status": count_match_status,
        "accuracy": accuracy,
        "confusion_matrix": [[total_tn, total_fp], [total_fn, total_tp]],
        "per_class": {
            "class_0": class_0,
            "class_1": class_1,
        },
        "macro_metrics": macro_metrics,
        "class_labels": class_labels,
    }


def _build_confusion_matrix_artifact(
    *, evaluation: dict[str, Any], source_evaluation_path: Path
) -> dict[str, Any]:
    return {
        "artifact_type": "confusion_matrix_full_validation",
        "validation_scope": "full",
        "generated_at": evaluation["generated_at"],
        "run_id": evaluation["run_id"],
        "dataset_id": evaluation["dataset_id"],
        "dataset_version": evaluation["dataset_version"],
        "config_id": evaluation["config_id"],
        "model_name": evaluation["model_name"],
        "model_version": evaluation["model_version"],
        "split": evaluation["split"],
        "total_samples": evaluation["total_samples"],
        "expected_validation_sample_count": evaluation["expected_validation_sample_count"],
        "count_match_status": evaluation["count_match_status"],
        "matrix": evaluation["confusion_matrix"],
        "labels": evaluation["class_labels"],
        "source_evaluation_path": str(source_evaluation_path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _compute_class_metrics(
    true_positive: int, false_positive: int, false_negative: int
) -> dict[str, float]:
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2 * (precision * recall) / (precision + recall))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
