"""Generate governed Track A prediction-level error analysis for classification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from inspection_ai.models.factory import create_model  # noqa: E402
from inspection_ai.training.data_loading import build_data_loaders  # noqa: E402


ALLOWED_SPLITS = {"validation"}
ALLOWED_MODEL_TYPES = {"mlp", "cnn", "resnet18"}
CLASS_LABEL_NAMES = {0: "good", 1: "defect"}
DECISION_RULE = "argmax_softmax"
DEFAULT_THRESHOLD = 0.5
OUTPUT_DIR = REPO_ROOT / "artifacts/models/error_analysis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Track A CNN prediction-level error analysis."
    )
    parser.add_argument(
        "--training-result",
        required=True,
        help="Path to the governed TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the governed model checkpoint artifact.",
    )
    parser.add_argument(
        "--validation-evaluation",
        required=True,
        help="Path to the committed validation evaluation JSON artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where the prediction-level analysis JSON will be written.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=sorted(ALLOWED_SPLITS),
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for smoke testing only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned output path without writing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1.")

    training_result_path = Path(args.training_result)
    checkpoint_path = Path(args.checkpoint)
    validation_evaluation_path = Path(args.validation_evaluation)
    output_dir = Path(args.output_dir)

    training_result = _load_json_file(training_result_path, "TrainingResult")
    validation_evaluation = _load_json_file(
        validation_evaluation_path, "validation evaluation"
    )
    if validation_evaluation.get("artifact_type") != "classification_validation_evaluation":
        raise ValueError(
            "validation evaluation artifact_type must be classification_validation_evaluation."
        )

    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")
    artifacts = _require_dict(training_result.get("artifacts"), "artifacts")

    run_id = _require_string(identity.get("run_id"), "identity.run_id")
    run_config = _load_run_config(training_result)
    class_mapping = _load_class_mapping(run_config)
    index_to_class = _build_index_to_class(class_mapping)

    validation_loader = None
    validation_entries = None
    if not args.dry_run:
        data_loaders = build_data_loaders(run_config)
        validation_loader = data_loaders.get(f"{args.split}_loader")
        validation_entries = data_loaders.get(args.split)
        if validation_loader is None:
            raise ValueError("validation_loader must exist for error analysis.")
        if not isinstance(validation_entries, list):
            raise ValueError("validation split must be a list.")

    _validate_checkpoint_path(checkpoint_path)
    _validate_training_result(training_result, metadata, identity, artifacts, run_config)
    _validate_validation_evaluation(validation_evaluation, run_id, metadata)

    model_type = _require_string(_model_type(run_config), "model_identity.model_type")
    model_version = _require_string(_model_version(run_config), "model_identity.model_version")
    if model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported Track A model type '{model_type}'. Allowed values: {sorted(ALLOWED_MODEL_TYPES)}."
        )
    planned_output_path = output_dir / (
        f"track_a_{model_type}_{model_version}_prediction_level_analysis__{run_id}__{args.split}.json"
    )

    if args.dry_run:
        _validate_required_paths(run_config, checkpoint_path, validation_evaluation_path)
        print("track_a_prediction_level_error_analysis_dry_run=pass")
        print(f"run_id={run_id}")
        print(f"run_config_id={_run_config_id(run_config)}")
        print(f"planned_output_path={planned_output_path}")
        print(f"split={args.split}")
        print(f"validation_sample_count={metadata.get('validation_sample_count')}")
        print(f"checkpoint_path={checkpoint_path}")
        print(f"validation_evaluation_path={validation_evaluation_path}")
        return 0

    model = create_model(run_config)
    _load_model_weights(model, checkpoint_path)
    model.eval()

    analysis = _generate_prediction_level_analysis(
        model=model,
        validation_loader=validation_loader,
        validation_entries=validation_entries,
        index_to_class=index_to_class,
        run_id=run_id,
        run_config=run_config,
        model_type=model_type,
        model_version=model_version,
        checkpoint_path=checkpoint_path,
        training_result_path=training_result_path,
        validation_evaluation_path=validation_evaluation_path,
        split_manifest_path=_split_manifest_path(run_config),
        preprocessing_path=_preprocessing_path(run_config),
        class_mapping_path=_class_mapping_path(run_config),
        max_samples=args.max_samples,
    )

    if args.max_samples is None:
        _validate_confusion_matrix_matches_evaluation(
            analysis["summary"]["confusion_matrix"], validation_evaluation
        )
    else:
        analysis["partial_sample"] = True
        analysis["max_samples"] = args.max_samples

    output_dir.mkdir(parents=True, exist_ok=True)
    with planned_output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=False)

    print(f"output_path={planned_output_path}")
    print(f"artifact_type={analysis['artifact_type']}")
    print(f"run_id={analysis['run_id']}")
    print(f"total_samples={analysis['total_samples']}")
    print(f"partial_sample={analysis['partial_sample']}")
    if args.max_samples is not None:
        print(f"max_samples={args.max_samples}")
    print(f"confusion_matrix={analysis['summary']['confusion_matrix']}")
    print(f"precision={analysis['summary']['precision']}")
    print(f"recall={analysis['summary']['recall']}")
    print(f"f1={analysis['summary']['f1']}")
    print(f"macro_f1={analysis['summary']['macro_f1']}")
    return 0


def _generate_prediction_level_analysis(
    *,
    model: Any,
    validation_loader: Any,
    validation_entries: list[dict[str, Any]] | None,
    index_to_class: dict[int, str],
    run_id: str,
    run_config: dict[str, Any],
    model_type: str,
    model_version: str,
    checkpoint_path: Path,
    training_result_path: Path,
    validation_evaluation_path: Path,
    split_manifest_path: str,
    preprocessing_path: str,
    class_mapping_path: str,
    max_samples: int | None,
) -> dict[str, Any]:
    if validation_loader is None:
        raise ValueError("validation_loader must be available for analysis.")

    records: list[dict[str, Any]] = []
    total_tn = total_fp = total_fn = total_tp = 0
    sample_index = 0

    with torch.no_grad():
        for batch in validation_loader:
            if not isinstance(batch, dict):
                raise ValueError("validation_loader batch must be a dictionary.")

            images = batch.get("image")
            labels = batch.get("label")
            paths = batch.get("path")
            raw_labels = batch.get("raw_label")
            if not isinstance(images, torch.Tensor):
                raise ValueError("validation_loader batch image must be a torch.Tensor.")
            if not isinstance(labels, torch.Tensor):
                raise ValueError("validation_loader batch label must be a torch.Tensor.")

            logits = model(images)
            if not isinstance(logits, torch.Tensor):
                raise ValueError("Model must return a torch.Tensor.")
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Validation logits must have shape [B, 2].")
            if logits.shape[0] != labels.shape[0]:
                raise ValueError("Validation logits and labels batch sizes must match.")

            probabilities = torch.softmax(logits, dim=1)
            predicted_ids = torch.argmax(probabilities, dim=1)
            labels = labels.reshape(-1).long()

            for batch_index in range(labels.shape[0]):
                true_label_id = int(labels[batch_index].item())
                predicted_label_id = int(predicted_ids[batch_index].item())
                if true_label_id not in (0, 1):
                    raise ValueError("Validation labels must be binary class ids 0 or 1.")
                if predicted_label_id not in (0, 1):
                    raise ValueError("Predicted labels must be binary class ids 0 or 1.")

                probability_values = [float(value) for value in probabilities[batch_index].tolist()]
                probability_good = float(probability_values[0])
                probability_defect = float(probability_values[1])
                confidence = max(probability_good, probability_defect)
                true_label_name = index_to_class[true_label_id]
                predicted_label_name = index_to_class[predicted_label_id]
                error_type = _error_type(true_label_id, predicted_label_id)
                image_path = _batch_value(paths, batch_index)
                if image_path is None:
                    if validation_entries is None or sample_index >= len(validation_entries):
                        raise ValueError("Unable to resolve validation image_path for record.")
                    image_path = validation_entries[sample_index].get("path")

                records.append(
                    {
                        "sample_index": sample_index,
                        "image_path": image_path,
                        "run_id": run_id,
                        "model_type": model_type,
                        "model_version": model_version,
                        "true_label": true_label_id,
                        "true_label_name": true_label_name,
                        "predicted_label": predicted_label_id,
                        "predicted_label_name": predicted_label_name,
                        "probability_good": probability_good,
                        "probability_defect": probability_defect,
                        "confidence": confidence,
                        "error_type": error_type,
                    }
                )

                if true_label_id == 0 and predicted_label_id == 0:
                    total_tn += 1
                elif true_label_id == 0 and predicted_label_id == 1:
                    total_fp += 1
                elif true_label_id == 1 and predicted_label_id == 0:
                    total_fn += 1
                else:
                    total_tp += 1

                sample_index += 1
                if max_samples is not None and len(records) >= max_samples:
                    break
            if max_samples is not None and len(records) >= max_samples:
                break

    if not records:
        raise ValueError("validation_loader must provide at least one sample.")

    total_samples = len(records)
    partial_sample = max_samples is not None
    if not partial_sample:
        expected_count = _expected_validation_sample_count(run_config)
        if total_samples != expected_count:
            raise ValueError(
                "Prediction-level records must cover the full validation split. "
                f"observed={total_samples} expected={expected_count}"
            )

    confusion_matrix = [[total_tn, total_fp], [total_fn, total_tp]]
    summary = {
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "true_negative": total_tn,
        "precision": _safe_ratio(total_tp, total_tp + total_fp),
        "recall": _safe_ratio(total_tp, total_tp + total_fn),
        "f1": _f1(total_tp, total_fp, total_fn),
        "macro_f1": _macro_f1(total_tn, total_fp, total_fn, total_tp),
        "confusion_matrix": confusion_matrix,
    }

    return {
        "artifact_type": "track_a_prediction_level_error_analysis",
        "run_id": run_id,
        "run_config_id": _run_config_id(run_config),
        "model_type": model_type,
        "model_version": model_version,
        "dataset_id": run_config.get("dataset_binding", {}).get("dataset_id"),
        "split": "validation",
        "decision_rule": DECISION_RULE,
        "threshold_used": DEFAULT_THRESHOLD,
        "total_samples": total_samples,
        "partial_sample": partial_sample,
        "summary": summary,
        "records": records,
        "source_artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "training_result_path": str(training_result_path),
            "validation_evaluation_path": str(validation_evaluation_path),
            "run_config_path": _run_config_path(run_config),
            "split_manifest_path": split_manifest_path,
            "preprocessing_path": preprocessing_path,
            "class_mapping_path": class_mapping_path,
        },
        "created_at": _utc_now_iso(),
    }


def _validate_confusion_matrix_matches_evaluation(
    confusion_matrix: list[list[int]], validation_evaluation: dict[str, Any]
) -> None:
    expected = validation_evaluation.get("confusion_matrix")
    if confusion_matrix != expected:
        print(f"generated_confusion_matrix={confusion_matrix}")
        print(f"expected_confusion_matrix={expected}")
        raise ValueError(
            "Generated prediction-level confusion matrix does not match the existing validation evaluation."
        )


def _validate_checkpoint_path(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")


def _validate_required_paths(
    run_config: dict[str, Any],
    checkpoint_path: Path,
    validation_evaluation_path: Path,
) -> None:
    for path in (
        checkpoint_path,
        validation_evaluation_path,
        Path(_run_config_path(run_config)),
        Path(_split_manifest_path(run_config)),
        Path(_preprocessing_path(run_config)),
        Path(_class_mapping_path(run_config)),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")


def _validate_training_result(
    training_result: dict[str, Any],
    metadata: dict[str, Any],
    identity: dict[str, Any],
    artifacts: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    if identity.get("task_type") != "classification":
        raise ValueError("TrainingResult task_type must be classification.")
    model_type = identity.get("model_type")
    if model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"TrainingResult model_type must be one of {sorted(ALLOWED_MODEL_TYPES)}."
        )
    if not isinstance(identity.get("run_config_id"), str) or not identity.get("run_config_id"):
        raise ValueError("TrainingResult run_config_id must be a non-empty string.")
    if not isinstance(metadata.get("model_version"), str) or not metadata.get("model_version"):
        raise ValueError("TrainingResult model_version must be a non-empty string.")
    if metadata.get("training_mode") != "full_epoch":
        raise ValueError("TrainingResult training_mode must be full_epoch.")
    if metadata.get("full_epoch_training") is not True:
        raise ValueError("TrainingResult full_epoch_training must be true.")
    if metadata.get("validation_sample_count") != 803:
        raise ValueError("TrainingResult validation_sample_count must be 803.")
    if metadata.get("train_sample_count") != 3748:
        raise ValueError("TrainingResult train_sample_count must be 3748.")
    model_artifact = artifacts.get("model_artifact")
    if not isinstance(model_artifact, dict) or not model_artifact.get("path"):
        raise ValueError("TrainingResult must reference a model_artifact checkpoint.")
    if run_config.get("identity", {}).get("run_config_id") != identity.get("run_config_id"):
        raise ValueError("Run config does not match TrainingResult run_config_id.")


def _validate_validation_evaluation(
    validation_evaluation: dict[str, Any],
    run_id: str,
    metadata: dict[str, Any],
) -> None:
    if validation_evaluation.get("artifact_type") != "classification_validation_evaluation":
        raise ValueError("validation evaluation artifact_type mismatch.")
    if validation_evaluation.get("run_id") != run_id:
        raise ValueError("validation evaluation run_id mismatch.")
    if validation_evaluation.get("dataset_id") != metadata.get("dataset_id"):
        raise ValueError("validation evaluation dataset_id mismatch.")
    if validation_evaluation.get("total_samples") != 803:
        raise ValueError("validation evaluation total_samples must be 803.")


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
        if (
            isinstance(identity_section, dict)
            and identity_section.get("run_config_id") == run_config_id
        ):
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
    if (
        expected_split_manifest
        and dataset_binding.get("split_manifest_path") != expected_split_manifest
    ):
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


def _error_type(true_label: int, predicted_label: int) -> str:
    if true_label == 0 and predicted_label == 0:
        return "true_negative"
    if true_label == 0 and predicted_label == 1:
        return "false_positive"
    if true_label == 1 and predicted_label == 0:
        return "false_negative"
    if true_label == 1 and predicted_label == 1:
        return "true_positive"
    raise ValueError("Unexpected binary label combination.")


def _expected_validation_sample_count(run_config: dict[str, Any]) -> int:
    dataset_binding = _require_dict(run_config.get("dataset_binding"), "dataset_binding")
    split_manifest_path = _require_string(
        dataset_binding.get("split_manifest_path"),
        "dataset_binding.split_manifest_path",
    )
    manifest = _load_yaml_file(Path(split_manifest_path), "split manifest")
    validation_entries = manifest.get("validation_entries")
    if not isinstance(validation_entries, list):
        raise ValueError("split manifest validation_entries must be a list.")
    return len(validation_entries)


def _batch_value(values: Any, index: int) -> Any:
    if isinstance(values, (list, tuple)):
        return values[index]
    return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    if precision + recall == 0.0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def _macro_f1(
    true_negative: int, false_positive: int, false_negative: int, true_positive: int
) -> float:
    class_0_precision = _safe_ratio(true_negative, true_negative + false_negative)
    class_0_recall = _safe_ratio(true_negative, true_negative + false_positive)
    class_0_f1 = 0.0
    if class_0_precision + class_0_recall > 0.0:
        class_0_f1 = float(
            2 * (class_0_precision * class_0_recall) / (class_0_precision + class_0_recall)
        )

    class_1_f1 = _f1(true_positive, false_positive, false_negative)
    return float((class_0_f1 + class_1_f1) / 2)


def _run_config_id(config: dict[str, Any]) -> str:
    identity = config.get("identity")
    if not isinstance(identity, dict):
        return "unknown_config"
    run_config_id = identity.get("run_config_id")
    if not isinstance(run_config_id, str) or not run_config_id:
        return "unknown_config"
    return run_config_id


def _run_config_path(config: dict[str, Any]) -> str:
    return f"configs/runs/{_run_config_id(config)}.yaml"


def _model_type(config: dict[str, Any]) -> Any:
    model_identity = config.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError("Run config is missing model_identity.")
    return model_identity.get("model_type")


def _model_version(config: dict[str, Any]) -> Any:
    model_identity = config.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError("Run config is missing model_identity.")
    return model_identity.get("model_version")


def _split_manifest_path(config: dict[str, Any]) -> str:
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    return _require_string(dataset_binding.get("split_manifest_path"), "split_manifest_path")


def _preprocessing_path(config: dict[str, Any]) -> str:
    preprocessing = _require_dict(config.get("preprocessing"), "preprocessing")
    return _require_string(
        preprocessing.get("preprocessing_policy_path"),
        "preprocessing.preprocessing_policy_path",
    )


def _class_mapping_path(config: dict[str, Any]) -> str:
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    return _require_string(
        dataset_binding.get("class_mapping_path"),
        "dataset_binding.class_mapping_path",
    )


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
