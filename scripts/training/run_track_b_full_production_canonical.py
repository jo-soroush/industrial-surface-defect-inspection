"""Run full production-canonical Track B anomaly training and evaluation.

This script executes only the implemented Track B anomaly path:
MVTec anomaly data, AutoencoderModel training, reconstruction-error
evaluation, governed artifact inventory, and read-only validation.
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

import numpy as np
from PIL import Image
import torch
import yaml

from inspection_ai.evaluation.anomaly_evaluation import (
    compute_anomaly_metrics,
    compute_reconstruction_scores,
    compute_threshold,
    generate_predictions,
    run_anomaly_inference,
)
from inspection_ai.models.factory import create_model
from inspection_ai.training.checkpointing import resolve_model_checkpoint_path
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
DEFAULT_RUN_ID = "b8ca43f5-0d53-4a42-ab37-b5fca9544a36"
QUALITATIVE_SAMPLE_LIMIT = 24
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full production-canonical Track B anomaly training."
    )
    parser.add_argument(
        "--run-config",
        default="configs/runs/autoencoder_train_v0_1_0.yaml",
        help="Governed Track B autoencoder run config.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Production-canonical run_id to stamp onto generated artifacts.",
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
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing artifacts for the supplied run_id.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = _require_string(args.run_id, "run_id")
    artifacts_root = Path(args.artifacts_root)

    config = _production_canonical_config(_load_run_config(Path(args.run_config)))
    _validate_supported_config(config)
    _set_reproducibility(config)

    data_loaders = build_data_loaders(config)
    _validate_loader_contract(data_loaders)

    output_paths = _output_paths(artifacts_root, run_id)
    _validate_output_policy(output_paths, allow_overwrite=args.allow_overwrite)

    model = create_model(config)
    result = run_training_loop(config=config, model=model, data_loader=data_loaders)
    result.identity["run_id"] = run_id
    _stamp_production_metadata(result, config, data_loaders)

    checkpoint_path = output_paths["model_checkpoint"]
    _save_checkpoint(model.state_dict(), checkpoint_path)
    result.add_artifact(
        "model_artifact",
        {"path": str(checkpoint_path), "type": MODEL_ARTIFACT_TYPE},
    )
    validate_training_result(result)

    training_result_path = persist_training_result(
        result=result,
        output_dir=artifacts_root / "analysis" / "training_results",
    )
    output_paths["training_result"] = training_result_path

    evaluation_context = _run_evaluation(
        model=model,
        data_loaders=data_loaders,
        threshold_strategy=args.threshold_strategy,
    )
    evaluation_path = _write_anomaly_evaluation(
        run_id=run_id,
        config=config,
        training_result_path=training_result_path,
        checkpoint_path=checkpoint_path,
        evaluation_context=evaluation_context,
        output_path=output_paths["anomaly_evaluation"],
    )
    learning_curves_path = _write_learning_curves(
        run_id=run_id,
        result_payload=result.to_dict(),
        output_path=output_paths["learning_curves"],
    )
    confusion_matrix_path = _write_confusion_matrix(
        run_id=run_id,
        evaluation_context=evaluation_context,
        output_path=output_paths["confusion_matrix"],
    )
    qualitative_path, explainability_path = _write_qualitative_and_explainability(
        run_id=run_id,
        model=model,
        data_loaders=data_loaders,
        evaluation_context=evaluation_context,
        output_dir=artifacts_root / "explainability" / run_id,
    )
    output_paths["learning_curves"] = learning_curves_path
    output_paths["confusion_matrix"] = confusion_matrix_path
    output_paths["qualitative_samples"] = qualitative_path
    output_paths["explainability"] = explainability_path

    inventory_path = _write_inventory(
        run_id=run_id,
        config=config,
        result_payload=result.to_dict(),
        evaluation_context=evaluation_context,
        output_paths=output_paths,
        output_path=output_paths["inventory"],
    )
    output_paths["inventory"] = inventory_path

    validation = _validate_artifacts_read_only(
        output_paths=output_paths,
        run_id=run_id,
    )
    summary_path = _write_summary(
        run_id=run_id,
        config=config,
        evaluation_context=evaluation_context,
        validation=validation,
        output_paths=output_paths,
        output_path=output_paths["summary"],
    )
    output_paths["summary"] = summary_path

    report = {
        "run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "validation_result": validation["status"],
        "validated_artifacts": validation["validated_artifacts"],
        "warnings": validation["warnings"],
        "metrics": evaluation_context["metrics"],
        "paths": {name: str(path) for name, path in output_paths.items()},
    }
    print(json.dumps(report, indent=2))
    return 0 if validation["status"] == "pass" else 1


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


def _validate_supported_config(config: dict[str, Any]) -> None:
    identity = _require_dict(config.get("identity"), "identity")
    model_identity = _require_dict(config.get("model_identity"), "model_identity")
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    if identity.get("task_type") != TASK_TYPE:
        raise ValueError("Stage 3 Track B script only supports anomaly_detection.")
    if identity.get("track_id") != TRACK_ID:
        raise ValueError("Production Track B config identity.track_id must be track_b.")
    if identity.get("is_experiment") is not False:
        raise ValueError("Production Track B runs require identity.is_experiment=false.")
    if model_identity.get("model_type") != MODEL_TYPE:
        raise ValueError("Stage 3 Track B script only supports autoencoder.")
    if dataset_binding.get("dataset_id") != DATASET_ID:
        raise ValueError("Stage 3 Track B dataset_id must be mvtec_anomaly.")
    split_manifest_path = Path(
        _require_string(
            dataset_binding.get("split_manifest_path"),
            "dataset_binding.split_manifest_path",
        )
    )
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")


def _set_reproducibility(config: dict[str, Any]) -> None:
    runtime = _require_dict(config.get("training_runtime"), "training_runtime")
    seed = runtime.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training_runtime.seed must be an integer.")
    random.seed(seed)
    np.random.seed(seed)
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


def _output_paths(artifacts_root: Path, run_id: str) -> dict[str, Path]:
    return {
        "training_result": artifacts_root
        / "analysis"
        / "training_results"
        / f"training_result__{run_id}.json",
        "model_checkpoint": resolve_model_checkpoint_path(run_id),
        "anomaly_evaluation": artifacts_root
        / "metrics"
        / f"anomaly_detection_evaluation__{run_id}__test.json",
        "learning_curves": artifacts_root
        / "metrics"
        / f"anomaly_learning_curves__{run_id}.json",
        "confusion_matrix": artifacts_root
        / "metrics"
        / f"anomaly_confusion_matrix__{run_id}__train_test.json",
        "qualitative_samples": artifacts_root
        / "explainability"
        / run_id
        / f"anomaly_qualitative_samples__{run_id}.json",
        "explainability": artifacts_root
        / "explainability"
        / run_id
        / f"anomaly_reconstruction_explainability__{run_id}.json",
        "inventory": artifacts_root
        / "inventory"
        / f"track_b_artifact_inventory__{run_id}.json",
        "summary": artifacts_root
        / "metadata"
        / f"track_b_full_production_canonical_summary__{run_id}.json",
    }


def _validate_output_policy(
    output_paths: dict[str, Path], allow_overwrite: bool
) -> None:
    if allow_overwrite:
        return
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing Stage 3 artifacts. "
            "Pass --allow-overwrite only when intentionally regenerating the "
            f"same run_id. Existing paths: {existing}"
        )


def _stamp_production_metadata(
    result: Any,
    config: dict[str, Any],
    data_loaders: dict[str, Any],
) -> None:
    runtime = _require_dict(config.get("training_runtime"), "training_runtime")
    dataset_binding = _require_dict(config.get("dataset_binding"), "dataset_binding")
    result.add_metadata("canonical_status", CANONICAL_STATUS)
    result.add_metadata("canonical_track_id", TRACK_ID)
    result.add_metadata("canonical_policy", "production_canonical_track_b_full")
    result.add_metadata("seed", runtime.get("seed"))
    result.add_metadata("dataset_version", dataset_binding.get("dataset_version"))
    result.add_metadata("split_manifest_path", dataset_binding.get("split_manifest_path"))
    result.add_metadata("train_sample_count", len(data_loaders["train"]))
    result.add_metadata("validation_sample_count", len(data_loaders["validation"]))
    result.add_metadata("test_sample_count", len(data_loaders["test"]))


def _save_checkpoint(state_dict: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def _run_evaluation(
    model: Any,
    data_loaders: dict[str, Any],
    threshold_strategy: str,
) -> dict[str, Any]:
    model.eval()
    train_inference = run_anomaly_inference(model, data_loaders["train_loader"])
    threshold = compute_threshold(
        train_inference["scores"], {"threshold_strategy": threshold_strategy}
    )
    test_inference = run_anomaly_inference(model, data_loaders["test_loader"])
    train_predictions = generate_predictions(train_inference["scores"], threshold)
    test_predictions = generate_predictions(test_inference["scores"], threshold)
    test_labels = _require_binary_label_mix(test_inference["labels"], "test")
    train_labels = [_require_binary_int(value, "train label") for value in train_inference["labels"]]
    metrics = compute_anomaly_metrics(
        labels=test_labels,
        scores=test_inference["scores"],
        predictions=test_predictions,
    )
    test_entries = _require_entries(data_loaders, "test")
    samples = _build_samples(test_entries, test_inference, test_predictions)
    counts = _build_counts(
        train_scores=train_inference["scores"],
        labels=test_labels,
        predictions=test_predictions,
        samples=samples,
    )
    return {
        "threshold_strategy": threshold_strategy,
        "threshold": float(threshold),
        "train_inference": train_inference,
        "test_inference": test_inference,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "metrics": metrics,
        "samples": samples,
        "counts": counts,
    }


def _write_anomaly_evaluation(
    run_id: str,
    config: dict[str, Any],
    training_result_path: Path,
    checkpoint_path: Path,
    evaluation_context: dict[str, Any],
    output_path: Path,
) -> Path:
    payload = {
        "artifact_type": "anomaly_detection_evaluation",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "model_id": MODEL_TYPE,
        "model_type": MODEL_TYPE,
        "dataset_id": DATASET_ID,
        "config_id": _config_id(config),
        "source_training_result": str(training_result_path),
        "source_model_checkpoint": str(checkpoint_path),
        "split_manifest_path": _split_manifest_path(config),
        "preprocessing_policy_path": _preprocessing_policy_path(config),
        "created_at": _utc_now_iso(),
        "score_definition": "mean_squared_reconstruction_error_per_image",
        "threshold_strategy": evaluation_context["threshold_strategy"],
        "threshold": evaluation_context["threshold"],
        "metrics": evaluation_context["metrics"],
        "train_score_summary": _score_summary(
            evaluation_context["train_inference"]["scores"]
        ),
        "test_score_summary": _score_summary(
            evaluation_context["test_inference"]["scores"]
        ),
        "counts": evaluation_context["counts"],
        "samples": evaluation_context["samples"],
    }
    _validate_evaluation_payload(payload)
    _write_json(output_path, payload)
    return output_path


def _write_learning_curves(
    run_id: str,
    result_payload: dict[str, Any],
    output_path: Path,
) -> Path:
    curves = _require_dict(result_payload.get("learning_curves"), "learning_curves")
    payload = {
        "artifact_type": "anomaly_learning_curves",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "created_at": _utc_now_iso(),
        "curves": {
            "train_loss": _require_numeric_list(curves.get("train_loss"), "train_loss"),
            "val_loss": _require_numeric_list(curves.get("val_loss"), "val_loss"),
        },
    }
    _write_json(output_path, payload)
    return output_path


def _write_confusion_matrix(
    run_id: str,
    evaluation_context: dict[str, Any],
    output_path: Path,
) -> Path:
    payload = {
        "artifact_type": "anomaly_confusion_matrix",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "created_at": _utc_now_iso(),
        "threshold": evaluation_context["threshold"],
        "splits": {
            "train": _confusion_matrix_payload(
                evaluation_context["train_labels"],
                evaluation_context["train_predictions"],
            ),
            "test": _confusion_matrix_payload(
                evaluation_context["test_labels"],
                evaluation_context["test_predictions"],
            ),
        },
    }
    _write_json(output_path, payload)
    return output_path


def _write_qualitative_and_explainability(
    run_id: str,
    model: Any,
    data_loaders: dict[str, Any],
    evaluation_context: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = _select_qualitative_indices(evaluation_context["samples"])
    qualitative_items: list[dict[str, Any]] = []
    explainability_items: list[dict[str, Any]] = []
    selected = set(sample_indices)
    test_loader = data_loaders["test_loader"]
    model_device = next(model.parameters()).device
    seen = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(model_device)
            reconstructions = model(images)
            batch_scores = compute_reconstruction_scores(images, reconstructions)
            batch_size = int(images.shape[0])
            for batch_index in range(batch_size):
                sample_index = seen + batch_index
                if sample_index not in selected:
                    continue
                sample = evaluation_context["samples"][sample_index]
                image = images[batch_index].detach().cpu()
                reconstruction = reconstructions[batch_index].detach().cpu()
                error_map = torch.mean((image - reconstruction) ** 2, dim=0)

                input_path = output_dir / f"sample_{sample_index:05d}__input.png"
                reconstruction_path = (
                    output_dir / f"sample_{sample_index:05d}__reconstruction.png"
                )
                heatmap_path = output_dir / f"sample_{sample_index:05d}__heatmap.png"
                overlay_path = output_dir / f"sample_{sample_index:05d}__overlay.png"

                input_image = _tensor_to_image(image)
                reconstruction_image = _tensor_to_image(reconstruction)
                heatmap_image = _error_map_to_heatmap(error_map)
                overlay_image = Image.blend(input_image, heatmap_image, alpha=0.45)

                input_image.save(input_path)
                reconstruction_image.save(reconstruction_path)
                heatmap_image.save(heatmap_path)
                overlay_image.save(overlay_path)

                channel_errors = torch.mean(
                    (image - reconstruction) ** 2, dim=(1, 2)
                ).detach().cpu().tolist()
                feature_importance = {
                    "red_channel_reconstruction_error": float(channel_errors[0]),
                    "green_channel_reconstruction_error": float(channel_errors[1]),
                    "blue_channel_reconstruction_error": float(channel_errors[2]),
                }
                item = {
                    "sample_id": sample_index,
                    "image_path": sample["image_path"],
                    "true_label": sample["true_label"],
                    "predicted_label": sample["predicted_label"],
                    "correct": sample["correct"],
                    "anomaly_score": float(batch_scores[batch_index]),
                    "input_path": str(input_path),
                    "reconstruction_path": str(reconstruction_path),
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
                }
                qualitative_items.append(item)
                explainability_items.append(
                    {
                        **item,
                        "method": "reconstruction_error_heatmap",
                        "score_definition": "per_pixel_mean_squared_reconstruction_error",
                        "feature_importance": feature_importance,
                    }
                )
            seen += batch_size

    qualitative_payload = {
        "artifact_type": "anomaly_qualitative_samples",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "created_at": _utc_now_iso(),
        "sample_count": len(qualitative_items),
        "samples": qualitative_items,
    }
    explainability_payload = {
        "artifact_type": "anomaly_reconstruction_explainability",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "created_at": _utc_now_iso(),
        "method": "reconstruction_error_heatmap",
        "sample_count": len(explainability_items),
        "heatmaps": explainability_items,
    }
    if not qualitative_items:
        raise ValueError("No qualitative samples were generated.")

    qualitative_path = output_dir / f"anomaly_qualitative_samples__{run_id}.json"
    explainability_path = (
        output_dir / f"anomaly_reconstruction_explainability__{run_id}.json"
    )
    _write_json(qualitative_path, qualitative_payload)
    _write_json(explainability_path, explainability_payload)
    return qualitative_path, explainability_path


def _write_inventory(
    run_id: str,
    config: dict[str, Any],
    result_payload: dict[str, Any],
    evaluation_context: dict[str, Any],
    output_paths: dict[str, Path],
    output_path: Path,
) -> Path:
    metadata = _require_dict(result_payload.get("metadata"), "metadata")
    artifact_names = (
        "training_result",
        "model_checkpoint",
        "anomaly_evaluation",
        "learning_curves",
        "confusion_matrix",
        "qualitative_samples",
        "explainability",
    )
    inventory = {
        "artifact_type": "track_b_artifact_inventory",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "model_id": MODEL_TYPE,
        "model_type": MODEL_TYPE,
        "dataset_id": DATASET_ID,
        "config_id": _config_id(config),
        "created_at": _utc_now_iso(),
        "counts": {
            "train_sample_count": metadata.get("train_sample_count"),
            "validation_sample_count": metadata.get("validation_sample_count"),
            "test_sample_count": metadata.get("test_sample_count"),
            **evaluation_context["counts"],
        },
        "linkage": {
            "training_result": str(output_paths["training_result"]),
            "model_checkpoint": str(output_paths["model_checkpoint"]),
            "anomaly_evaluation": str(output_paths["anomaly_evaluation"]),
            "learning_curves": str(output_paths["learning_curves"]),
            "confusion_matrix": str(output_paths["confusion_matrix"]),
            "qualitative_samples": str(output_paths["qualitative_samples"]),
            "explainability": str(output_paths["explainability"]),
            "evaluation_source_training_result": str(output_paths["training_result"]),
            "evaluation_source_model_checkpoint": str(output_paths["model_checkpoint"]),
        },
        "artifacts": {
            name: _artifact_entry(output_paths[name], _artifact_type_for(name))
            for name in artifact_names
        },
    }
    _validate_inventory_payload(inventory)
    _write_json(output_path, inventory)
    return output_path


def _validate_artifacts_read_only(
    output_paths: dict[str, Path],
    run_id: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    validated_artifacts: list[str] = []
    hashes: dict[str, str] = {}
    required = (
        "training_result",
        "model_checkpoint",
        "anomaly_evaluation",
        "learning_curves",
        "confusion_matrix",
        "qualitative_samples",
        "explainability",
        "inventory",
    )
    try:
        for name in required:
            path = output_paths[name]
            if not path.is_file():
                raise FileNotFoundError(f"{name} missing: {path}")
            hashes[name] = _sha256(path)
            validated_artifacts.append(str(path))
        training_result = _load_json_file(output_paths["training_result"], "TrainingResult")
        evaluation = _load_json_file(output_paths["anomaly_evaluation"], "anomaly evaluation")
        inventory = _load_json_file(output_paths["inventory"], "Track B inventory")
        _validate_training_result_payload(training_result, run_id)
        _validate_evaluation_payload(evaluation)
        _validate_inventory_payload(inventory)
        _validate_inventory_hashes(inventory)
        return {
            "status": "pass",
            "validated_artifacts": validated_artifacts,
            "warnings": warnings,
            "hashes": hashes,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "validated_artifacts": validated_artifacts,
            "warnings": warnings,
            "hashes": hashes,
            "error": str(exc),
        }


def _write_summary(
    run_id: str,
    config: dict[str, Any],
    evaluation_context: dict[str, Any],
    validation: dict[str, Any],
    output_paths: dict[str, Path],
    output_path: Path,
) -> Path:
    artifact_names = (
        "training_result",
        "model_checkpoint",
        "anomaly_evaluation",
        "learning_curves",
        "confusion_matrix",
        "qualitative_samples",
        "explainability",
        "inventory",
    )
    summary = {
        "artifact_type": "track_b_full_production_canonical_summary",
        "task_type": TASK_TYPE,
        "track_id": TRACK_ID,
        "run_id": run_id,
        "canonical_status": CANONICAL_STATUS,
        "created_at": _utc_now_iso(),
        "config_id": _config_id(config),
        "dataset_id": DATASET_ID,
        "model_type": MODEL_TYPE,
        "seed": _require_dict(config.get("training_runtime"), "training_runtime").get(
            "seed"
        ),
        "metrics": evaluation_context["metrics"],
        "paths": {name: str(output_paths[name]) for name in artifact_names},
        "artifacts": {name: _summary_artifact(output_paths[name]) for name in artifact_names},
        "validation": validation,
    }
    _write_json(output_path, summary)
    return output_path


def _validate_training_result_payload(payload: dict[str, Any], run_id: str) -> None:
    identity = _require_dict(payload.get("identity"), "identity")
    metadata = _require_dict(payload.get("metadata"), "metadata")
    artifacts = _require_dict(payload.get("artifacts"), "artifacts")
    metrics = _require_dict(payload.get("metrics"), "metrics")
    curves = _require_dict(payload.get("learning_curves"), "learning_curves")
    if identity.get("run_id") != run_id:
        raise ValueError("TrainingResult run_id mismatch.")
    if identity.get("task_type") != TASK_TYPE:
        raise ValueError("TrainingResult task_type mismatch.")
    if identity.get("model_type") != MODEL_TYPE:
        raise ValueError("TrainingResult model_type mismatch.")
    if identity.get("is_experiment") is not False:
        raise ValueError("TrainingResult is_experiment must be false.")
    if metadata.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("TrainingResult canonical_status mismatch.")
    if metadata.get("dataset_id") != DATASET_ID:
        raise ValueError("TrainingResult dataset_id mismatch.")
    _require_numeric(metrics.get("reconstruction_loss"), "reconstruction_loss")
    _require_numeric_list(curves.get("train_loss"), "learning_curves.train_loss")
    _require_numeric_list(curves.get("val_loss"), "learning_curves.val_loss")
    model_artifact = _require_dict(artifacts.get("model_artifact"), "model_artifact")
    if model_artifact.get("type") != MODEL_ARTIFACT_TYPE:
        raise ValueError("model_artifact.type must be pytorch_state_dict.")
    if not Path(_require_string(model_artifact.get("path"), "model_artifact.path")).is_file():
        raise FileNotFoundError("TrainingResult checkpoint link is missing.")


def _validate_evaluation_payload(payload: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "anomaly_detection_evaluation":
        raise ValueError("Evaluation artifact_type mismatch.")
    if payload.get("task_type") != TASK_TYPE:
        raise ValueError("Evaluation task_type mismatch.")
    if payload.get("track_id") != TRACK_ID:
        raise ValueError("Evaluation track_id mismatch.")
    _require_numeric(payload.get("threshold"), "threshold")
    metrics = _require_dict(payload.get("metrics"), "metrics")
    for name in ("roc_auc", "precision", "recall", "f1"):
        _require_numeric(metrics.get(name), f"metrics.{name}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Evaluation samples must be a non-empty list.")


def _validate_inventory_payload(payload: dict[str, Any]) -> None:
    if payload.get("artifact_type") != "track_b_artifact_inventory":
        raise ValueError("Inventory artifact_type mismatch.")
    if payload.get("canonical_status") != CANONICAL_STATUS:
        raise ValueError("Inventory canonical_status mismatch.")
    artifacts = _require_dict(payload.get("artifacts"), "artifacts")
    for name, entry in artifacts.items():
        artifact = _require_dict(entry, f"artifacts.{name}")
        path = Path(_require_string(artifact.get("path"), f"artifacts.{name}.path"))
        if not path.is_file():
            raise FileNotFoundError(f"Inventory artifact missing: {path}")
        if artifact.get("exists") is not True:
            raise ValueError(f"Inventory artifact {name}.exists must be true.")
        _require_string(artifact.get("sha256"), f"artifacts.{name}.sha256")


def _validate_inventory_hashes(inventory: dict[str, Any]) -> None:
    artifacts = _require_dict(inventory.get("artifacts"), "inventory.artifacts")
    for name, entry in artifacts.items():
        artifact = _require_dict(entry, f"inventory.artifacts.{name}")
        path = Path(_require_string(artifact.get("path"), f"{name}.path"))
        expected = _require_string(artifact.get("sha256"), f"{name}.sha256")
        actual = _sha256(path)
        if actual != expected:
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
    if not (
        len(scores)
        == len(labels)
        == len(paths)
        == len(mask_paths)
        == len(predictions)
        == len(test_entries)
    ):
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
                "anomaly_score": float(
                    _require_numeric(scores[index], f"scores[{index}]")
                ),
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


def _confusion_matrix_payload(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("Confusion matrix labels and predictions counts must match.")
    tn = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    return {
        "labels": ["normal", "anomaly"],
        "matrix": [[tn, fp], [fn, tp]],
        "counts": {
            "true_normal_predicted_normal": tn,
            "true_normal_predicted_anomaly": fp,
            "true_anomaly_predicted_normal": fn,
            "true_anomaly_predicted_anomaly": tp,
        },
    }


def _select_qualitative_indices(samples: list[dict[str, Any]]) -> list[int]:
    correct = [sample for sample in samples if sample["correct"]]
    incorrect = [sample for sample in samples if not sample["correct"]]
    normal = [sample for sample in samples if sample["true_label_id"] == 0]
    anomaly = [sample for sample in samples if sample["true_label_id"] == 1]
    ranked_high = sorted(samples, key=lambda item: item["anomaly_score"], reverse=True)
    selected: list[int] = []
    for group in (incorrect, correct, anomaly, normal, ranked_high):
        for sample in group:
            sample_id = int(sample["sample_id"])
            if sample_id not in selected:
                selected.append(sample_id)
            if len(selected) >= QUALITATIVE_SAMPLE_LIMIT:
                return selected
    return selected


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = (tensor.detach().cpu() * IMAGENET_STD) + IMAGENET_MEAN
    image = image.clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _error_map_to_heatmap(error_map: torch.Tensor) -> Image.Image:
    values = error_map.detach().cpu().numpy().astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = (values - min_value) / (max_value - min_value)
    red = (normalized * 255.0).round().astype(np.uint8)
    green = np.zeros_like(red, dtype=np.uint8)
    blue = ((1.0 - normalized) * 80.0).round().astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=2), mode="RGB")


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
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return {
        "path": str(path),
        "exists": True,
        "artifact_type": artifact_type,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "frontend_ready": artifact_type != "model_checkpoint",
        "required_for_frontend": artifact_type != "model_checkpoint",
    }


def _summary_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Summary artifact source not found: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_type_for(name: str) -> str:
    mapping = {
        "training_result": "TrainingResult",
        "model_checkpoint": "model_checkpoint",
        "anomaly_evaluation": "anomaly_detection_evaluation",
        "learning_curves": "anomaly_learning_curves",
        "confusion_matrix": "anomaly_confusion_matrix",
        "qualitative_samples": "anomaly_qualitative_samples",
        "explainability": "anomaly_reconstruction_explainability",
    }
    return mapping[name]


def _config_id(config: dict[str, Any]) -> str:
    return _require_string(
        _require_dict(config.get("identity"), "identity").get("run_config_id"),
        "identity.run_config_id",
    )


def _split_manifest_path(config: dict[str, Any]) -> str:
    return _require_string(
        _require_dict(config.get("dataset_binding"), "dataset_binding").get(
            "split_manifest_path"
        ),
        "dataset_binding.split_manifest_path",
    )


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


def _require_binary_label_mix(values: Any, split_name: str) -> list[int]:
    labels = [_require_binary_int(value, f"{split_name} label") for value in _require_list(values, "labels")]
    if set(labels) != {0, 1}:
        raise ValueError(f"{split_name} labels must include both normal and anomaly.")
    return labels


def _label_name(label_id: int) -> str:
    return "anomaly" if label_id == 1 else "normal"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _require_numeric_list(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return [_require_numeric(item, f"{field_name}[]") for item in value]


def _require_binary_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a binary integer.")
    if value not in {0, 1}:
        raise ValueError(f"{field_name} must be 0 or 1.")
    return value


if __name__ == "__main__":
    sys.exit(main())
