"""Training data-loader construction boundary.

This module owns future construction of training data loaders. Notebooks must
not become the canonical implementation for data-loading behavior. Framework
dataset construction, image loading, and batching are intentionally deferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from inspection_ai.preprocessing.mvtec_dataset import (
    MVTecAnomalyDataset,
    MVTecBinaryClassificationDataset,
)


def build_data_loaders(config: dict[str, Any]) -> dict[str, Any]:
    """Return governed split entries for a training config."""
    dataset_binding = config.get("dataset_binding")
    if dataset_binding is None:
        raise ValueError("Training config is missing required dataset_binding section.")
    if not isinstance(dataset_binding, dict):
        raise ValueError("Training config section dataset_binding must be a dictionary.")

    dataset_id = dataset_binding.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise ValueError(
            "Training config is missing required dataset_binding.dataset_id."
        )

    dataset_version = dataset_binding.get("dataset_version")
    if not isinstance(dataset_version, str):
        raise ValueError(
            "Training config is missing required dataset_binding.dataset_version."
        )

    split_manifest_path = dataset_binding.get("split_manifest_path")
    if not isinstance(split_manifest_path, str) or not split_manifest_path:
        raise ValueError(
            "Training config is missing required dataset_binding.split_manifest_path."
        )

    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Training config is missing required preprocessing section.")

    preprocessing_policy_path = preprocessing.get("preprocessing_policy_path")
    if not isinstance(preprocessing_policy_path, str) or not preprocessing_policy_path:
        raise ValueError(
            "Training config is missing required preprocessing.preprocessing_policy_path."
        )

    training_runtime = config.get("training_runtime")
    if not isinstance(training_runtime, dict):
        raise ValueError("Training config is missing required training_runtime section.")

    batch_size = training_runtime.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Training config training_runtime.batch_size must be > 0.")

    seed = training_runtime.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Training config training_runtime.seed must be an integer.")

    identity = config.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Training config is missing required identity section.")

    task_type = identity.get("task_type")
    if not isinstance(task_type, str):
        raise ValueError("Training config is missing required identity.task_type.")

    manifest = _load_split_manifest(split_manifest_path)
    _validate_manifest_identity(manifest, dataset_id, dataset_version)
    preprocessing_config = _load_yaml_config(
        preprocessing_policy_path, "preprocessing policy"
    )

    if task_type == "classification":
        return _build_classification_data_loaders(
            dataset_binding=dataset_binding,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            split_manifest_path=split_manifest_path,
            preprocessing_policy_path=preprocessing_policy_path,
            preprocessing_config=preprocessing_config,
            manifest=manifest,
            batch_size=batch_size,
            seed=seed,
            task_type=task_type,
        )

    if task_type == "anomaly_detection":
        return _build_anomaly_data_loaders(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            split_manifest_path=split_manifest_path,
            preprocessing_policy_path=preprocessing_policy_path,
            preprocessing_config=preprocessing_config,
            manifest=manifest,
            batch_size=batch_size,
            seed=seed,
            task_type=task_type,
        )

    raise ValueError(f"Unsupported training config identity.task_type: {task_type}.")


def _build_classification_data_loaders(
    *,
    dataset_binding: dict[str, Any],
    dataset_id: str,
    dataset_version: str,
    split_manifest_path: str,
    preprocessing_policy_path: str,
    preprocessing_config: dict[str, Any],
    manifest: dict[str, Any],
    batch_size: int,
    seed: int,
    task_type: str,
) -> dict[str, Any]:
    class_mapping_path = dataset_binding.get("class_mapping_path")
    if not isinstance(class_mapping_path, str) or not class_mapping_path:
        raise ValueError(
            "Training config is missing required dataset_binding.class_mapping_path."
        )

    train_entries = _require_split_entries(manifest, "train")
    validation_entries = _require_split_entries(manifest, "validation")
    test_entries = _require_split_entries(manifest, "test")
    class_mapping_config = _load_yaml_config(class_mapping_path, "class mapping")

    train_dataset = MVTecBinaryClassificationDataset(
        train_entries, preprocessing_config, class_mapping_config
    )
    validation_dataset = MVTecBinaryClassificationDataset(
        validation_entries, preprocessing_config, class_mapping_config
    )
    test_dataset = MVTecBinaryClassificationDataset(
        test_entries, preprocessing_config, class_mapping_config
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = (
        None
        if len(validation_entries) == 0
        else torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "task_type": task_type,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "split_manifest_path": split_manifest_path,
        "class_mapping_path": class_mapping_path,
        "preprocessing_policy_path": preprocessing_policy_path,
        "train": train_entries,
        "validation": validation_entries,
        "test": test_entries,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
    }


def _build_anomaly_data_loaders(
    *,
    dataset_id: str,
    dataset_version: str,
    split_manifest_path: str,
    preprocessing_policy_path: str,
    preprocessing_config: dict[str, Any],
    manifest: dict[str, Any],
    batch_size: int,
    seed: int,
    task_type: str,
) -> dict[str, Any]:
    _validate_anomaly_manifest_task_type(manifest)
    train_entries = _require_anomaly_split_entries(manifest, "train")
    validation_entries = _require_anomaly_split_entries(manifest, "validation")
    test_entries = _require_anomaly_split_entries(manifest, "test")
    if not train_entries:
        raise ValueError("Anomaly split manifest train_entries must not be empty.")
    if not test_entries:
        raise ValueError("Anomaly split manifest test_entries must not be empty.")

    train_dataset = MVTecAnomalyDataset(train_entries, preprocessing_config)
    validation_dataset = MVTecAnomalyDataset(validation_entries, preprocessing_config)
    test_dataset = MVTecAnomalyDataset(test_entries, preprocessing_config)

    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = (
        None
        if len(validation_entries) == 0
        else torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "task_type": task_type,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "split_manifest_path": split_manifest_path,
        "class_mapping_path": None,
        "preprocessing_policy_path": preprocessing_policy_path,
        "train": train_entries,
        "validation": validation_entries,
        "test": test_entries,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
    }


def _load_yaml_config(path: str, config_name: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_name} file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"{config_name} YAML is invalid: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must parse to a dictionary.")

    return config


def _load_split_manifest(split_manifest_path: str) -> dict[str, Any]:
    manifest_path = Path(split_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest file not found: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Split manifest YAML is invalid: {manifest_path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must parse to a dictionary.")

    return manifest


def _validate_manifest_identity(
    manifest: dict[str, Any], dataset_id: str, dataset_version: str
) -> None:
    manifest_dataset_id = manifest.get("dataset_id")
    if manifest_dataset_id != dataset_id:
        raise ValueError(
            "Split manifest dataset_id does not match training config dataset_id."
        )

    manifest_dataset_version = manifest.get("dataset_version")
    if manifest_dataset_version != dataset_version:
        raise ValueError(
            "Split manifest dataset_version does not match training config dataset_version."
        )


def _require_split_entries(
    manifest: dict[str, Any], split_name: str
) -> list[dict[str, Any]]:
    field_name = f"{split_name}_entries"
    entries = manifest.get(field_name)
    if not isinstance(entries, list):
        raise ValueError(f"Split manifest {field_name} must be a list.")

    for index, entry in enumerate(entries):
        _validate_split_entry(entry, split_name, field_name, index)

    return entries


def _validate_split_entry(
    entry: Any, split_name: str, field_name: str, index: int
) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"Split manifest {field_name}[{index}] must be a dictionary.")

    for field in ("path", "category", "split", "label"):
        value = entry.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Split manifest {field_name}[{index}].{field} must be a non-empty string."
            )

    if entry["split"] != split_name:
        raise ValueError(
            f"Split manifest {field_name}[{index}].split must match {split_name}."
        )


def _validate_anomaly_manifest_task_type(manifest: dict[str, Any]) -> None:
    task_type = manifest.get("task_type")
    if task_type != "anomaly_detection":
        raise ValueError("Anomaly split manifest task_type must be anomaly_detection.")


def _require_anomaly_split_entries(
    manifest: dict[str, Any], split_name: str
) -> list[dict[str, Any]]:
    field_name = f"{split_name}_entries"
    entries = manifest.get(field_name)
    if not isinstance(entries, list):
        raise ValueError(f"Anomaly split manifest {field_name} must be a list.")

    for index, entry in enumerate(entries):
        _validate_anomaly_split_entry(entry, split_name, field_name, index)

    return entries


def _validate_anomaly_split_entry(
    entry: Any, split_name: str, field_name: str, index: int
) -> None:
    if not isinstance(entry, dict):
        raise ValueError(
            f"Anomaly split manifest {field_name}[{index}] must be a dictionary."
        )

    for field in ("image_path", "category", "split", "label", "defect_type"):
        value = entry.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Anomaly split manifest {field_name}[{index}].{field} "
                "must be a non-empty string."
            )

    image_path = Path(entry["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Anomaly split manifest {field_name}[{index}].image_path not found: "
            f"{image_path}"
        )

    if entry["split"] != split_name:
        raise ValueError(
            f"Anomaly split manifest {field_name}[{index}].split must match {split_name}."
        )

    label = entry["label"]
    label_id = entry.get("label_id")
    if label not in {"normal", "anomaly"}:
        raise ValueError(
            f"Anomaly split manifest {field_name}[{index}].label must be normal or anomaly."
        )

    if isinstance(label_id, bool) or not isinstance(label_id, int):
        raise ValueError(
            f"Anomaly split manifest {field_name}[{index}].label_id must be an integer."
        )

    expected_label_id = 0 if label == "normal" else 1
    if label_id != expected_label_id:
        raise ValueError(
            f"Anomaly split manifest {field_name}[{index}].label_id must be "
            f"{expected_label_id} for label {label}."
        )

    mask_path = entry.get("mask_path")
    if label == "normal":
        if mask_path is not None:
            raise ValueError(
                f"Anomaly split manifest {field_name}[{index}].mask_path must be null "
                "for normal samples."
            )
        return

    if split_name == "train":
        raise ValueError(
            f"Anomaly split manifest {field_name}[{index}] cannot contain anomaly "
            "samples in the train split."
        )

    if not isinstance(mask_path, str) or not mask_path:
        raise ValueError(
            f"Anomaly split manifest {field_name}[{index}].mask_path must be a "
            "non-empty string for anomaly samples."
        )

    if not Path(mask_path).is_file():
        raise FileNotFoundError(
            f"Anomaly split manifest {field_name}[{index}].mask_path not found: "
            f"{mask_path}"
        )
