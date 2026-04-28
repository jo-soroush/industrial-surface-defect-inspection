"""Training loop boundary skeleton for Phase 3 model development.

This module defines the governed source location for future training-loop
execution. In Phase 3 it establishes the interface boundary for orchestrating a
minimal deterministic training simulation without real optimization or dataset
access.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import torch

from inspection_ai.training.training_result import TrainingResult


class TrainingLoop:
    """Interface for Phase 3 training loop orchestration."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, model: Any, data_loader: Any) -> TrainingResult:
        """Run the Phase 3 training simulation."""
        result = TrainingResult(self.config)
        start_time = time.perf_counter()
        _add_training_outputs(result, self.config, model, data_loader)
        _add_forward_contract_metadata(result, model, self.config)
        duration_seconds = time.perf_counter() - start_time
        _add_completion_metadata(result, duration_seconds)
        _add_data_loader_metadata(result, data_loader)
        _add_model_metadata(result, self.config)
        _add_config_summary_metadata(result, self.config)
        return result


def run_training_loop(
    config: dict[str, Any], model: Any, data_loader: Any
) -> TrainingResult:
    """Run the Phase 3 training simulation."""
    result = TrainingResult(config)
    start_time = time.perf_counter()
    _add_training_outputs(result, config, model, data_loader)
    _add_forward_contract_metadata(result, model, config)
    duration_seconds = time.perf_counter() - start_time
    _add_completion_metadata(result, duration_seconds)
    _add_data_loader_metadata(result, data_loader)
    _add_model_metadata(result, config)
    _add_config_summary_metadata(result, config)
    return result


def _add_training_outputs(
    result: TrainingResult, config: dict[str, Any], model: Any, data_loader: Any
) -> None:
    task_type = config["identity"]["task_type"]
    model_identity = config.get("model_identity", {})
    model_type = (
        model_identity.get("model_type") if isinstance(model_identity, dict) else None
    )
    if model_type == "mlp" and task_type == "classification":
        _add_one_batch_mlp_training_outputs(result, config, model, data_loader)
        return

    _add_simulated_outputs(result, config)


def _add_one_batch_mlp_training_outputs(
    result: TrainingResult, config: dict[str, Any], model: Any, data_loader: Any
) -> None:
    training_runtime = config.get("training_runtime", {})
    num_epochs = training_runtime.get("epochs")
    if isinstance(num_epochs, bool) or not isinstance(num_epochs, int) or num_epochs < 1:
        raise ValueError("Training config requires training_runtime.epochs >= 1.")

    learning_rate = training_runtime.get("learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError(
            "Training config requires numeric training_runtime.learning_rate."
        )

    if not isinstance(data_loader, dict):
        raise ValueError("data_loader must be a dictionary.")

    train_loader = data_loader.get("train_loader")
    if train_loader is None:
        raise ValueError("data_loader is missing required train_loader.")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_curve = []
    val_loss_curve = []
    last_batch_size = 0
    total_correct = 0
    total_samples = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    train_iterator = iter(train_loader)
    for epoch_index in range(num_epochs):
        try:
            batch = next(train_iterator)
        except StopIteration as exc:
            raise ValueError("train_loader must provide at least one batch per epoch.") from exc
        if not isinstance(batch, dict):
            raise ValueError("train_loader batch must be a dictionary.")

        images = batch.get("image")
        labels = batch.get("label")
        if not isinstance(images, torch.Tensor):
            raise ValueError("train_loader batch image must be a torch.Tensor.")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("train_loader batch label must be a torch.Tensor.")

        model_device = next(model.parameters()).device
        images = images.to(model_device)
        labels = labels.to(model_device).reshape(-1)

        optimizer.zero_grad()
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("MLP epoch logits must have shape [B, 2].")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("MLP epoch logits and labels batch sizes must match.")

        labels = labels.long()
        predictions = torch.argmax(logits, dim=1)
        correct, total, tp, fp, fn = _compute_binary_classification_counts(
            predictions, labels
        )
        epoch_accuracy = _safe_ratio(correct, total)
        epoch_f1 = _compute_f1(tp, fp, fn)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        if isinstance(loss_value, bool) or not isinstance(loss_value, (int, float)):
            raise ValueError("MLP epoch training loss must be numeric.")
        real_train_loss = float(loss_value)
        train_loss_curve.append(real_train_loss)
        val_loss_curve.append(real_train_loss)
        last_batch_size = int(images.shape[0])
        total_correct += correct
        total_samples += total
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(
            "epoch_metrics "
            f"epoch={epoch_index + 1} "
            f"loss={real_train_loss:.6f} "
            f"accuracy={epoch_accuracy:.6f} "
            f"f1={epoch_f1:.6f}"
        )

    result.add_learning_point("train_loss", train_loss_curve)
    result.add_learning_point("val_loss", val_loss_curve)
    train_accuracy = _safe_ratio(total_correct, total_samples)
    train_f1 = _compute_f1(total_tp, total_fp, total_fn)
    result.add_metric("train_accuracy", train_accuracy)
    result.add_metric("train_f1", train_f1)
    result.add_metric("accuracy", train_accuracy)
    result.add_metric("f1", train_f1)
    result.add_metadata("epochs", training_runtime.get("epochs"))
    result.add_metadata("device", training_runtime.get("device"))
    result.add_metadata("real_training_batch_checked", True)
    result.add_metadata("real_training_batch_size", last_batch_size)
    result.add_metadata("real_training_batches_per_epoch", 1)
    result.add_metadata("real_training_epoch_count", num_epochs)
    result.add_metadata(
        "real_training_loss_source", "one_batch_per_epoch_cross_entropy"
    )


def _compute_binary_classification_counts(
    predictions: torch.Tensor, labels: torch.Tensor
) -> tuple[int, int, int, int, int]:
    if predictions.device != labels.device:
        raise ValueError("Predictions and labels must be on the same device.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    positive_class = 1
    correct = int((predictions == labels).sum().item())
    total = int(labels.numel())
    tp = int(((predictions == positive_class) & (labels == positive_class)).sum().item())
    fp = int(((predictions == positive_class) & (labels != positive_class)).sum().item())
    fn = int(((predictions != positive_class) & (labels == positive_class)).sum().item())
    return correct, total, tp, fp, fn


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _compute_f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def _add_simulated_outputs(result: TrainingResult, config: dict[str, Any]) -> None:
    task_type = config["identity"]["task_type"]
    training_runtime = config.get("training_runtime", {})
    num_epochs = training_runtime.get("epochs")
    if not isinstance(num_epochs, int) or num_epochs < 1:
        raise ValueError("Training config requires training_runtime.epochs >= 1.")

    train_loss_curve = []
    val_loss_curve = []
    simulated_metrics: dict[str, float] = {}
    for epoch in range(num_epochs):
        train_loss_curve.append(1.0 / (epoch + 1))
        val_loss_curve.append(1.2 / (epoch + 1))

        if task_type == "classification":
            simulated_metrics["accuracy"] = _interpolate_metric(
                epoch, num_epochs, 0.5, 0.9
            )
            simulated_metrics["f1"] = _interpolate_metric(epoch, num_epochs, 0.45, 0.88)
        elif task_type == "object_detection":
            simulated_metrics["mAP"] = _interpolate_metric(epoch, num_epochs, 0.2, 0.75)

    result.add_learning_point("train_loss", train_loss_curve)
    result.add_learning_point("val_loss", val_loss_curve)

    final_val_loss = val_loss_curve[-1]

    if task_type == "classification":
        result.add_metric("accuracy", simulated_metrics["accuracy"])
        result.add_metric("f1", simulated_metrics["f1"])
    elif task_type == "anomaly_detection":
        result.add_metric("reconstruction_loss", final_val_loss)
    elif task_type == "object_detection":
        result.add_metric("mAP", simulated_metrics["mAP"])

    result.add_metadata("epochs", training_runtime.get("epochs"))
    result.add_metadata("device", training_runtime.get("device"))


def _interpolate_metric(
    epoch: int, num_epochs: int, start: float, end: float
) -> float:
    if num_epochs <= 1:
        return end

    progress = epoch / (num_epochs - 1)
    return start + ((end - start) * progress)


def _add_completion_metadata(
    result: TrainingResult, duration_seconds: float
) -> None:
    completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result.add_metadata("completed_at", completed_at)
    result.add_metadata("duration_seconds", max(0.0, duration_seconds))


def _add_forward_contract_metadata(
    result: TrainingResult, model: Any, config: dict[str, Any]
) -> None:
    model_identity = config.get("model_identity")
    model_type = (
        model_identity.get("model_type") if isinstance(model_identity, dict) else None
    )
    if model_type != "mlp":
        result.add_metadata("forward_contract_checked", False)
        return

    forward = getattr(model, "forward", None)
    if not callable(forward):
        raise ValueError("MLP model must provide a callable forward method.")

    contract_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    with torch.no_grad():
        forward_output = forward(contract_input)

    if not isinstance(forward_output, torch.Tensor):
        raise ValueError("MLP forward contract output must be a torch.Tensor.")
    if list(forward_output.shape) != [1, 2]:
        raise ValueError("MLP forward contract output shape must be [1, 2].")

    result.add_metadata("forward_contract_checked", True)
    result.add_metadata("forward_contract_name", "torch_mlp_forward")
    result.add_metadata("forward_contract_batch_size", 1)
    result.add_metadata("forward_contract_output_dimension", 2)


def _add_data_loader_metadata(result: TrainingResult, data_loader: Any) -> None:
    if not isinstance(data_loader, dict):
        raise ValueError("data_loader must be a dictionary.")

    dataset_id = data_loader.get("dataset_id")
    task_type_from_loader = data_loader.get("task_type")
    train_entries = _require_data_loader_split(data_loader, "train")
    validation_entries = _require_data_loader_split(data_loader, "validation")
    test_entries = _require_data_loader_split(data_loader, "test")

    result.add_metadata("dataset_id", dataset_id)
    result.add_metadata("task_type_from_loader", task_type_from_loader)
    result.add_metadata("train_sample_count", len(train_entries))
    result.add_metadata("validation_sample_count", len(validation_entries))
    result.add_metadata("test_sample_count", len(test_entries))


def _require_data_loader_split(
    data_loader: dict[str, Any], split_name: str
) -> list[Any]:
    if split_name not in data_loader:
        raise ValueError(f"data_loader is missing required split: {split_name}.")

    entries = data_loader[split_name]
    if not isinstance(entries, list):
        raise ValueError(f"data_loader split {split_name} must be a list.")

    return entries


def _add_model_metadata(result: TrainingResult, config: dict[str, Any]) -> None:
    model_identity = config.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError("Training config is missing required model_identity section.")

    model_type = model_identity.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError(
            "Training config is missing required model_identity.model_type."
        )

    result.add_metadata("model_type", model_type)
    result.add_metadata("model_name", model_identity.get("model_name"))
    result.add_metadata("model_version", model_identity.get("model_version"))


def _add_config_summary_metadata(
    result: TrainingResult, config: dict[str, Any]
) -> None:
    identity = config.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Training config is missing required identity section.")

    track_id = identity.get("track_id")
    if not isinstance(track_id, str):
        raise ValueError("Training config is missing required identity.track_id.")

    dataset_binding = config.get("dataset_binding")
    preprocessing = config.get("preprocessing")

    dataset_version = (
        dataset_binding.get("dataset_version")
        if isinstance(dataset_binding, dict)
        else None
    )
    if isinstance(preprocessing, dict):
        preprocessing_version = preprocessing.get("preprocessing_version")
        augmentation_policy_version = preprocessing.get("augmentation_policy_version")
    else:
        preprocessing_version = None
        augmentation_policy_version = None

    result.add_metadata("track_id", track_id)
    result.add_metadata("dataset_version", dataset_version)
    result.add_metadata("preprocessing_version", preprocessing_version)
    result.add_metadata("augmentation_policy_version", augmentation_policy_version)
    result.add_metadata("training_config_id", identity.get("run_config_id"))

    training_runtime = config.get("training_runtime")
    if isinstance(training_runtime, dict):
        result.add_metadata("seed", training_runtime.get("seed"))
        result.add_metadata("device_policy", training_runtime.get("device"))

    if isinstance(dataset_binding, dict):
        result.add_metadata(
            "split_manifest_path", dataset_binding.get("split_manifest_path")
        )
