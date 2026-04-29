"""Reconstruction-based anomaly evaluation utilities."""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch


THRESHOLD_STRATEGIES = {"mean", "percentile95"}


def compute_reconstruction_scores(
    images: torch.Tensor, reconstructions: torch.Tensor
) -> list[float]:
    """Return per-sample mean squared reconstruction error scores."""
    if not isinstance(images, torch.Tensor):
        raise ValueError("images must be a torch.Tensor.")
    if not isinstance(reconstructions, torch.Tensor):
        raise ValueError("reconstructions must be a torch.Tensor.")
    if images.shape != reconstructions.shape:
        raise ValueError("images and reconstructions must have matching shapes.")
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W].")

    scores = torch.mean((images - reconstructions) ** 2, dim=(1, 2, 3))
    return [float(score) for score in scores.detach().cpu().tolist()]


def run_anomaly_inference(model: Any, dataloader: Any) -> dict[str, list[Any]]:
    """Run autoencoder inference and collect scores, labels, paths, and masks."""
    if dataloader is None:
        raise ValueError("dataloader must not be None.")

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    paths: list[str] = []
    mask_paths: list[str | None] = []
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, dict):
                raise ValueError("Anomaly dataloader batch must be a dictionary.")

            images = batch.get("image")
            if not isinstance(images, torch.Tensor):
                raise ValueError("Anomaly dataloader batch image must be a torch.Tensor.")
            images = images.to(model_device)

            reconstructions = model(images)
            batch_scores = compute_reconstruction_scores(images, reconstructions)
            scores.extend(batch_scores)
            labels.extend(_batch_values_to_ints(batch.get("label"), "label"))
            paths.extend(_batch_values_to_strings(batch.get("path"), "path"))
            mask_paths.extend(_batch_mask_paths(batch.get("mask_path"), len(batch_scores)))

    if not scores:
        raise ValueError("Anomaly inference produced no scores.")
    if len(labels) != len(scores):
        raise ValueError("Anomaly inference labels count must match scores count.")
    if len(paths) != len(scores):
        raise ValueError("Anomaly inference paths count must match scores count.")
    if len(mask_paths) != len(scores):
        raise ValueError("Anomaly inference mask_paths count must match scores count.")

    return {
        "scores": scores,
        "labels": labels,
        "paths": paths,
        "mask_paths": mask_paths,
    }


def compute_threshold(train_scores: list[float], config: dict[str, Any]) -> float:
    """Compute the anomaly decision threshold from train scores."""
    scores = _validate_scores(train_scores, "train_scores")
    strategy = config.get("threshold_strategy", "percentile95")
    if strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(
            "threshold_strategy must be one of: "
            f"{sorted(THRESHOLD_STRATEGIES)}."
        )

    if strategy == "mean":
        return float(sum(scores) / len(scores))

    return float(_percentile(scores, 95.0))


def generate_predictions(scores: list[float], threshold: float) -> list[int]:
    """Return binary anomaly predictions from continuous scores."""
    validated_scores = _validate_scores(scores, "scores")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("threshold must be numeric.")

    threshold_value = float(threshold)
    return [1 if score > threshold_value else 0 for score in validated_scores]


def compute_anomaly_metrics(
    labels: list[int], scores: list[float], predictions: list[int]
) -> dict[str, float]:
    """Compute image-level anomaly metrics from labels, scores, and predictions."""
    validated_labels = _validate_binary_labels(labels)
    validated_scores = _validate_scores(scores, "scores")
    validated_predictions = _validate_binary_predictions(predictions)
    if len(validated_labels) != len(validated_scores):
        raise ValueError("labels and scores must have the same length.")
    if len(validated_labels) != len(validated_predictions):
        raise ValueError("labels and predictions must have the same length.")
    if set(validated_labels) != {0, 1}:
        raise ValueError("ROC-AUC requires both normal and anomaly labels.")

    return {
        "roc_auc": _compute_roc_auc(validated_labels, validated_scores),
        "precision": _precision(validated_labels, validated_predictions),
        "recall": _recall(validated_labels, validated_predictions),
        "f1": _f1(validated_labels, validated_predictions),
    }


def evaluate_anomaly_detection(
    model: Any,
    train_dataloader: Any,
    test_dataloader: Any,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run train-threshold scoring and test-set anomaly evaluation."""
    evaluation_config = {} if config is None else config
    train_inference = run_anomaly_inference(model, train_dataloader)
    test_inference = run_anomaly_inference(model, test_dataloader)
    threshold = compute_threshold(train_inference["scores"], evaluation_config)
    predictions = generate_predictions(test_inference["scores"], threshold)
    metrics = compute_anomaly_metrics(
        labels=test_inference["labels"],
        scores=test_inference["scores"],
        predictions=predictions,
    )

    return {
        "anomaly_scores": test_inference["scores"],
        "predictions": predictions,
        "threshold": float(threshold),
        "metrics": metrics,
    }


def _batch_values_to_ints(values: Any, field_name: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        flattened = values.detach().cpu().reshape(-1).tolist()
        return [_validate_binary_int(value, field_name) for value in flattened]
    if isinstance(values, list):
        return [_validate_binary_int(value, field_name) for value in values]
    raise ValueError(f"Anomaly dataloader batch {field_name} must be a tensor or list.")


def _batch_values_to_strings(values: Any, field_name: str) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError(f"Anomaly dataloader batch {field_name} must be a string list.")

    strings = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Anomaly dataloader batch {field_name} values must be non-empty strings."
            )
        strings.append(value)
    return strings


def _batch_mask_paths(values: Any, expected_count: int) -> list[str | None]:
    if values is None:
        return [None] * expected_count
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("Anomaly dataloader batch mask_path must be a string list.")

    mask_paths = []
    for value in values:
        if value in (None, ""):
            mask_paths.append(None)
        elif isinstance(value, str):
            mask_paths.append(value)
        else:
            raise ValueError("Anomaly dataloader batch mask_path values are invalid.")
    return mask_paths


def _validate_scores(scores: list[float], field_name: str) -> list[float]:
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"{field_name} must be a non-empty list.")

    validated = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError(f"{field_name}[{index}] must be numeric.")
        validated.append(float(score))
    return validated


def _validate_binary_labels(labels: list[int]) -> list[int]:
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must be a non-empty list.")
    return [_validate_binary_int(label, "labels") for label in labels]


def _validate_binary_predictions(predictions: list[int]) -> list[int]:
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("predictions must be a non-empty list.")
    return [_validate_binary_int(prediction, "predictions") for prediction in predictions]


def _validate_binary_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} values must be binary integers.")
    if value not in {0, 1}:
        raise ValueError(f"{field_name} values must be 0 or 1.")
    return value


def _percentile(scores: list[float], percentile: float) -> float:
    sorted_scores = sorted(scores)
    if len(sorted_scores) == 1:
        return sorted_scores[0]

    rank = (percentile / 100.0) * (len(sorted_scores) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_scores) - 1)
    fraction = rank - lower_index
    return sorted_scores[lower_index] + (
        (sorted_scores[upper_index] - sorted_scores[lower_index]) * fraction
    )


def _compute_roc_auc(labels: list[int], scores: list[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
    except ModuleNotFoundError:
        return _rank_based_roc_auc(labels, scores)

    return float(roc_auc_score(labels, scores))


def _rank_based_roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both positive and negative labels.")

    wins = 0.0
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return float(wins / (len(positives) * len(negatives)))


def _precision(labels: list[int], predictions: list[int]) -> float:
    true_positive = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1)
    false_positive = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
    return _safe_ratio(true_positive, true_positive + false_positive)


def _recall(labels: list[int], predictions: list[int]) -> float:
    true_positive = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1)
    false_negative = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 0)
    return _safe_ratio(true_positive, true_positive + false_negative)


def _f1(labels: list[int], predictions: list[int]) -> float:
    precision = _precision(labels, predictions)
    recall = _recall(labels, predictions)
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
