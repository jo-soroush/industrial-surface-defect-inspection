"""Generate Grad-CAM style heatmap artifacts for Track A CNN classification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from inspection_ai.models.factory import create_model
from inspection_ai.preprocessing.image_to_tensor import load_and_preprocess_image


MODEL_ARTIFACT_KEYS = (
    "model_artifact",
    "model",
    "model_artifact_path",
    "checkpoint",
    "checkpoint_path",
    "model_state_dict",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Track A CNN classification Grad-CAM heatmaps."
    )
    parser.add_argument(
        "--training-result",
        required=True,
        help="Path to TrainingResult JSON with model_artifact checkpoint.",
    )
    parser.add_argument(
        "--sample-predictions",
        required=True,
        help="Path to sample_predictions JSON artifact.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Maximum number of heatmaps to generate.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/explainability",
        help="Directory where explainability artifacts will be written.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1.")

    training_result_path = Path(args.training_result)
    sample_predictions_path = Path(args.sample_predictions)
    training_result = _load_json_file(training_result_path, "TrainingResult")
    sample_predictions = _load_json_file(sample_predictions_path, "sample_predictions")

    identity = _require_dict(training_result.get("identity"), "identity")
    metadata = _require_dict(training_result.get("metadata"), "metadata")
    artifacts = _require_dict(training_result.get("artifacts"), "artifacts")
    run_id = _require_string(identity.get("run_id"), "identity.run_id")
    task_type = _require_string(identity.get("task_type"), "identity.task_type")
    if task_type != "classification":
        raise ValueError("Grad-CAM heatmap generation only supports classification.")

    model_type = _require_string(identity.get("model_type"), "identity.model_type")
    if model_type != "cnn":
        raise ValueError(
            "Grad-CAM explainability is only supported for CNN models in this step."
        )

    _validate_sample_predictions(
        sample_predictions=sample_predictions,
        run_id=run_id,
        dataset_id=metadata.get("dataset_id"),
        model_name=metadata.get("model_name"),
    )
    config = _load_run_config(training_result)
    preprocessing_config = _load_preprocessing_config(config)
    checkpoint_path = _resolve_model_artifact_path(artifacts, metadata)
    model = create_model(config)
    _load_model_weights(model, checkpoint_path)
    model.eval()
    target_layer = _find_last_conv2d(model)

    run_output_dir = Path(args.output_dir) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    heatmaps = _generate_heatmaps(
        model=model,
        target_layer=target_layer,
        samples=sample_predictions["samples"],
        preprocessing_config=preprocessing_config,
        output_dir=run_output_dir,
        num_requested=args.num_samples,
    )
    if not heatmaps:
        raise ValueError("No Grad-CAM heatmaps were generated.")

    payload = {
        "artifact_type": "classification_heatmaps",
        "task_type": "classification",
        "explainability_method": "grad_cam",
        "run_id": run_id,
        "model_id": metadata.get("model_name"),
        "dataset_id": metadata.get("dataset_id"),
        "source_training_result": str(training_result_path),
        "source_sample_predictions": str(sample_predictions_path),
        "created_at": _utc_now_iso(),
        "num_requested": args.num_samples,
        "num_written": len(heatmaps),
        "heatmaps": heatmaps,
    }

    metadata_path = run_output_dir / f"classification_heatmaps__{run_id}.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"classification_heatmaps_artifact_path={metadata_path}")
    print(f"num_written={len(heatmaps)}")
    return 0


def _generate_heatmaps(
    model: Any,
    target_layer: nn.Conv2d,
    samples: list[dict[str, Any]],
    preprocessing_config: dict[str, Any],
    output_dir: Path,
    num_requested: int,
) -> list[dict[str, Any]]:
    _validate_heatmap_samples(samples)
    heatmaps = []
    for sample in samples[:num_requested]:
        heatmaps.append(
            _generate_one_heatmap(
                model=model,
                target_layer=target_layer,
                sample=sample,
                preprocessing_config=preprocessing_config,
                output_dir=output_dir,
            )
        )
    return heatmaps


def _generate_one_heatmap(
    model: Any,
    target_layer: nn.Conv2d,
    sample: dict[str, Any],
    preprocessing_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    sample_id = _require_string(sample.get("sample_id"), "sample.sample_id")
    image_path = Path(
        _require_string(
            sample.get("image_path") or sample.get("input_reference"),
            "sample.image_path",
        )
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"Sample image file not found: {image_path}")

    target_class_id = _require_int(
        sample.get("predicted_label_id"), "sample.predicted_label_id"
    )
    target_class_label = _require_string(
        sample.get("predicted_label"), "sample.predicted_label"
    )

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        activations.append(output.detach())

    def backward_hook(
        _module: Any, _grad_input: Any, grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        image_tensor = load_and_preprocess_image(str(image_path), preprocessing_config)
        logits = model(image_tensor.unsqueeze(0))
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("CNN model must output classification logits.")
        if target_class_id < 0 or target_class_id >= logits.shape[1]:
            raise ValueError("Sample predicted_label_id is outside model output range.")
        logits[0, target_class_id].backward()
        if not activations or not gradients:
            raise ValueError("Grad-CAM hooks did not capture activations and gradients.")
        heatmap = _build_grad_cam_heatmap(
            activation=activations[-1],
            gradient=gradients[-1],
            output_size=tuple(image_tensor.shape[-2:]),
        )
    finally:
        forward_handle.remove()
        backward_handle.remove()

    heatmap_path = output_dir / f"heatmap__{sample_id}.png"
    overlay_path = output_dir / f"overlay__{sample_id}.png"
    _save_heatmap_image(heatmap, heatmap_path)
    _save_overlay_image(image_path, heatmap, overlay_path)

    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "true_label": sample.get("true_label"),
        "predicted_label": target_class_label,
        "confidence": sample.get("confidence"),
        "correct": sample.get("correct"),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "target_class_id": target_class_id,
        "target_class_label": target_class_label,
    }


def _build_grad_cam_heatmap(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    max_value = cam.max()
    if float(max_value.item()) <= 0.0:
        raise ValueError("Grad-CAM heatmap is empty for the requested sample.")
    cam = cam / max_value
    return cam.detach().cpu().numpy().astype(np.float32)


def _save_heatmap_image(heatmap: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), mode="L")
    image.save(output_path)


def _save_overlay_image(image_path: Path, heatmap: np.ndarray, output_path: Path) -> None:
    with Image.open(image_path) as image:
        base = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    base_array = np.asarray(base, dtype=np.float32)
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    color = np.zeros_like(base_array)
    color[..., 0] = heatmap_uint8
    overlay = np.uint8((0.65 * base_array) + (0.35 * color))
    Image.fromarray(overlay, mode="RGB").save(output_path)


def _find_last_conv2d(model: Any) -> nn.Conv2d:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Unable to find a convolution layer for Grad-CAM.")
    return last_conv


def _validate_sample_predictions(
    sample_predictions: dict[str, Any],
    run_id: str,
    dataset_id: Any,
    model_name: Any,
) -> None:
    if sample_predictions.get("artifact_type") != "sample_predictions":
        raise ValueError("sample_predictions artifact_type must be sample_predictions.")
    if sample_predictions.get("task_type") != "classification":
        raise ValueError("sample_predictions task_type must be classification.")
    if sample_predictions.get("run_id") != run_id:
        raise ValueError("sample_predictions run_id must match TrainingResult run_id.")
    if dataset_id and sample_predictions.get("dataset_id") != dataset_id:
        raise ValueError(
            "sample_predictions dataset_id must match TrainingResult dataset_id."
        )
    _require_string(
        sample_predictions.get("source_training_result"),
        "sample_predictions.source_training_result",
    )
    if model_name is not None:
        model_id = _require_string(
            sample_predictions.get("model_id"),
            "sample_predictions.model_id",
        )
        if model_id != model_name:
            raise ValueError(
                "sample_predictions model_id must match TrainingResult model_name."
            )
    num_written = _require_int(
        sample_predictions.get("num_written"), "sample_predictions.num_written"
    )
    if num_written <= 0:
        raise ValueError("sample_predictions num_written must be > 0.")
    samples = sample_predictions.get("samples")
    if not isinstance(samples, list):
        raise ValueError("sample_predictions samples must be a list.")
    if not samples:
        raise ValueError("sample_predictions samples must be a non-empty list.")
    if len(samples) > num_written:
        raise ValueError(
            "sample_predictions samples length cannot exceed num_written."
        )


def _validate_heatmap_samples(samples: list[dict[str, Any]]) -> None:
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample_predictions.samples[{index}] must be an object.")
        _require_string(
            sample.get("sample_id"),
            f"sample_predictions.samples[{index}].sample_id",
        )
        _require_string(
            sample.get("image_path") or sample.get("input_reference"),
            f"sample_predictions.samples[{index}].image_path",
        )
        _require_string(
            sample.get("true_label"),
            f"sample_predictions.samples[{index}].true_label",
        )
        _require_string(
            sample.get("predicted_label"),
            f"sample_predictions.samples[{index}].predicted_label",
        )
        _require_int(
            sample.get("predicted_label_id"),
            f"sample_predictions.samples[{index}].predicted_label_id",
        )
        _require_number(
            sample.get("confidence"),
            f"sample_predictions.samples[{index}].confidence",
        )
        _require_bool(
            sample.get("correct"),
            f"sample_predictions.samples[{index}].correct",
        )


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


def _load_preprocessing_config(config: dict[str, Any]) -> dict[str, Any]:
    preprocessing = _require_dict(config.get("preprocessing"), "preprocessing")
    path = Path(
        _require_string(
            preprocessing.get("preprocessing_policy_path"),
            "preprocessing.preprocessing_policy_path",
        )
    )
    return _load_yaml_file(path, "preprocessing policy")


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
    raise FileNotFoundError("TrainingResult does not reference a saved model artifact.")


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


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
