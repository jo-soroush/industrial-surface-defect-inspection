"""Factory boundary for resolving governed run configs to model skeletons."""

from __future__ import annotations

from typing import Any

from inspection_ai.models.autoencoder import AutoencoderModel
from inspection_ai.models.cnn import CNNModel
from inspection_ai.models.efficientnet_b0 import EfficientNetB0Model
from inspection_ai.models.mlp import MLPModel
from inspection_ai.models.mobilenet_v3 import MobileNetV3Model
from inspection_ai.models.resnet18 import ResNet18Model
from inspection_ai.models.yolo_model import YOLOModel


_MODEL_CLASSES = {
    "mlp": MLPModel,
    "cnn": CNNModel,
    "resnet18": ResNet18Model,
    "efficientnet_b0": EfficientNetB0Model,
    "mobilenet_v3": MobileNetV3Model,
    "autoencoder": AutoencoderModel,
    "yolo": YOLOModel,
}


def create_model(config: dict[str, Any]) -> object:
    """Create the configured model skeleton from a resolved run config."""
    model_identity = config.get("model_identity")
    if model_identity is None:
        raise ValueError("Missing required model_identity section.")
    if not isinstance(model_identity, dict):
        raise ValueError("model_identity must be a dictionary.")

    model_type = model_identity.get("model_type")
    if model_type is None:
        raise ValueError("Missing required model_identity.model_type value.")
    if not isinstance(model_type, str):
        raise ValueError("model_identity.model_type must be a string.")

    try:
        model_class = _MODEL_CLASSES[model_type]
    except KeyError as exc:
        allowed_values = ", ".join(sorted(_MODEL_CLASSES))
        raise ValueError(
            f"Unsupported model_identity.model_type '{model_type}'. "
            f"Allowed values: {allowed_values}."
        ) from exc

    return model_class(config)
