"""Config-driven image-to-tensor preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


def load_and_preprocess_image(path: str, config: dict[str, Any]) -> torch.Tensor:
    """Load one image and return a normalized CHW tensor without a batch dimension.

    The preprocessing behavior is governed by the provided config. Supported
    policies match the current MVTec preprocessing contract: force RGB, fixed
    resize, zero-to-one float32 pixel scaling, and configured mean/std
    normalization.
    """
    image_path = _validate_image_path(path)
    image = _load_image(image_path, config)
    image = _apply_rgb_policy(image, config)
    image = _apply_resize_policy(image, config)
    tensor = _image_to_scaled_tensor(image, config)
    return _apply_normalization(tensor, config)


def _validate_image_path(path: str) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError("Image path must be a non-empty string.")

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    return image_path


def _load_image(path: Path, config: dict[str, Any]) -> Image.Image:
    decode_failure_policy = config.get("decode_failure_policy")
    if decode_failure_policy != "reject":
        raise ValueError("Unsupported decode_failure_policy for image preprocessing.")

    try:
        with Image.open(path) as image:
            return image.copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unable to decode image file: {path}") from exc


def _apply_rgb_policy(image: Image.Image, config: dict[str, Any]) -> Image.Image:
    if config.get("rgb_conversion_policy") != "force_rgb":
        raise ValueError("Unsupported rgb_conversion_policy for image preprocessing.")

    converted = image.convert("RGB")
    allowed_modes = config.get("allowed_image_modes")
    if not isinstance(allowed_modes, list) or converted.mode not in allowed_modes:
        raise ValueError("Converted image mode is not allowed by preprocessing config.")

    return converted


def _apply_resize_policy(image: Image.Image, config: dict[str, Any]) -> Image.Image:
    if config.get("resize_policy") != "fixed":
        raise ValueError("Unsupported resize_policy for image preprocessing.")

    width, height = _require_image_size(config)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _require_image_size(config: dict[str, Any]) -> tuple[int, int]:
    image_size = config.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or isinstance(image_size[0], bool)
        or isinstance(image_size[1], bool)
        or not isinstance(image_size[0], int)
        or not isinstance(image_size[1], int)
        or image_size[0] <= 0
        or image_size[1] <= 0
    ):
        raise ValueError("preprocessing image_size must be [width, height].")

    return image_size[0], image_size[1]


def _image_to_scaled_tensor(
    image: Image.Image, config: dict[str, Any]
) -> torch.Tensor:
    if config.get("pixel_scale") != "zero_to_one_float32":
        raise ValueError("Unsupported pixel_scale for image preprocessing.")

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def _apply_normalization(
    tensor: torch.Tensor, config: dict[str, Any]
) -> torch.Tensor:
    if config.get("normalization_policy") != "imagenet":
        raise ValueError("Unsupported normalization_policy for image preprocessing.")

    mean = _require_float_sequence(config, "normalization_mean")
    std = _require_float_sequence(config, "normalization_std")
    if len(mean) != tensor.shape[0] or len(std) != tensor.shape[0]:
        raise ValueError("Normalization mean/std must match tensor channel count.")
    if any(value == 0.0 for value in std):
        raise ValueError("Normalization std values must be non-zero.")

    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def _require_float_sequence(config: dict[str, Any], field: str) -> list[float]:
    values = config.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"preprocessing {field} must be a non-empty list.")

    converted_values = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"preprocessing {field} values must be numeric.")
        converted_values.append(float(value))

    return converted_values
