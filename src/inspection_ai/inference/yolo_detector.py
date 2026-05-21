"""Reusable YOLO live detection helper for image inspection."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from ..contracts.inspection import (
    DetectionBox,
    DetectionResult,
    DetectionTraceability,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "yolo_train_v0_2_0"
DEFAULT_MODEL_NAME = "yolo"
DEFAULT_MODEL_VERSION = "0.2.0"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_IMAGE_SIZE = 640
DEFAULT_DEVICE = "cpu"

WEIGHTS_PATH = REPO_ROOT / "artifacts/detection/yolo/runs/yolo_train_v0_2_0/weights/best.pt"
RUN_CONFIG_PATH = REPO_ROOT / "configs/runs/yolo_train_v0_2_0.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "configs/models/yolo.yaml"
DATASET_YAML_PATH = REPO_ROOT / "data/processed/gc10det_yolo/dataset.yaml"
SOURCE_CONTRACT_PATH = (
    REPO_ROOT
    / "artifacts/models/predictions/detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
)


class YOLODetector:
    """Run governed YOLO v0.2.0 detection on one image."""

    def __init__(
        self,
        *,
        weights_path: Path = WEIGHTS_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.class_labels = _load_dataset_labels(DATASET_YAML_PATH)
        self._model: Any | None = None

    def predict(self, image: Image.Image) -> DetectionResult:
        """Run live detection on one PIL image."""
        if not isinstance(image, Image.Image):
            raise TypeError("YOLODetector.predict requires a PIL.Image.Image input.")
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {_repo_relative(self.weights_path)}")

        model = self._load_model()
        results = model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not isinstance(results, list):
            raise RuntimeError("YOLO prediction output must be a list of results.")
        if not results:
            return build_detection_result_from_yolo_result(
                result=None,
                image_width=image.width,
                image_height=image.height,
                class_labels=self.class_labels,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
            )
        return build_detection_result_from_yolo_result(
            result=results[0],
            image_width=image.width,
            image_height=image.height,
            class_labels=self.class_labels,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )

    def _load_model(self) -> Any:
        """Load the ultralytics YOLO model lazily."""
        if self._model is not None:
            return self._model
        try:
            ultralytics = import_module("ultralytics")
        except ImportError as exc:
            raise RuntimeError(
                "YOLO live detection requires the 'ultralytics' package. "
                "Install the backend dependency before enabling detection prediction."
            ) from exc

        yolo_cls = getattr(ultralytics, "YOLO", None)
        if yolo_cls is None:
            raise RuntimeError("The installed 'ultralytics' package does not expose YOLO.")
        self._model = yolo_cls(str(self.weights_path))
        return self._model


def build_detection_result_from_yolo_result(
    *,
    result: Any | None,
    image_width: int,
    image_height: int,
    class_labels: list[str],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> DetectionResult:
    """Convert one ultralytics result into the unified DetectionResult schema."""
    _validate_image_dimensions(image_width, image_height)
    if not class_labels:
        raise ValueError("class_labels must not be empty.")

    detections = _extract_detection_boxes(
        result=result,
        image_width=image_width,
        image_height=image_height,
        class_labels=class_labels,
    )
    best_detection = detections[0] if detections else None
    review_status = "review_required" if detections else "no_detection"

    return DetectionResult(
        status="success",
        model_name=DEFAULT_MODEL_NAME,
        model_version=DEFAULT_MODEL_VERSION,
        run_id=DEFAULT_RUN_ID,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_width=image_width,
        image_height=image_height,
        predicted_box_count=len(detections),
        defect_count=len(detections),
        detections=detections,
        best_detection=best_detection,
        review_status=review_status,
        production_ready=False,
        deployment_safe=False,
        limitations=[
            "Detection output is local model output and not production-ready.",
            "Detection output is not deployment-safe.",
        ],
        traceability=DetectionTraceability(
            checkpoint_path=_repo_relative(WEIGHTS_PATH),
            run_config_path=_repo_relative(RUN_CONFIG_PATH),
            model_config_path=_repo_relative(MODEL_CONFIG_PATH),
            source_contract=_repo_relative(SOURCE_CONTRACT_PATH),
        ),
    )


def _extract_detection_boxes(
    *,
    result: Any | None,
    image_width: int,
    image_height: int,
    class_labels: list[str],
) -> list[DetectionBox]:
    if result is None:
        return []

    boxes = getattr(result, "boxes", None)
    xyxy_values = _tensor_to_list(getattr(boxes, "xyxy", None))
    conf_values = _tensor_to_list(getattr(boxes, "conf", None))
    cls_values = _tensor_to_list(getattr(boxes, "cls", None))
    if not xyxy_values:
        return []
    if len(xyxy_values) != len(conf_values) or len(xyxy_values) != len(cls_values):
        raise ValueError("YOLO boxes tensor lengths are inconsistent.")

    detections: list[DetectionBox] = []
    for index, (xyxy, confidence_value, class_id_value) in enumerate(
        zip(xyxy_values, conf_values, cls_values, strict=True)
    ):
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly four values.")

        confidence = float(confidence_value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence values must be between 0 and 1.")

        class_id = int(class_id_value)
        if class_id < 0 or class_id >= len(class_labels):
            raise ValueError("class_id is outside the configured class label range.")
        class_label = class_labels[class_id]

        x1, y1, x2, y2 = (float(value) for value in xyxy)
        warnings: list[str] = []
        if x1 < 0.0 or y1 < 0.0 or x2 > float(image_width) or y2 > float(image_height):
            x1 = min(max(x1, 0.0), float(image_width))
            y1 = min(max(y1, 0.0), float(image_height))
            x2 = min(max(x2, 0.0), float(image_width))
            y2 = min(max(y2, 0.0), float(image_height))
            warnings.append("clamped_to_image_bounds")
        if x1 > x2 or y1 > y2:
            raise ValueError("bbox_xyxy coordinates must satisfy x1 <= x2 and y1 <= y2.")

        detections.append(
            DetectionBox(
                box_id=index,
                class_id=class_id,
                class_label=class_label,
                display_label=_display_label(class_label),
                confidence=confidence,
                bbox_format="xyxy",
                bbox_xyxy=[x1, y1, x2, y2],
                score_rank=index + 1,
                is_best_prediction=index == 0,
                warnings=warnings,
            )
        )
    return detections


def _load_dataset_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {_repo_relative(path)}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YOLO dataset YAML is invalid: {_repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"YOLO dataset YAML must contain an object: {_repo_relative(path)}")
    names = payload.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError("YOLO dataset YAML names must be a non-empty list.")
    labels = []
    for item in names:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("YOLO dataset YAML names must contain non-empty strings.")
        labels.append(item.strip())
    return labels


def _tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, list):
        return value
    raise ValueError("YOLO tensor-like output must provide tolist().")


def _display_label(class_label: str) -> str:
    return class_label.replace("_", " ").strip().capitalize()


def _validate_image_dimensions(image_width: int, image_height: int) -> None:
    if image_width <= 0:
        raise ValueError("image_width must be positive.")
    if image_height <= 0:
        raise ValueError("image_height must be positive.")


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
