"""Unit tests for the YOLO live detection helper."""

from __future__ import annotations

from src.inspection_ai.inference.yolo_detector import (
    build_detection_result_from_yolo_result,
)


class _TensorLike:
    def __init__(self, value: list[object]) -> None:
        self._value = value

    def tolist(self) -> list[object]:
        return self._value


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = _TensorLike(
            [
                [1840.27, -1.0, 1987.1, 1001.0],
                [640.0, 220.0, 790.0, 390.0],
            ]
        )
        self.conf = _TensorLike([0.802, 0.612])
        self.cls = _TensorLike([2.0, 4.0])


class _Result:
    boxes = _Boxes()


CLASS_LABELS = [
    "crease",
    "crescent_gap",
    "inclusion",
    "oil_spot",
    "punching_hole",
]


def test_build_detection_result_maps_yolo_boxes_to_contract() -> None:
    result = build_detection_result_from_yolo_result(
        result=_Result(),
        image_width=2048,
        image_height=1000,
        class_labels=CLASS_LABELS,
    )

    assert result.status == "success"
    assert result.predicted_box_count == 2
    assert result.defect_count == 2
    assert result.best_detection == result.detections[0]
    assert result.review_status == "review_required"
    assert result.production_ready is False
    assert result.deployment_safe is False

    first_box = result.detections[0]
    assert first_box.box_id == 0
    assert first_box.class_id == 2
    assert first_box.class_label == "inclusion"
    assert first_box.display_label == "Inclusion"
    assert first_box.confidence == 0.802
    assert first_box.bbox_format == "xyxy"
    assert first_box.bbox_xyxy == [1840.27, 0.0, 1987.1, 1000.0]
    assert first_box.score_rank == 1
    assert first_box.is_best_prediction is True
    assert first_box.warnings == ["clamped_to_image_bounds"]

    second_box = result.detections[1]
    assert second_box.class_label == "punching_hole"
    assert second_box.display_label == "Punching hole"
    assert second_box.is_best_prediction is False


def test_build_detection_result_handles_no_detections() -> None:
    result = build_detection_result_from_yolo_result(
        result=None,
        image_width=2048,
        image_height=1000,
        class_labels=CLASS_LABELS,
    )

    assert result.status == "success"
    assert result.predicted_box_count == 0
    assert result.defect_count == 0
    assert result.detections == []
    assert result.best_detection is None
    assert result.review_status == "no_detection"
    assert result.production_ready is False
    assert result.deployment_safe is False
    assert result.traceability.checkpoint_path.endswith("weights/best.pt")


def test_build_detection_result_rejects_invalid_box_shape() -> None:
    class BadBoxes:
        xyxy = _TensorLike([[1.0, 2.0, 3.0]])
        conf = _TensorLike([0.9])
        cls = _TensorLike([0.0])

    class BadResult:
        boxes = BadBoxes()

    try:
        build_detection_result_from_yolo_result(
            result=BadResult(),
            image_width=2048,
            image_height=1000,
            class_labels=CLASS_LABELS,
        )
    except ValueError as exc:
        assert "bbox_xyxy" in str(exc)
    else:
        raise AssertionError("Expected invalid bbox shape to raise ValueError.")
