from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data/manifests/split_gc10det_detection.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/processed/gc10det_yolo"


def _repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must parse to a dictionary: {_repo_relative(path)}")
    return data


def _load_annotation(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid annotation JSON: {_repo_relative(path)}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Annotation must parse to an object: {_repo_relative(path)}")
    return data


def _validate_dimension(value: Any, field_name: str, annotation_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{field_name} must be a positive integer in {_repo_relative(annotation_path)}"
        )
    return value


def _resolve_dimensions(annotation: dict[str, Any], image_path: Path, annotation_path: Path) -> tuple[int, int]:
    size = annotation.get("size")
    if isinstance(size, dict) and "width" in size and "height" in size:
        width = _validate_dimension(size["width"], "size.width", annotation_path)
        height = _validate_dimension(size["height"], "size.height", annotation_path)
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        if (actual_width, actual_height) != (width, height):
            raise ValueError(
                f"Image size mismatch for {_repo_relative(image_path)}: "
                f"annotation size {width}x{height} vs image size {actual_width}x{actual_height}"
            )
        return width, height

    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions for {_repo_relative(image_path)}")
    return width, height


def _bbox_to_yolo(points: Any, width: int, height: int, annotation_path: Path) -> tuple[float, float, float, float]:
    if not isinstance(points, dict):
        raise ValueError(f"points must be a dictionary in {_repo_relative(annotation_path)}")

    exterior = points.get("exterior")
    if not isinstance(exterior, list) or len(exterior) != 2:
        raise ValueError(
            f"Rectangle annotations must contain exactly two exterior points: {_repo_relative(annotation_path)}"
        )

    coords: list[tuple[float, float]] = []
    for point in exterior:
        if not (
            isinstance(point, list)
            and len(point) == 2
            and not isinstance(point[0], bool)
            and not isinstance(point[1], bool)
            and isinstance(point[0], (int, float))
            and isinstance(point[1], (int, float))
        ):
            raise ValueError(
                f"Invalid exterior point in {_repo_relative(annotation_path)}: {point!r}"
            )
        coords.append((float(point[0]), float(point[1])))

    x_values = [coord[0] for coord in coords]
    y_values = [coord[1] for coord in coords]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            f"Bounding box is outside image bounds in {_repo_relative(annotation_path)}"
        )
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Bounding box has non-positive area in {_repo_relative(annotation_path)}"
        )

    x_center = ((x_min + x_max) / 2.0) / float(width)
    y_center = ((y_min + y_max) / 2.0) / float(height)
    box_width = (x_max - x_min) / float(width)
    box_height = (y_max - y_min) / float(height)

    for value, name in (
        (x_center, "x_center"),
        (y_center, "y_center"),
        (box_width, "width"),
        (box_height, "height"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Normalized {name} out of range for {_repo_relative(annotation_path)}: {value}"
            )

    return x_center, y_center, box_width, box_height


def _build_split_rows(manifest: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    entries = manifest[f"{split_name}_entries"]
    if not isinstance(entries, list):
        raise ValueError(f"{split_name}_entries must be a list")
    return entries


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _export_dataset_yaml(output_root: Path, class_labels: list[str]) -> Path:
    dataset_yaml = {
        "path": _repo_relative(output_root),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "nc": len(class_labels),
        "names": class_labels,
    }
    dataset_yaml_path = output_root / "dataset.yaml"
    _write_text(dataset_yaml_path, yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True))
    return dataset_yaml_path


def export_gc10det_yolo_dataset(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _load_yaml(manifest_path)

    class_labels = manifest.get("class_labels")
    if not isinstance(class_labels, list) or not class_labels:
        raise ValueError("Manifest class_labels must be a non-empty list.")
    class_labels = [str(label) for label in class_labels]
    class_to_index = {label: index for index, label in enumerate(class_labels)}

    split_entries = {
        "train": _build_split_rows(manifest, "train"),
        "validation": _build_split_rows(manifest, "validation"),
        "test": _build_split_rows(manifest, "test"),
    }

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_yaml_path = _export_dataset_yaml(output_root, class_labels)

    split_counts: dict[str, int] = {}
    bbox_counts_by_split: dict[str, int] = {}
    class_counts_by_split: dict[str, dict[str, int]] = {}
    empty_label_file_count = 0
    total_boxes = 0
    total_samples = 0

    for split_name, entries in split_entries.items():
        image_output_dir = output_root / "images" / split_name
        label_output_dir = output_root / "labels" / split_name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)

        split_class_counts: Counter[str] = Counter()
        split_box_count = 0

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest entry in {split_name} must be a dictionary.")

            sample_id = entry.get("sample_id")
            image_path = entry.get("image_path")
            annotation_path = entry.get("annotation_path")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Missing sample_id in {split_name}.")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"Missing image_path for {sample_id}.")
            if not isinstance(annotation_path, str) or not annotation_path:
                raise ValueError(f"Missing annotation_path for {sample_id}.")

            source_image_path = REPO_ROOT / image_path
            source_annotation_path = REPO_ROOT / annotation_path
            if not source_image_path.is_file():
                raise FileNotFoundError(f"Missing image file: {image_path}")
            if not source_annotation_path.is_file():
                raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

            annotation = _load_annotation(source_annotation_path)
            width, height = _resolve_dimensions(annotation, source_image_path, source_annotation_path)
            objects = annotation.get("objects", [])
            if not isinstance(objects, list):
                raise ValueError(f"objects must be a list in {_repo_relative(source_annotation_path)}")

            label_lines: list[str] = []
            for obj in objects:
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Each object must be a dictionary in {_repo_relative(source_annotation_path)}"
                    )
                class_title = obj.get("classTitle")
                if class_title not in class_to_index:
                    raise ValueError(
                        f"Unknown classTitle {class_title!r} in {_repo_relative(source_annotation_path)}"
                    )
                x_center, y_center, box_width, box_height = _bbox_to_yolo(
                    obj.get("points"), width, height, source_annotation_path
                )
                label_lines.append(
                    f"{class_to_index[class_title]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                )
                split_class_counts[class_title] += 1
                split_box_count += 1

            _copy_image(source_image_path, image_output_dir / source_image_path.name)
            label_path = label_output_dir / f"{source_image_path.stem}.txt"
            _write_text(label_path, "\n".join(label_lines) + ("\n" if label_lines else ""))

            if not label_lines:
                empty_label_file_count += 1

            total_samples += 1

        split_counts[split_name] = len(entries)
        bbox_counts_by_split[split_name] = split_box_count
        class_counts_by_split[split_name] = dict(sorted(split_class_counts.items()))
        total_boxes += split_box_count

    export_manifest = {
        "manifest_type": "yolo_dataset_export_manifest",
        "source_manifest_path": _repo_relative(manifest_path),
        "dataset_id": manifest["dataset_id"],
        "dataset_name": manifest.get("dataset_name", "GC10-DET"),
        "dataset_version": manifest["dataset_version"],
        "task_type": manifest["task_type"],
        "track_id": manifest["track_id"],
        "output_root": _repo_relative(output_root),
        "dataset_yaml_path": _repo_relative(dataset_yaml_path),
        "source_bbox_format": manifest.get("bbox_format"),
        "output_bbox_format": "yolo_normalized_xywh",
        "class_labels": class_labels,
        "class_to_index": class_to_index,
        "split_counts": split_counts,
        "bbox_counts_by_split": bbox_counts_by_split,
        "class_counts_by_split": class_counts_by_split,
        "empty_label_file_count": empty_label_file_count,
        "total_sample_count": total_samples,
        "total_bbox_count": total_boxes,
        "conversion_status": "success",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_validation_report": "artifacts/reports/data_validation/gc10det_validation_report.yaml",
        "source_reconciliation_report": "artifacts/reports/data_governance/gc10det_reconciliation_report.yaml",
        "known_limitations": [
            "This export boundary converts governed rectangle annotations into YOLO-normalized labels without performing model training.",
            "Images are copied into data/processed/gc10det_yolo and are intentionally left untracked by repository policy.",
        ],
    }

    export_manifest_path = output_root / "export_manifest.yaml"
    _write_text(
        export_manifest_path,
        yaml.safe_dump(export_manifest, sort_keys=False, allow_unicode=True, width=1000),
    )

    return {
        "manifest": export_manifest,
        "dataset_yaml_path": dataset_yaml_path,
        "export_manifest_path": export_manifest_path,
        "split_counts": split_counts,
        "bbox_counts_by_split": bbox_counts_by_split,
        "class_counts_by_split": class_counts_by_split,
        "empty_label_file_count": empty_label_file_count,
        "total_bbox_count": total_boxes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a governed GC10-DET YOLO dataset.")
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the governed GC10-DET split manifest.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory for the exported YOLO dataset.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = export_gc10det_yolo_dataset(Path(args.manifest_path), Path(args.output_root))

    print(f"output_root={result['manifest']['output_root']}")
    print(f"dataset_yaml_path={result['manifest']['dataset_yaml_path']}")
    print(f"export_manifest_path={_repo_relative(result['export_manifest_path'])}")
    print(f"train_count={result['split_counts']['train']}")
    print(f"validation_count={result['split_counts']['validation']}")
    print(f"test_count={result['split_counts']['test']}")
    print(f"empty_label_file_count={result['empty_label_file_count']}")
    print(f"total_bbox_count={result['total_bbox_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
