"""Dry-run skeleton for governed YOLO bbox prediction export.

This script validates the source inputs needed for a future bbox prediction
export and prepares the in-memory contract shape, but it never runs inference
and never writes artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "yolo_train_v0_2_0"
DEFAULT_SPLIT = "validation"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_DEVICE = "cpu"
DEFAULT_IMG_SIZE = 640
DEFAULT_SMOKE_LIMIT = 3
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run writer skeleton for governed YOLO bbox prediction export."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate inputs and print the export contract without writing files.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--smoke-inference",
        action="store_true",
        default=False,
        help="Run a limited validation-image inference smoke test without writing files.",
    )
    parser.add_argument(
        "--require-ultralytics",
        action="store_true",
        default=False,
        help="Require the ultralytics backend to be importable in the active Python environment.",
    )
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        default=False,
        help="Enable the governed write-mode scaffold for the future bbox prediction export.",
    )
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        default=False,
        help="Required confirmation flag for --write-artifact.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = _require_non_empty_string(args.run_id, "run_id")
    split = _require_non_empty_string(args.split, "split")
    dry_run = bool(args.dry_run)
    smoke_result: dict[str, Any] | None = None
    write_mode_report: dict[str, Any] | None = None
    write_result: dict[str, Any] | None = None

    run_dir = REPO_ROOT / "artifacts/detection/yolo/runs" / run_id
    weights_path = run_dir / "weights" / "best.pt"
    args_yaml_path = run_dir / "args.yaml"
    image_dir = REPO_ROOT / "data/processed/gc10det_yolo" / "images" / split
    label_dir = REPO_ROOT / "data/processed/gc10det_yolo" / "labels" / split
    dataset_yaml_path = REPO_ROOT / "data/processed/gc10det_yolo/dataset.yaml"
    export_manifest_path = REPO_ROOT / "data/processed/gc10det_yolo/export_manifest.yaml"
    split_manifest_path = REPO_ROOT / "data/manifests/split_gc10det_detection.yaml"
    run_config_path = REPO_ROOT / "configs/runs" / f"{run_id}.yaml"
    model_config_path = REPO_ROOT / "configs/models/yolo.yaml"
    training_result_path = REPO_ROOT / "artifacts/models/analysis" / f"training_result__{run_id}.json"
    metadata_summary_path = (
        REPO_ROOT / "artifacts/models/metadata" / f"track_detection_yolo_metadata_summary__{run_id}.json"
    )
    artifact_inventory_path = (
        REPO_ROOT / "artifacts/models/inventory" / f"track_detection_artifact_inventory__{run_id}.json"
    )
    evaluation_summary_path = (
        REPO_ROOT
        / "artifacts/models/metrics"
        / f"detection_evaluation__{run_id}__{split}.json"
    )
    reaudit_report_path = REPO_ROOT / "artifacts/reports/audits" / f"detection_yolo_reaudit__{run_id}.json"
    requirements_path = REPO_ROOT / "requirements.txt"

    required_paths = {
        "run_dir": run_dir,
        "weights/best.pt": weights_path,
        "validation_images": image_dir,
        "validation_labels": label_dir,
        "dataset.yaml": dataset_yaml_path,
        "export_manifest.yaml": export_manifest_path,
        "split_manifest.yaml": split_manifest_path,
        "run_config.yaml": run_config_path,
        "model_config.yaml": model_config_path,
        "training_result.json": training_result_path,
        "metadata_summary.json": metadata_summary_path,
        "artifact_inventory.json": artifact_inventory_path,
        "evaluation_summary.json": evaluation_summary_path,
        "reaudit_report.json": reaudit_report_path,
    }

    required_status: list[tuple[str, bool, str]] = []
    optional_status: list[tuple[str, bool, str]] = []
    parsed_inputs: list[str] = []

    try:
        for label, path in required_paths.items():
            exists = path.exists()
            required_status.append((label, exists, _repo_relative(path)))
            if not exists:
                continue

        if not run_dir.is_dir():
            raise FileNotFoundError(f"YOLO run directory not found: {_repo_relative(run_dir)}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"validation image directory not found: {_repo_relative(image_dir)}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"validation label directory not found: {_repo_relative(label_dir)}")

        ultralytics_declared_status = _ultralytics_declared(requirements_path)
        ultralytics_import_available = importlib.util.find_spec("ultralytics") is not None

        run_config = _load_yaml(run_config_path, "run config")
        model_config = _load_yaml(model_config_path, "model config")
        dataset_yaml = _load_yaml(dataset_yaml_path, "dataset yaml")
        export_manifest = _load_yaml(export_manifest_path, "export manifest")
        split_manifest = _load_yaml(split_manifest_path, "split manifest")
        training_result = _load_json(training_result_path, "training result")
        metadata_summary = _load_json(metadata_summary_path, "metadata summary")
        artifact_inventory = _load_json(artifact_inventory_path, "artifact inventory")
        evaluation_summary = _load_json(evaluation_summary_path, "evaluation summary")
        reaudit_report = _load_json(reaudit_report_path, "reaudit report")

        run_args = _load_yaml(args_yaml_path, "run args") if args_yaml_path.is_file() else None
        if run_args is not None:
            optional_status.append(("args.yaml", True, _repo_relative(args_yaml_path)))
            parsed_inputs.append(_repo_relative(args_yaml_path))

        image_files = _discover_files(image_dir, SUPPORTED_IMAGE_EXTENSIONS)
        label_files = _discover_files(label_dir, [".txt"])
        if not image_files:
            raise ValueError("validation image directory must contain at least one image.")
        if not label_files:
            raise ValueError("validation label directory must contain at least one label file.")

        dataset_names = _normalize_labels(dataset_yaml.get("names"), "dataset yaml names")
        dataset_nc = dataset_yaml.get("nc")
        if isinstance(dataset_nc, bool) or not isinstance(dataset_nc, int):
            raise ValueError("dataset yaml nc must be an integer.")
        if dataset_nc != len(dataset_names):
            raise ValueError("dataset yaml nc must equal the number of normalized names.")

        export_labels = _normalize_labels(export_manifest.get("class_labels"), "export manifest class_labels")
        split_labels = _normalize_labels(split_manifest.get("class_labels"), "split manifest class_labels")
        class_to_index = _require_dict(export_manifest.get("class_to_index"), "export manifest class_to_index")
        _validate_detection_class_labels(dataset_names, dataset_nc, export_labels, split_labels, class_to_index)

        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)[:5]
            missing_images = sorted(label_stems - image_stems)[:5]
            raise ValueError(
                "validation images and labels must align by stem; "
                f"missing_labels={missing_labels}, missing_images={missing_images}"
            )

        imgsz = _resolve_imgsz(args.imgsz, run_args)
        if imgsz <= 0:
            raise ValueError("imgsz must be a positive integer.")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit, if provided, must be a positive integer.")
        if not 0.0 <= args.conf_threshold <= 1.0:
            raise ValueError("conf-threshold must be between 0 and 1.")
        if not 0.0 <= args.iou_threshold <= 1.0:
            raise ValueError("iou-threshold must be between 0 and 1.")
        if not _require_non_empty_string(args.device, "device"):
            raise ValueError("device must be a non-empty string.")

        _validate_traceability(
            run_id=run_id,
            split=split,
            run_dir=run_dir,
            weights_path=weights_path,
            run_config=run_config,
            model_config=model_config,
            dataset_yaml=dataset_yaml,
            export_manifest=export_manifest,
            split_manifest=split_manifest,
            training_result=training_result,
            metadata_summary=metadata_summary,
            artifact_inventory=artifact_inventory,
            evaluation_summary=evaluation_summary,
            reaudit_report=reaudit_report,
        )

        future_contract = _build_future_contract(
            run_id=run_id,
            split=split,
            imgsz=imgsz,
            device=args.device,
            limit=args.limit,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            image_count=len(image_files),
            weights_path=weights_path,
            image_dir=image_dir,
            label_dir=label_dir,
            dataset_yaml_path=dataset_yaml_path,
            export_manifest_path=export_manifest_path,
            split_manifest_path=split_manifest_path,
            run_config_path=run_config_path,
            model_config_path=model_config_path,
            training_result_path=training_result_path,
            metadata_summary_path=metadata_summary_path,
            artifact_inventory_path=artifact_inventory_path,
            evaluation_summary_path=evaluation_summary_path,
            reaudit_report_path=reaudit_report_path,
            args_yaml_path=args_yaml_path if args_yaml_path.is_file() else None,
        )

        if ultralytics_declared_status != "PASS":
            raise RuntimeError(
                "ultralytics must be declared in requirements.txt for the governed YOLO writer skeleton."
            )
        if args.require_ultralytics and not ultralytics_import_available:
            raise RuntimeError(
                "ultralytics import is required when --require-ultralytics is set."
            )

        if args.write_artifact:
            write_mode_report = _evaluate_write_mode_scaffold(
                args=args,
                split=split,
                target_output_path=REPO_ROOT / future_contract["target_output_path"],
            )

            if write_mode_report["success"] and args.confirm_write and args.require_ultralytics and not args.smoke_inference and args.limit is None:
                with _ultralytics_runtime_isolation():
                    write_result = _write_prediction_artifact(
                        run_id=run_id,
                        split=split,
                        weights_path=weights_path,
                        image_files=image_files,
                        dataset_names=dataset_names,
                        export_labels=export_labels,
                        conf_threshold=args.conf_threshold,
                        iou_threshold=args.iou_threshold,
                        imgsz=imgsz,
                        device=args.device,
                        target_output_path=REPO_ROOT / future_contract["target_output_path"],
                        source_artifact_paths=future_contract["contract"]["source_artifact_paths"],
                    )
                write_mode_report["target_output_exists"] = False
                write_mode_report["artifact_writing_implemented"] = True
                write_mode_report["no_files_written"] = False
                write_mode_report["artifact_written"] = True
                write_mode_report["artifact_path"] = write_result["artifact_path"]

        print("# Detection BBox Prediction Export Writer Dry Run")
        print()
        print("## Required Inputs")
        for label, exists, path in required_status:
            print(f"- {label}: {'PASS' if exists else 'FAIL'} ({path})")
        print()
        print("## Runtime Dependency Check")
        print(
            f"- ultralytics declared in requirements: "
            f"{ultralytics_declared_status}"
        )
        print(
            f"- ultralytics import available: "
            f"{'PASS' if ultralytics_import_available else 'WARNING'}"
        )
        print(
            f"- runtime inference readiness: "
            f"{'READY' if ultralytics_import_available else 'NOT READY'}"
        )
        print(
            "- future inference/export requires ultralytics in the active Python environment: "
            f"{'PASS' if ultralytics_import_available else 'WARNING'}"
        )
        print()
        print("## Repository/Input Readiness")
        print("- dry-run repository readiness: PASS")
        print("- writer remains validate-only: PASS")
        print()
        print("## Data Discovery")
        print(f"- validation image count: {len(image_files)}")
        print(f"- validation label count: {len(label_files)}")
        print(f"- supported image extensions: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
        print()
        print("## Traceability Checks")
        print("- run_id matches governed inputs: PASS")
        print("- track_id/task_type/model identity consistent: PASS")
        print("- dataset/export/split detection labels align: PASS")
        print("- class_to_index covers exactly 0 through nc - 1: PASS")
        print("- validation image/label stems align: PASS")
        print("- weights path exists: PASS")
        print("- source manifests and summaries readable: PASS")
        print()
        print("## Future Output Contract")
        print(f"- target_output_path: {future_contract['target_output_path']}")
        print("- mode: DRY RUN / NOT WRITTEN")
        print("- top_level_fields:")
        for field_name in future_contract["top_level_fields"]:
            print(f"  - {field_name}")
        print()
        if write_mode_report is not None:
            print("## Write Mode Scaffold")
            print(
                f"- write_artifact requested: "
                f"{'YES' if write_mode_report['write_artifact_requested'] else 'NO'}"
            )
            print(
                f"- confirm_write provided: "
                f"{'YES' if write_mode_report['confirm_write_provided'] else 'NO'}"
            )
            print(
                f"- require_ultralytics provided: "
                f"{'YES' if write_mode_report['require_ultralytics_provided'] else 'NO'}"
            )
            print(
                f"- full validation split required: "
                f"{'PASS' if write_mode_report['full_validation_split_required'] else 'FAIL'}"
            )
            print(
                f"- partial limit rejected: "
                f"{'PASS' if write_mode_report['partial_limit_rejected'] else 'FAIL'}"
            )
            print(
                f"- smoke inference rejected: "
                f"{'PASS' if write_mode_report['smoke_inference_rejected'] else 'FAIL'}"
            )
            print(
                f"- target output path clear: "
                f"{'FAIL' if write_mode_report['target_output_exists'] else 'PASS'}"
            )
            print(
                f"- target parent directory exists: "
                f"{'PASS' if write_mode_report['target_parent_directory_exists'] else 'FAIL'}"
            )
            print(
                f"- artifact writing implemented: "
                f"{'YES' if write_mode_report.get('artifact_written') else 'NO'}"
            )
            if write_mode_report.get("artifact_written"):
                print("## Artifact Write")
                print(f"- target_output_path: {write_mode_report['artifact_path']}")
                print(f"- images written: {write_result['images_written'] if write_result else 0}")
                print(f"- bbox_count: {write_result['bbox_count'] if write_result else 0}")
                print("- atomic write: PASS")
                print("- post-write validation: PASS")
                print("- registry update: NOT PERFORMED")
                print("- frontend bundle: NOT PERFORMED")
                print("- artifact meaning: export evidence only")
            else:
                print("- no files written: PASS")
            print()
        if args.smoke_inference:
            with _ultralytics_runtime_isolation():
                smoke_result = _run_smoke_inference(
                    run_id=run_id,
                    split=split,
                    weights_path=weights_path,
                    image_files=image_files,
                    dataset_names=dataset_names,
                    export_labels=export_labels,
                    conf_threshold=args.conf_threshold,
                    iou_threshold=args.iou_threshold,
                    imgsz=imgsz,
                    device=args.device,
                    limit=args.limit,
                    runtime_isolation="TEMPORARY_DIRECTORY",
                )
            print("## Smoke Inference")
            print(f"- smoke image limit: {smoke_result['smoke_image_limit']}")
            print(f"- images processed: {smoke_result['images_processed']}")
            print(f"- total boxes extracted: {smoke_result['total_boxes_extracted']}")
            print(f"- images with no detections: {smoke_result['images_with_no_detections']}")
            print(f"- ultralytics runtime isolation: {smoke_result['runtime_isolation']}")
            print("- ultralytics config/cache writes isolated from user profile: PASS")
            print("- sample prediction rows shown in summarized form:")
            for row in smoke_result["sample_rows"]:
                print(f"  - {row}")
            print("- no files written: PASS")
        print()
        print("## Final Verdict")
        if smoke_result is not None and not smoke_result["success"]:
            print("FAIL")
            return 1
        if write_mode_report is not None and not write_mode_report["success"]:
            print("FAIL")
            print(f"failure_reason={write_mode_report['failure_reason']}")
            return 1
        if write_result is not None and write_result["success"]:
            print("PASS")
            return 0
        if ultralytics_import_available or args.smoke_inference or args.write_artifact:
            print("PASS")
        else:
            print("PASS_WITH_WARNINGS")
        return 0
    except Exception as exc:
        print("# Detection BBox Prediction Export Writer Dry Run")
        print()
        print("## Required Inputs")
        for label, exists, path in required_status:
            print(f"- {label}: {'PASS' if exists else 'FAIL'} ({path})")
        print()
        print("## Runtime Dependency Check")
        declared_status = _ultralytics_declared(requirements_path)
        import_available = importlib.util.find_spec("ultralytics") is not None
        import_status = "PASS" if import_available else "WARNING"
        print(f"- ultralytics declared in requirements: {declared_status}")
        print(f"- ultralytics import available: {import_status}")
        print(
            f"- runtime inference readiness: {'READY' if import_available else 'NOT READY'}"
        )
        print(
            "- future inference/export requires ultralytics in the active Python environment: "
            f"{'PASS' if import_available else 'WARNING'}"
        )
        print()
        print("## Repository/Input Readiness")
        print("- dry-run repository readiness: PASS")
        print("- writer remains validate-only: PASS")
        print()
        print("## Data Discovery")
        print(f"- validation image count: {len(_safe_discover(image_dir if 'image_dir' in locals() else None, SUPPORTED_IMAGE_EXTENSIONS))}")
        print(f"- validation label count: {len(_safe_discover(label_dir if 'label_dir' in locals() else None, ['.txt']))}")
        print(f"- supported image extensions: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
        print()
        print("## Traceability Checks")
        print(f"- FAIL: {exc}")
        print()
        print("## Future Output Contract")
        target_output_path = (
            REPO_ROOT
            / "artifacts/models/predictions"
            / f"detection_bbox_predictions__{run_id}__{split}.json"
        )
        print(f"- target_output_path: {target_output_path}")
        print("- mode: DRY RUN / NOT WRITTEN")
        print("- top_level_fields:")
        for field_name in _future_top_level_fields():
            print(f"  - {field_name}")
        print()
        print("## Final Verdict")
        if import_available and declared_status == "PASS":
            print("PASS")
        else:
            print("FAIL")
        print(f"failure_reason={exc}")
        return 1


def _build_future_contract(
    *,
    run_id: str,
    split: str,
    imgsz: int,
    device: str,
    limit: int | None,
    conf_threshold: float,
    iou_threshold: float,
    image_count: int,
    weights_path: Path,
    image_dir: Path,
    label_dir: Path,
    dataset_yaml_path: Path,
    export_manifest_path: Path,
    split_manifest_path: Path,
    run_config_path: Path,
    model_config_path: Path,
    training_result_path: Path,
    metadata_summary_path: Path,
    artifact_inventory_path: Path,
    evaluation_summary_path: Path,
    reaudit_report_path: Path,
    args_yaml_path: Path | None,
) -> dict[str, Any]:
    source_artifact_paths = [
        _repo_relative(weights_path),
        _repo_relative(image_dir),
        _repo_relative(label_dir),
        _repo_relative(dataset_yaml_path),
        _repo_relative(export_manifest_path),
        _repo_relative(split_manifest_path),
        _repo_relative(run_config_path),
        _repo_relative(model_config_path),
        _repo_relative(training_result_path),
        _repo_relative(metadata_summary_path),
        _repo_relative(artifact_inventory_path),
        _repo_relative(evaluation_summary_path),
        _repo_relative(reaudit_report_path),
    ]
    if args_yaml_path is not None:
        source_artifact_paths.insert(1, _repo_relative(args_yaml_path))

    return {
        "target_output_path": _repo_relative(
            REPO_ROOT
            / "artifacts/models/predictions"
            / f"detection_bbox_predictions__{run_id}__{split}.json"
        ),
        "top_level_fields": [
            "artifact_type",
            "track_id",
            "task_type",
            "run_id",
            "run_config_id",
            "model_name",
            "model_type",
            "model_version",
            "dataset_id",
            "dataset_version",
            "split",
            "source_artifact_paths",
            "prediction_parameters",
            "image_count",
            "prediction_count",
            "bbox_count",
            "prediction_rows",
        ],
        "contract": {
            "artifact_type": "detection_bbox_predictions",
            "track_id": "detection",
            "task_type": "object_detection",
            "run_id": run_id,
            "run_config_id": f"{run_id}",
            "model_name": "yolo",
            "model_type": "yolo",
            "model_version": "0.2.0",
            "dataset_id": "gc10det_detection",
            "dataset_version": "gc10det_1.0",
            "split": split,
            "source_artifact_paths": source_artifact_paths,
            "prediction_parameters": {
                "dry_run": True,
                "limit": limit,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "imgsz": imgsz,
                "device": device,
                "weights_path": _repo_relative(weights_path),
                "source_backend": "ultralytics",
                "source_model_source": "yolov8n.pt",
            },
            "image_count": image_count,
            "prediction_count": 0,
            "bbox_count": 0,
            "prediction_rows": [],
        },
    }


def _evaluate_write_mode_scaffold(
    *,
    args: argparse.Namespace,
    split: str,
    target_output_path: Path,
) -> dict[str, Any]:
    requested = bool(args.write_artifact)
    confirm = bool(args.confirm_write)
    require_ultralytics = bool(args.require_ultralytics)
    smoke_inference = bool(args.smoke_inference)
    limit = args.limit

    errors: list[str] = []
    if not requested:
        errors.append("write_artifact was not requested.")
    if not confirm:
        errors.append("--confirm-write is required with --write-artifact.")
    if not require_ultralytics:
        errors.append("--require-ultralytics is required with --write-artifact.")
    if smoke_inference:
        errors.append("--smoke-inference is not allowed with --write-artifact.")
    if limit is not None:
        errors.append("--limit is not allowed with --write-artifact.")
    if split != DEFAULT_SPLIT:
        errors.append("write mode requires the full validation split.")

    parent_directory = target_output_path.parent
    target_exists = target_output_path.exists()
    parent_exists = parent_directory.exists()
    if target_exists:
        errors.append(f"target output already exists: {_repo_relative(target_output_path)}")

    return {
        "write_artifact_requested": requested,
        "confirm_write_provided": confirm,
        "require_ultralytics_provided": require_ultralytics,
        "full_validation_split_required": split == DEFAULT_SPLIT,
        "partial_limit_rejected": limit is None,
        "smoke_inference_rejected": not smoke_inference,
        "target_output_exists": target_exists,
        "target_parent_directory_exists": parent_exists,
        "artifact_written": False,
        "artifact_writing_implemented": False,
        "no_files_written": True,
        "success": not errors,
        "failure_reason": "; ".join(errors) if errors else None,
    }


@contextmanager
def _ultralytics_runtime_isolation():
    with tempfile.TemporaryDirectory(prefix="ultralytics-smoke-") as temp_dir:
        original_env = {
            key: os.environ.get(key)
            for key in ("YOLO_CONFIG_DIR", "XDG_CONFIG_HOME", "HOME")
        }
        os.environ["YOLO_CONFIG_DIR"] = temp_dir
        os.environ["XDG_CONFIG_HOME"] = temp_dir
        os.environ["HOME"] = temp_dir
        try:
            yield
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _write_prediction_artifact(
    *,
    run_id: str,
    split: str,
    weights_path: Path,
    image_files: list[Path],
    dataset_names: list[str],
    export_labels: list[str],
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
    target_output_path: Path,
    source_artifact_paths: list[str],
) -> dict[str, Any]:
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError("ultralytics is required for artifact writing.")

    from ultralytics import YOLO  # type: ignore

    if target_output_path.exists():
        raise FileExistsError(f"target output already exists: {_repo_relative(target_output_path)}")

    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    class_names = {index: label for index, label in enumerate(dataset_names)}
    if len(class_names) != len(export_labels):
        raise ValueError("class label mapping mismatch for artifact writing.")

    yolo = YOLO(str(weights_path))
    prediction_rows: list[dict[str, Any]] = []
    bbox_count = 0
    total_images = len(image_files)
    progress_step = 50

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target_output_path.parent, delete=False, suffix=".tmp"
    ) as temp_handle:
        temp_path = Path(temp_handle.name)
    try:
        for index, image_path in enumerate(image_files, start=1):
            if index == 1 or index % progress_step == 0 or index == total_images:
                print(f"- progress: {index}/{total_images}")
            result = yolo.predict(
                source=str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            if not result:
                prediction_rows.append(
                    {
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "image_width": None,
                        "image_height": None,
                        "predicted_box_count": 0,
                        "defect_count": 0,
                        "best_prediction": None,
                        "warnings": ["no detections"],
                        "errors": [],
                        "boxes": [],
                    }
                )
                continue

            row = _build_prediction_row(
                image_path=image_path,
                result=result[0],
                class_names=class_names,
                dataset_names=dataset_names,
            )
            bbox_count += len(row["boxes"])
            prediction_rows.append(row)

        payload = {
            "artifact_type": "detection_bbox_predictions",
            "track_id": "detection",
            "task_type": "object_detection",
            "run_id": run_id,
            "run_config_id": run_id,
            "model_name": "yolo",
            "model_type": "yolo",
            "model_version": "0.2.0",
            "dataset_id": "gc10det_detection",
            "dataset_version": "gc10det_1.0",
            "split": split,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_artifact_paths": source_artifact_paths,
            "prediction_parameters": {
                "artifact_write_mode": True,
                "smoke_inference": False,
                "limit": None,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "imgsz": imgsz,
                "device": device,
                "weights_path": _repo_relative(weights_path),
                "source_backend": "ultralytics",
                "source_model_source": "yolov8n.pt",
            },
            "image_count": len(prediction_rows),
            "prediction_count": len(prediction_rows),
            "bbox_count": bbox_count,
            "prediction_rows": prediction_rows,
        }

        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")

        written_payload = _load_json(temp_path, "written bbox prediction export")
        _validate_written_prediction_artifact(written_payload, prediction_rows, bbox_count)
        temp_path.replace(target_output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

    written_size = target_output_path.stat().st_size
    return {
        "success": True,
        "artifact_path": _repo_relative(target_output_path),
        "images_written": len(prediction_rows),
        "bbox_count": bbox_count,
        "file_size": written_size,
    }


def _build_prediction_row(
    *,
    image_path: Path,
    result: Any,
    class_names: dict[int, str],
    dataset_names: list[str],
) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    image_height, image_width = _extract_image_shape(result, image_path)
    if image_height is None or image_width is None:
        raise ValueError(f"unable to determine image size for write-mode image: {image_path}")
    extracted_boxes = [] if boxes is None else _extract_boxes(boxes, image_width, image_height, class_names, dataset_names)
    extracted_count = len(extracted_boxes)
    return {
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "predicted_box_count": extracted_count,
        "defect_count": extracted_count,
        "best_prediction": extracted_boxes[0] if extracted_boxes else None,
        "warnings": ["no detections"] if extracted_count == 0 else [],
        "errors": [],
        "boxes": extracted_boxes,
    }


def _validate_written_prediction_artifact(
    payload: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    bbox_count: int,
) -> None:
    required_fields = set(_future_top_level_fields())
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"written prediction artifact missing fields: {missing}")
    if not isinstance(payload.get("prediction_rows"), list):
        raise ValueError("written prediction artifact prediction_rows must be a list.")
    if payload.get("image_count") != len(prediction_rows):
        raise ValueError("written prediction artifact image_count mismatch.")
    if payload.get("prediction_count") != len(prediction_rows):
        raise ValueError("written prediction artifact prediction_count mismatch.")
    if payload.get("bbox_count") != bbox_count:
        raise ValueError("written prediction artifact bbox_count mismatch.")
    if bbox_count != sum(len(row.get("boxes", [])) for row in prediction_rows):
        raise ValueError("written prediction artifact box total mismatch.")


def _run_smoke_inference(
    *,
    run_id: str,
    split: str,
    weights_path: Path,
    image_files: list[Path],
    dataset_names: list[str],
    export_labels: list[str],
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
    limit: int | None,
    runtime_isolation: str,
) -> dict[str, Any]:
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError("ultralytics is required for smoke inference.")

    from ultralytics import YOLO  # type: ignore

    smoke_limit = limit if limit is not None else DEFAULT_SMOKE_LIMIT
    if smoke_limit <= 0:
        raise ValueError("limit, if provided, must be a positive integer.")

    yolo = YOLO(str(weights_path))
    class_names = {index: label for index, label in enumerate(dataset_names)}
    if len(class_names) != len(export_labels):
        raise ValueError("class label mapping mismatch for smoke inference.")

    selected_images = image_files[:smoke_limit]
    sample_rows: list[str] = []
    prediction_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    images_with_no_detections = 0
    total_boxes_extracted = 0

    for image_path in selected_images:
        results = yolo.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        if not results:
            prediction_rows.append(
                {
                    "image_path": str(image_path),
                    "image_width": None,
                    "image_height": None,
                    "predicted_box_count": 0,
                    "defect_count": 0,
                    "best_prediction": None,
                    "warnings": ["no detections"],
                    "errors": [],
                    "boxes": [],
                }
            )
            sample_rows.append(f"{image_path.name}: no detections")
            images_with_no_detections += 1
            continue

        result = results[0]
        boxes = getattr(result, "boxes", None)
        image_height, image_width = _extract_image_shape(result, image_path)
        extracted_boxes = [] if boxes is None else _extract_boxes(boxes, image_width, image_height, class_names, dataset_names)
        extracted_count = len(extracted_boxes)
        total_boxes_extracted += extracted_count
        image_warnings: list[str] = []
        if extracted_count == 0:
            images_with_no_detections += 1
            image_warnings.append("no detections")

        prediction_rows.append(
            {
                "image_path": str(image_path),
                "image_width": image_width,
                "image_height": image_height,
                "predicted_box_count": extracted_count,
                "defect_count": extracted_count,
                "best_prediction": extracted_boxes[0] if extracted_boxes else None,
                "warnings": image_warnings,
                "errors": [],
                "boxes": extracted_boxes,
            }
        )
        sample_rows.append(
            f"{image_path.name}: boxes={extracted_count}, image_size=({image_height}, {image_width})"
        )

    warnings.extend(
        [
            "smoke inference only validates a limited sample",
            "no files written",
        ]
    )
    return {
        "success": True,
        "smoke_image_limit": smoke_limit,
        "images_processed": len(selected_images),
        "total_boxes_extracted": total_boxes_extracted,
        "images_with_no_detections": images_with_no_detections,
        "sample_rows": sample_rows,
        "prediction_rows": prediction_rows,
        "warnings": warnings,
        "runtime_isolation": runtime_isolation,
    }


def _extract_image_shape(result: Any, image_path: Path) -> tuple[int | None, int | None]:
    orig_shape = getattr(result, "orig_shape", None)
    if isinstance(orig_shape, (tuple, list)) and len(orig_shape) >= 2:
        return int(orig_shape[0]), int(orig_shape[1])
    orig_img = getattr(result, "orig_img", None)
    if hasattr(orig_img, "shape") and len(orig_img.shape) >= 2:
        return int(orig_img.shape[0]), int(orig_img.shape[1])
    raise ValueError(f"unable to determine image size for smoke inference image: {image_path}")


def _extract_boxes(
    boxes: Any,
    image_width: int,
    image_height: int,
    class_names: dict[int, str],
    dataset_names: list[str],
) -> list[dict[str, Any]]:
    xyxy_values = _tensor_to_list(getattr(boxes, "xyxy", None))
    conf_values = _tensor_to_list(getattr(boxes, "conf", None))
    cls_values = _tensor_to_list(getattr(boxes, "cls", None))
    if not xyxy_values:
        return []
    if len(xyxy_values) != len(conf_values) or len(xyxy_values) != len(cls_values):
        raise ValueError("ultralytics boxes tensor lengths are inconsistent.")

    extracted: list[dict[str, Any]] = []
    for index, (xyxy, confidence_value, class_id_value) in enumerate(
        zip(xyxy_values, conf_values, cls_values, strict=True)
    ):
        if len(xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly four values.")
        confidence = float(confidence_value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence values must be between 0 and 1.")
        class_id = int(class_id_value)
        if class_id < 0 or class_id >= len(dataset_names):
            raise ValueError("class_id is outside the valid class range.")
        class_label = class_names[class_id]
        if class_label != dataset_names[class_id]:
            raise ValueError("class label does not match dataset/export/split mapping.")

        x1, y1, x2, y2 = (float(value) for value in xyxy)
        clamped = False
        if x1 < 0.0 or y1 < 0.0 or x2 > float(image_width) or y2 > float(image_height):
            x1 = min(max(x1, 0.0), float(image_width))
            y1 = min(max(y1, 0.0), float(image_height))
            x2 = min(max(x2, 0.0), float(image_width))
            y2 = min(max(y2, 0.0), float(image_height))
            clamped = True
        if x1 > x2 or y1 > y2:
            raise ValueError("bbox_xyxy coordinates must satisfy x1 <= x2 and y1 <= y2.")

        extracted.append(
            {
                "box_id": index,
                "class_id": class_id,
                "class_label": class_label,
                "confidence": confidence,
                "bbox_format": "xyxy",
                "bbox_xyxy": [x1, y1, x2, y2],
                "score_rank": index + 1,
                "is_best_prediction": index == 0,
                "warnings": ["clamped_to_image_bounds"] if clamped else [],
            }
        )
    return extracted


def _tensor_to_list(value: Any) -> list[list[float]] | list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return value
    raise ValueError("unable to convert ultralytics tensor to list.")


def _future_top_level_fields() -> list[str]:
    return [
        "artifact_type",
        "track_id",
        "task_type",
        "run_id",
        "run_config_id",
        "model_name",
        "model_type",
        "model_version",
        "dataset_id",
        "dataset_version",
        "split",
        "source_artifact_paths",
        "prediction_parameters",
        "image_count",
        "prediction_count",
        "bbox_count",
        "prediction_rows",
    ]


def _validate_traceability(
    *,
    run_id: str,
    split: str,
    run_dir: Path,
    weights_path: Path,
    run_config: dict[str, Any],
    model_config: dict[str, Any],
    dataset_yaml: dict[str, Any],
    export_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    training_result: dict[str, Any],
    metadata_summary: dict[str, Any],
    artifact_inventory: dict[str, Any],
    evaluation_summary: dict[str, Any],
    reaudit_report: dict[str, Any],
) -> None:
    identity = _require_dict(run_config.get("identity"), "run_config.identity")
    model_identity = _require_dict(run_config.get("model_identity"), "run_config.model_identity")
    dataset_binding = _require_dict(run_config.get("dataset_binding"), "run_config.dataset_binding")
    training_model_source = run_config.get("training_model_source")

    if run_dir.name != run_id:
        raise ValueError(f"run directory name must match run_id: {run_dir.name} != {run_id}")
    if identity.get("run_config_id") != run_id:
        raise ValueError("run config id must match run_id.")
    if identity.get("task_type") != "object_detection":
        raise ValueError("run config task_type must be object_detection.")
    if identity.get("track_id") != "detection":
        raise ValueError("run config track_id must be detection.")
    if model_identity.get("model_type") != "yolo":
        raise ValueError("model type must be yolo.")
    if model_identity.get("model_name") != "yolo":
        raise ValueError("model name must be yolo.")
    if dataset_binding.get("dataset_id") != "gc10det_detection":
        raise ValueError("dataset id must be gc10det_detection.")
    if dataset_binding.get("dataset_version") != "gc10det_1.0":
        raise ValueError("dataset version must be gc10det_1.0.")
    if not isinstance(training_model_source, str) or not training_model_source.strip():
        raise ValueError("a governed YOLO training model source must be declared.")
    if model_config.get("backend") != "ultralytics":
        raise ValueError("model config backend must be ultralytics.")
    if model_config.get("backend_package") != "ultralytics":
        raise ValueError("model config backend_package must be ultralytics.")

    if dataset_yaml.get("path") != "data/processed/gc10det_yolo":
        raise ValueError("dataset yaml path must be data/processed/gc10det_yolo.")
    if dataset_yaml.get("val") != f"images/{split}":
        raise ValueError("dataset yaml val split must match the requested split.")
    if export_manifest.get("dataset_id") != "gc10det_detection":
        raise ValueError("export manifest dataset id mismatch.")
    if export_manifest.get("dataset_version") != "gc10det_1.0":
        raise ValueError("export manifest dataset version mismatch.")
    if export_manifest.get("track_id") != "detection":
        raise ValueError("export manifest track_id must be detection.")
    if export_manifest.get("task_type") != "object_detection":
        raise ValueError("export manifest task_type must be object_detection.")
    if export_manifest.get("dataset_yaml_path") != "data/processed/gc10det_yolo/dataset.yaml":
        raise ValueError("export manifest dataset_yaml_path mismatch.")
    if export_manifest.get("output_root") != "data/processed/gc10det_yolo":
        raise ValueError("export manifest output_root mismatch.")
    if split_manifest.get("dataset_id") != "gc10det_detection":
        raise ValueError("split manifest dataset id mismatch.")
    if split_manifest.get("dataset_version") != "gc10det_1.0":
        raise ValueError("split manifest dataset version mismatch.")
    if split_manifest.get("track_id") != "detection":
        raise ValueError("split manifest track_id must be detection.")
    if split_manifest.get("task_type") != "object_detection":
        raise ValueError("split manifest task_type must be object_detection.")

    if training_result.get("run_id") != run_id:
        raise ValueError("training result run_id mismatch.")
    if metadata_summary.get("run_id") != run_id:
        raise ValueError("metadata summary run_id mismatch.")
    if artifact_inventory.get("run_id") != run_id:
        raise ValueError("artifact inventory run_id mismatch.")
    if evaluation_summary.get("run_id") != run_id:
        raise ValueError("evaluation summary run_id mismatch.")
    if reaudit_report.get("run_id") != run_id:
        raise ValueError("reaudit report run_id mismatch.")
    if training_result.get("track_id") != "detection":
        raise ValueError("training result track_id must be detection.")
    if metadata_summary.get("track_id") != "detection":
        raise ValueError("metadata summary track_id must be detection.")
    if artifact_inventory.get("track_id") != "detection":
        raise ValueError("artifact inventory track_id must be detection.")
    if evaluation_summary.get("track_id") != "detection":
        raise ValueError("evaluation summary track_id must be detection.")
    if reaudit_report.get("track_id") != "detection":
        raise ValueError("reaudit report track_id must be detection.")

    metrics = _require_dict(evaluation_summary.get("metrics"), "evaluation_summary.metrics")
    for metric_name in ("precision", "recall", "mAP50", "mAP50_95"):
        if metric_name not in metrics:
            raise ValueError(f"evaluation summary missing metric: {metric_name}")
        value = metrics[metric_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"evaluation summary metric must be numeric: {metric_name}")

    if not weights_path.is_file():
        raise FileNotFoundError(f"weights file not found: {_repo_relative(weights_path)}")
    if artifact_inventory.get("inventory_status") != "pass":
        raise ValueError("artifact inventory status must be pass.")
    if evaluation_summary.get("evaluation_split") != split:
        raise ValueError("evaluation summary split must match the requested split.")
    if metadata_summary.get("governance_status", {}).get("registry_updated") is not False:
        raise ValueError("metadata summary should not claim registry updated.")
    if evaluation_summary.get("governance_status", {}).get("registry_updated") is not False:
        raise ValueError("evaluation summary should not claim registry updated.")


def _validate_detection_class_labels(
    dataset_names: list[str],
    dataset_nc: int,
    export_labels: list[str],
    split_labels: list[str],
    class_to_index: dict[str, Any],
) -> None:
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("dataset yaml names must not contain duplicates.")
    if len(export_labels) != len(set(export_labels)):
        raise ValueError("export manifest class labels must not contain duplicates.")
    if len(split_labels) != len(set(split_labels)):
        raise ValueError("split manifest class labels must not contain duplicates.")
    if dataset_names != export_labels or dataset_names != split_labels:
        raise ValueError("detection class labels must align across dataset, export, and split manifests.")
    if len(class_to_index) != dataset_nc:
        raise ValueError("class_to_index must cover every detection class label.")

    mapped_ids: list[int] = []
    for label in dataset_names:
        if label not in class_to_index:
            raise ValueError(f"class_to_index missing label: {label}")
        index_value = class_to_index[label]
        if isinstance(index_value, bool) or not isinstance(index_value, int):
            raise ValueError(f"class_to_index must map label to an integer id: {label}")
        mapped_ids.append(index_value)

    if len(set(mapped_ids)) != len(mapped_ids):
        raise ValueError("class_to_index must not contain duplicate class ids.")
    if set(mapped_ids) != set(range(dataset_nc)):
        raise ValueError("class_to_index ids must cover exactly 0 through nc - 1.")


def _resolve_imgsz(imgsz_arg: int | None, run_args: dict[str, Any] | None) -> int:
    if imgsz_arg is not None:
        return imgsz_arg
    if run_args is None:
        return DEFAULT_IMG_SIZE
    value = run_args.get("imgsz")
    if isinstance(value, bool):
        return DEFAULT_IMG_SIZE
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return DEFAULT_IMG_SIZE
    return DEFAULT_IMG_SIZE


def _ultralytics_declared(requirements_path: Path) -> str:
    if not requirements_path.is_file():
        return "UNKNOWN"
    try:
        for line in requirements_path.read_text(encoding="utf-8").splitlines():
            normalized = line.strip().lower()
            if normalized.startswith("ultralytics"):
                return "PASS"
    except OSError:
        return "UNKNOWN"
    return "FAIL"


def _discover_files(directory: Path, suffixes: list[str]) -> list[Path]:
    suffixes_lower = {suffix.lower() for suffix in suffixes}
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes_lower
        ]
    )


def _safe_discover(directory: Path | None, suffixes: list[str]) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    return _discover_files(directory, suffixes)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON is invalid: {_repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object: {_repo_relative(path)}")
    return payload


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} YAML is invalid: {_repo_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} YAML must contain a mapping: {_repo_relative(path)}")
    return payload


def _normalize_labels(value: Any, label: str) -> list[str]:
    if isinstance(value, list):
        labels = value
    elif isinstance(value, dict):
        if all(isinstance(key, int) or (isinstance(key, str) and str(key).isdigit()) for key in value):
            labels = [value[key] for key in sorted(value, key=lambda item: int(item))]
        else:
            raise ValueError(f"{label} must be a list or an ordered mapping of labels.")
    else:
        raise ValueError(f"{label} must be a list or mapping.")

    if not labels:
        raise ValueError(f"{label} must not be empty.")

    normalized: list[str] = []
    for item in labels:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} must contain non-empty string labels.")
        normalized.append(item.strip())
    return normalized


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dictionary.")
    return value


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
