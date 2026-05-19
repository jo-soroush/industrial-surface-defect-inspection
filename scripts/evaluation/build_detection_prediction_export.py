"""Validate governed YOLO prediction-export inputs without writing outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "yolo_train_v0_2_0"
DEFAULT_RUN_DIR = REPO_ROOT / "artifacts/detection/yolo/runs/yolo_train_v0_2_0"
DEFAULT_RUN_CONFIG = REPO_ROOT / "configs/runs/yolo_train_v0_2_0.yaml"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs/models/yolo.yaml"
DEFAULT_DATASET_YAML = REPO_ROOT / "data/processed/gc10det_yolo/dataset.yaml"
DEFAULT_EXPORT_MANIFEST = REPO_ROOT / "data/processed/gc10det_yolo/export_manifest.yaml"
DEFAULT_SPLIT_MANIFEST = REPO_ROOT / "data/manifests/split_gc10det_detection.yaml"
DEFAULT_TRAINING_RESULT = REPO_ROOT / "artifacts/models/analysis/training_result__yolo_train_v0_2_0.json"
DEFAULT_METADATA_SUMMARY = REPO_ROOT / "artifacts/models/metadata/track_detection_yolo_metadata_summary__yolo_train_v0_2_0.json"
DEFAULT_ARTIFACT_INVENTORY = REPO_ROOT / "artifacts/models/inventory/track_detection_artifact_inventory__yolo_train_v0_2_0.json"
DEFAULT_EVALUATION_SUMMARY = REPO_ROOT / "artifacts/models/metrics/detection_evaluation__yolo_train_v0_2_0__validation.json"
DEFAULT_REAUDIT_REPORT = REPO_ROOT / "artifacts/reports/audits/detection_yolo_reaudit__yolo_train_v0_2_0.json"


OPTIONAL_VISUAL_FILES = [
    "val_batch0_pred.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_pred.jpg",
    "val_batch0_labels.jpg",
    "val_batch1_labels.jpg",
    "val_batch2_labels.jpg",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "results.png",
    "labels.jpg",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate governed YOLO prediction-export inputs without writing outputs."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--run-config", default=str(DEFAULT_RUN_CONFIG))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--dataset-yaml", default=str(DEFAULT_DATASET_YAML))
    parser.add_argument("--export-manifest", default=str(DEFAULT_EXPORT_MANIFEST))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--training-result", default=str(DEFAULT_TRAINING_RESULT))
    parser.add_argument("--metadata-summary", default=str(DEFAULT_METADATA_SUMMARY))
    parser.add_argument("--artifact-inventory", default=str(DEFAULT_ARTIFACT_INVENTORY))
    parser.add_argument("--evaluation-summary", default=str(DEFAULT_EVALUATION_SUMMARY))
    parser.add_argument("--reaudit-report", default=str(DEFAULT_REAUDIT_REPORT))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = _require_non_empty_string(args.run_id, "run_id")
    run_dir = Path(args.run_dir)
    required_paths = {
        "run_dir": run_dir,
        "weights/best.pt": run_dir / "weights" / "best.pt",
        "results.csv": run_dir / "results.csv",
        "args.yaml": run_dir / "args.yaml",
        "data/processed/gc10det_yolo/dataset.yaml": Path(args.dataset_yaml),
        "data/processed/gc10det_yolo/export_manifest.yaml": Path(args.export_manifest),
        "data/manifests/split_gc10det_detection.yaml": Path(args.split_manifest),
        "configs/runs/yolo_train_v0_2_0.yaml": Path(args.run_config),
        "configs/models/yolo.yaml": Path(args.model_config),
        "artifacts/models/analysis/training_result__yolo_train_v0_2_0.json": Path(args.training_result),
        "artifacts/models/metadata/track_detection_yolo_metadata_summary__yolo_train_v0_2_0.json": Path(args.metadata_summary),
        "artifacts/models/inventory/track_detection_artifact_inventory__yolo_train_v0_2_0.json": Path(args.artifact_inventory),
        "artifacts/models/metrics/detection_evaluation__yolo_train_v0_2_0__validation.json": Path(args.evaluation_summary),
        "artifacts/reports/audits/detection_yolo_reaudit__yolo_train_v0_2_0.json": Path(args.reaudit_report),
    }

    parsed_inputs: list[str] = []
    required_status: list[tuple[str, bool, str]] = []
    optional_status: list[tuple[str, bool, str]] = []

    try:
        for label, path in required_paths.items():
            exists = path.exists()
            required_status.append((label, exists, _repo_relative(path)))
            if not exists:
                continue
        if not run_dir.is_dir():
            required_status[0] = ("run_dir", False, _repo_relative(run_dir))
            raise FileNotFoundError(f"YOLO run directory not found: {_repo_relative(run_dir)}")

        run_config = _load_yaml(Path(args.run_config), "run config")
        model_config = _load_yaml(Path(args.model_config), "model config")
        dataset_yaml = _load_yaml(Path(args.dataset_yaml), "dataset yaml")
        export_manifest = _load_yaml(Path(args.export_manifest), "export manifest")
        split_manifest = _load_yaml(Path(args.split_manifest), "split manifest")
        training_result = _load_json(Path(args.training_result), "training result")
        metadata_summary = _load_json(Path(args.metadata_summary), "metadata summary")
        artifact_inventory = _load_json(Path(args.artifact_inventory), "artifact inventory")
        evaluation_summary = _load_json(Path(args.evaluation_summary), "evaluation summary")
        reaudit_report = _load_json(Path(args.reaudit_report), "reaudit report")
        args_yaml = _load_yaml(run_dir / "args.yaml", "run args")
        results_csv_rows = _load_csv(run_dir / "results.csv", "results.csv")

        parsed_inputs.extend(
            [
                _repo_relative(Path(args.run_config)),
                _repo_relative(Path(args.model_config)),
                _repo_relative(Path(args.dataset_yaml)),
                _repo_relative(Path(args.export_manifest)),
                _repo_relative(Path(args.split_manifest)),
                _repo_relative(Path(args.training_result)),
                _repo_relative(Path(args.metadata_summary)),
                _repo_relative(Path(args.artifact_inventory)),
                _repo_relative(Path(args.evaluation_summary)),
                _repo_relative(Path(args.reaudit_report)),
                _repo_relative(run_dir / "args.yaml"),
                _repo_relative(run_dir / "results.csv"),
            ]
        )

        _validate_traceability(
            run_id=run_id,
            run_dir=run_dir,
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
            args_yaml=args_yaml,
            results_csv_rows=results_csv_rows,
        )

        for file_name in OPTIONAL_VISUAL_FILES:
            optional_path = run_dir / file_name
            optional_status.append((file_name, optional_path.exists(), _repo_relative(optional_path)))

        print("# Detection Prediction Export Preflight")
        print()
        print("## Required Inputs")
        for label, exists, path in required_status:
            print(f"- {label}: {'PASS' if exists else 'FAIL'} ({path})")
        print()
        print("## Optional Visual Inputs")
        for label, exists, path in optional_status:
            print(f"- {label}: {'PRESENT' if exists else 'MISSING'} ({path})")
        print()
        print("## Parsed Evidence")
        for item in parsed_inputs:
            print(f"- parsed: {item}")
        print()
        print("## Traceability Checks")
        print("- run_id matches governed inputs: PASS")
        print("- model type matches governed inputs: PASS")
        print("- evaluation metrics include precision/recall/mAP50/mAP50-95: PASS")
        print("- detection class labels align across dataset/export/split manifests: PASS")
        print("- metadata summary readable: PASS")
        print("- artifact inventory readable: PASS")
        print("- audit report readable: PASS")
        print()
        print("## Future Output Contract")
        for future_path in _future_output_paths(run_id):
            print(f"- {future_path} (NOT WRITTEN)")
        print()
        print("## Final Verdict")
        print("PASS")
        return 0
    except Exception as exc:
        print("# Detection Prediction Export Preflight")
        print()
        print("## Required Inputs")
        for label, exists, path in required_status:
            print(f"- {label}: {'PASS' if exists else 'FAIL'} ({path})")
        print()
        print("## Optional Visual Inputs")
        for label, exists, path in optional_status:
            print(f"- {label}: {'PRESENT' if exists else 'MISSING'} ({path})")
        print()
        print("## Parsed Evidence")
        for item in parsed_inputs:
            print(f"- parsed: {item}")
        print()
        print("## Traceability Checks")
        print(f"- FAIL: {exc}")
        print()
        print("## Future Output Contract")
        for future_path in _future_output_paths(run_id):
            print(f"- {future_path} (NOT WRITTEN)")
        print()
        print("## Final Verdict")
        print("FAIL")
        print(f"failure_reason={exc}")
        return 1


def _validate_traceability(
    *,
    run_id: str,
    run_dir: Path,
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
    args_yaml: dict[str, Any],
    results_csv_rows: list[dict[str, str]],
) -> None:
    identity = _require_dict(run_config.get("identity"), "run_config.identity")
    model_identity = _require_dict(run_config.get("model_identity"), "run_config.model_identity")
    dataset_binding = _require_dict(run_config.get("dataset_binding"), "run_config.dataset_binding")

    if run_dir.name != run_id:
        raise ValueError(f"run directory name must match run_id: {run_dir.name} != {run_id}")
    if identity.get("run_config_id") != "yolo_train_v0_2_0":
        raise ValueError("run config id must be yolo_train_v0_2_0.")
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

    if run_config.get("training_model_source") != "yolov8n.pt":
        raise ValueError("training_model_source must be yolov8n.pt.")
    if model_config.get("backend") != "ultralytics":
        raise ValueError("model config backend must be ultralytics.")
    if model_config.get("backend_package") != "ultralytics":
        raise ValueError("model config backend_package must be ultralytics.")

    if dataset_yaml.get("nc") != 10:
        raise ValueError("dataset yaml nc must be 10.")
    _validate_detection_class_labels(dataset_yaml, export_manifest, split_manifest)
    if export_manifest.get("dataset_id") != "gc10det_detection":
        raise ValueError("export manifest dataset id mismatch.")
    if export_manifest.get("dataset_version") != "gc10det_1.0":
        raise ValueError("export manifest dataset version mismatch.")
    if split_manifest.get("dataset_id") != "gc10det_detection":
        raise ValueError("split manifest dataset id mismatch.")
    if split_manifest.get("dataset_version") != "gc10det_1.0":
        raise ValueError("split manifest dataset version mismatch.")

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

    if len(results_csv_rows) == 0:
        raise ValueError("results.csv must contain at least one data row.")

    if args_has_missing_values(args_yaml):
        raise ValueError("run args yaml contains missing required values.")

    if artifact_inventory.get("inventory_status") != "pass":
        raise ValueError("artifact inventory status must be pass.")
    if metadata_summary.get("governance_status", {}).get("registry_updated") is not False:
        raise ValueError("metadata summary should not claim registry updated.")
    if evaluation_summary.get("governance_status", {}).get("registry_updated") is not False:
        raise ValueError("evaluation summary should not claim registry updated.")


def _validate_detection_class_labels(
    dataset_yaml: dict[str, Any],
    export_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> None:
    dataset_names = _normalize_detection_class_labels(dataset_yaml.get("names"), "dataset yaml names")
    dataset_nc = dataset_yaml.get("nc")
    if isinstance(dataset_nc, bool) or not isinstance(dataset_nc, int):
        raise ValueError("dataset yaml nc must be an integer.")
    if dataset_nc != len(dataset_names):
        raise ValueError("dataset yaml nc must equal the number of normalized names.")

    export_labels = _normalize_detection_class_labels(
        export_manifest.get("class_labels"), "export manifest class_labels"
    )
    split_labels = _normalize_detection_class_labels(
        split_manifest.get("class_labels"), "split manifest class_labels"
    )
    if len(export_labels) != len(set(export_labels)):
        raise ValueError("export manifest class labels must not contain duplicates.")
    if len(split_labels) != len(set(split_labels)):
        raise ValueError("split manifest class labels must not contain duplicates.")
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("dataset yaml names must not contain duplicates.")
    if dataset_names != export_labels or dataset_names != split_labels:
        raise ValueError("detection class labels must align across dataset, export, and split manifests.")

    class_to_index = _require_dict(export_manifest.get("class_to_index"), "export manifest class_to_index")
    if len(class_to_index) != len(dataset_names):
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
    if set(mapped_ids) != set(range(len(dataset_names))):
        raise ValueError("class_to_index ids must cover exactly 0 through nc - 1.")


def _normalize_detection_class_labels(value: Any, label: str) -> list[str]:
    if isinstance(value, list):
        labels = value
    elif isinstance(value, dict):
        if all(isinstance(key, int) or (isinstance(key, str) and str(key).isdigit()) for key in value.keys()):
            labels = [value[key] for key in sorted(value.keys(), key=lambda item: int(item))]
        elif all(isinstance(item, str) for item in value.keys()) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value.values()
        ):
            labels = [key for key, _ in sorted(value.items(), key=lambda item: int(item[1]))]
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


def args_has_missing_values(args_yaml: dict[str, Any]) -> bool:
    required = ["task", "mode", "model", "data", "epochs", "batch", "imgsz", "device", "project", "name", "pretrained", "seed", "deterministic", "optimizer", "save_dir"]
    for field in required:
        if field not in args_yaml:
            return True
    return False


def _future_output_paths(run_id: str) -> list[str]:
    base = f"artifacts/models/predictions"
    return [
        f"{base}/detection_bbox_predictions__{run_id}__validation.json",
        f"{base}/detection_per_image_summary__{run_id}__validation.json",
        f"{base}/detection_confidence_distribution__{run_id}__validation.json",
        f"{base}/detection_sample_gallery__{run_id}__validation.json",
    ]


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


def _load_csv(path: Path, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


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
