from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_CONFIG_PATH = Path("configs/runs/yolo_train_v0_1_0.yaml")
DEFAULT_DATASET_YAML_PATH = Path("data/processed/gc10det_yolo/dataset.yaml")
DEFAULT_REQUIREMENTS_PATH = Path("requirements.txt")
DEFAULT_MODEL_CONFIG_PATH = Path("configs/models/yolo.yaml")
DEFAULT_TRAINING_OUTPUT_ROOT = Path("artifacts/detection/yolo/runs")


def _repo_relative(path: Path) -> str:
    return _resolve_repo_path(path).relative_to(REPO_ROOT).as_posix()


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML file not found: {_repo_relative(path)}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML file: {_repo_relative(path)}") from exc

    if data is None:
        raise ValueError(f"YAML file is empty: {_repo_relative(path)}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must parse to a dictionary: {_repo_relative(path)}")
    return data


def _load_requirements(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"requirements file not found: {_repo_relative(path)}")
    requirements: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


def _ultralytics_declared(requirements_path: Path) -> tuple[bool, str | None]:
    for requirement in _load_requirements(requirements_path):
        normalized = requirement.lower().replace(" ", "")
        if normalized.startswith("ultralytics"):
            return True, requirement
    return False, None


def _ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def _validate_run_config(run_config_path: Path) -> tuple[dict[str, Any], Path]:
    run_config = _load_yaml(run_config_path)

    identity = run_config.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Run config missing dictionary section: identity")
    if identity.get("task_type") != "object_detection":
        raise ValueError("Run config identity.task_type must be object_detection")

    model_identity = run_config.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError("Run config missing dictionary section: model_identity")
    if model_identity.get("model_name") != "yolo":
        raise ValueError("Run config model_identity.model_name must be yolo")
    if model_identity.get("model_type") != "yolo":
        raise ValueError("Run config model_identity.model_type must be yolo")

    dataset_binding = run_config.get("dataset_binding")
    if not isinstance(dataset_binding, dict):
        raise ValueError("Run config missing dictionary section: dataset_binding")
    if dataset_binding.get("dataset_id") != "gc10det_detection":
        raise ValueError("Run config dataset_binding.dataset_id must be gc10det_detection")
    if dataset_binding.get("dataset_version") != "gc10det_1.0":
        raise ValueError("Run config dataset_binding.dataset_version must be gc10det_1.0")

    split_manifest_path = dataset_binding.get("split_manifest_path")
    if not isinstance(split_manifest_path, str) or not split_manifest_path:
        raise ValueError("Run config dataset_binding.split_manifest_path must be set")

    manifest_path = REPO_ROOT / split_manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {split_manifest_path}"
        )

    return run_config, manifest_path


def _validate_model_config(model_config_path: Path) -> dict[str, Any]:
    model_config = _load_yaml(model_config_path)
    if model_config.get("backend") != "ultralytics":
        raise ValueError("YOLO model config backend must be ultralytics")
    if model_config.get("backend_package") != "ultralytics":
        raise ValueError("YOLO model config backend_package must be ultralytics")
    if model_config.get("backend_status") != "dependency_declared_lazy_loaded":
        raise ValueError(
            "YOLO model config backend_status must be dependency_declared_lazy_loaded"
        )
    return model_config


def _resolve_training_model_source(
    model_config: dict[str, Any],
    run_config: dict[str, Any],
) -> str:
    for candidate in (
        model_config.get("training_model_source"),
        model_config.get("model_source"),
        model_config.get("base_model_source"),
        run_config.get("training_model_source"),
        run_config.get("model_source"),
        run_config.get("base_model_source"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    raise ValueError(
        "YOLO training requires a governed model source, but none is declared in "
        "configs/models/yolo.yaml or configs/runs/yolo_train_v0_1_0.yaml. Add a "
        "governed training_model_source before enabling --run-training."
    )


def _validate_dataset_yaml(
    dataset_yaml_path: Path,
    class_labels: list[str],
    expected_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    dataset_yaml = _load_yaml(dataset_yaml_path)

    required_keys = {"train", "val", "test", "nc", "names"}
    missing = sorted(required_keys.difference(dataset_yaml))
    if missing:
        raise ValueError(f"Dataset YAML missing required keys: {missing}")

    if not isinstance(dataset_yaml["names"], list):
        raise ValueError("Dataset YAML names must be a list")

    yaml_names = [str(name) for name in dataset_yaml["names"]]
    if yaml_names != class_labels:
        raise ValueError("Dataset YAML names must match manifest class_labels exactly")

    if dataset_yaml["nc"] != len(class_labels):
        raise ValueError("Dataset YAML nc must equal number of class labels")

    dataset_root = dataset_yaml_path.parent
    split_dir_map = {"train": "train", "val": "validation", "test": "test"}
    split_counts: dict[str, dict[str, int]] = {}

    for yaml_key, split_name in split_dir_map.items():
        image_dir = dataset_root / dataset_yaml[yaml_key]
        label_dir = dataset_root / "labels" / split_name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {_repo_relative(image_dir)}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {_repo_relative(label_dir)}")

        image_files = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        label_files = sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix == ".txt")

        if len(image_files) != len(label_files):
            raise ValueError(
                f"Image/label count mismatch for {split_name}: {len(image_files)} vs {len(label_files)}"
            )
        if len(image_files) != expected_counts[split_name]:
            raise ValueError(
                f"Unexpected {split_name} count: {len(image_files)} expected {expected_counts[split_name]}"
            )

        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        if image_stems != label_stems:
            raise ValueError(
                f"Image/label stems do not match for {split_name}: "
                f"missing labels {sorted(image_stems - label_stems)[:5]} "
                f"missing images {sorted(label_stems - image_stems)[:5]}"
            )

        total_boxes = 0
        empty_label_files = 0
        for label_path in label_files:
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                empty_label_files += 1
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"YOLO label row must contain 5 columns: {_repo_relative(label_path)} -> {line!r}"
                    )
                try:
                    class_index = int(parts[0])
                    floats = [float(value) for value in parts[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid YOLO label row in {_repo_relative(label_path)}: {line!r}"
                    ) from exc
                if not 0 <= class_index < len(class_labels):
                    raise ValueError(
                        f"Invalid class index in {_repo_relative(label_path)}: {class_index}"
                    )
                if any(value < 0.0 or value > 1.0 for value in floats):
                    raise ValueError(
                        f"Normalized bbox values must be in [0, 1] in {_repo_relative(label_path)}"
                    )
                total_boxes += 1

        split_counts[split_name] = {
            "images": len(image_files),
            "labels": len(label_files),
            "empty_label_files": empty_label_files,
            "total_boxes": total_boxes,
        }

    return dataset_yaml, split_counts


def _validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    manifest = _load_yaml(manifest_path)
    if manifest.get("manifest_type") != "detection_split_manifest":
        raise ValueError("Split manifest must be detection_split_manifest")
    if manifest.get("dataset_id") != "gc10det_detection":
        raise ValueError("Split manifest dataset_id must be gc10det_detection")
    if manifest.get("dataset_version") != "gc10det_1.0":
        raise ValueError("Split manifest dataset_version must be gc10det_1.0")
    if manifest.get("task_type") != "object_detection":
        raise ValueError("Split manifest task_type must be object_detection")

    class_labels = manifest.get("class_labels")
    if not isinstance(class_labels, list) or not class_labels:
        raise ValueError("Split manifest class_labels must be a non-empty list")
    class_labels = [str(label) for label in class_labels]

    split_counts = manifest.get("split_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("Split manifest split_counts must be a dictionary")

    expected_counts: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        value = split_counts.get(split_name)
        if not isinstance(value, int):
            raise ValueError(f"Split manifest split_counts.{split_name} must be an integer")
        expected_counts[split_name] = value

    return manifest, class_labels, expected_counts


def _build_training_plan(
    run_config: dict[str, Any],
    model_config: dict[str, Any],
    dataset_yaml_path: Path,
    *,
    epochs_override: int | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    training_runtime = run_config.get("training_runtime")
    if not isinstance(training_runtime, dict):
        raise ValueError("Run config missing dictionary section: training_runtime")

    identity = run_config.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Run config missing dictionary section: identity")

    output_name = str(identity.get("run_config_id", "yolo_training")).strip()

    planned_epochs = epochs_override if epochs_override is not None else training_runtime.get("epochs")
    planned_batch_size = training_runtime.get("batch_size")
    planned_learning_rate = training_runtime.get("learning_rate")
    planned_optimizer = training_runtime.get("optimizer")
    planned_device = device_override if device_override is not None else training_runtime.get("device", "auto")

    if planned_epochs is None:
        raise ValueError("Run config training_runtime.epochs must be set")

    return {
        "output_project": _repo_relative(DEFAULT_TRAINING_OUTPUT_ROOT),
        "output_name": output_name,
        "dataset_yaml_path": _repo_relative(dataset_yaml_path),
        "epochs": planned_epochs,
        "batch_size": planned_batch_size,
        "learning_rate": planned_learning_rate,
        "optimizer": planned_optimizer,
        "device": planned_device,
        "model_source": model_config.get("training_model_source")
        or model_config.get("model_source")
        or model_config.get("base_model_source")
        or run_config.get("training_model_source")
        or run_config.get("model_source")
        or run_config.get("base_model_source"),
    }


def _run_ultralytics_training(
    run_config: dict[str, Any],
    model_config: dict[str, Any],
    dataset_yaml_path: Path,
    *,
    epochs_override: int | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    from inspection_ai.models.yolo_model import _load_ultralytics_yolo

    training_plan = _build_training_plan(
        run_config,
        model_config,
        dataset_yaml_path,
        epochs_override=epochs_override,
        device_override=device_override,
    )
    model_source = _resolve_training_model_source(model_config, run_config)
    yolo_cls = _load_ultralytics_yolo()
    model = yolo_cls(model_source)

    train_kwargs: dict[str, Any] = {
        "data": training_plan["dataset_yaml_path"],
        "epochs": training_plan["epochs"],
        "batch": training_plan["batch_size"],
        "device": training_plan["device"],
        "project": training_plan["output_project"],
        "name": training_plan["output_name"],
    }
    if training_plan["learning_rate"] is not None:
        train_kwargs["lr0"] = training_plan["learning_rate"]
    if training_plan["optimizer"] is not None:
        train_kwargs["optimizer"] = training_plan["optimizer"]

    result = model.train(**train_kwargs)
    return {
        "training_plan": training_plan,
        "model_source": model_source,
        "result": result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Governed YOLO detection training boundary (validate-only)."
    )
    parser.add_argument(
        "--run-config",
        default=str(DEFAULT_RUN_CONFIG_PATH),
        help="Path to the governed YOLO run config.",
    )
    parser.add_argument(
        "--dataset-yaml",
        default=str(DEFAULT_DATASET_YAML_PATH),
        help="Path to the exported YOLO dataset YAML.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the governed detection boundary without starting training.",
    )
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Explicitly enable governed YOLO training execution.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override recorded in the training plan only.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epochs override recorded in the training plan only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.validate_only and args.run_training:
        raise ValueError(
            "Do not pass both --validate-only and --run-training. "
            "Validate-only is the safe default."
        )

    run_config_path = _resolve_repo_path(Path(args.run_config))
    dataset_yaml_path = _resolve_repo_path(Path(args.dataset_yaml))

    run_config, manifest_path = _validate_run_config(run_config_path)
    model_config = _validate_model_config(_resolve_repo_path(DEFAULT_MODEL_CONFIG_PATH))
    manifest, class_labels, expected_counts = _validate_manifest(manifest_path)
    dataset_yaml, split_validation = _validate_dataset_yaml(
        dataset_yaml_path,
        class_labels,
        expected_counts,
    )

    requirements_has_ultralytics, requirements_ultralytics_line = _ultralytics_declared(
        _resolve_repo_path(DEFAULT_REQUIREMENTS_PATH)
    )
    ultralytics_available = _ultralytics_available()

    training_plan = _build_training_plan(
        run_config,
        model_config,
        dataset_yaml_path,
        epochs_override=args.epochs,
        device_override=args.device,
    )

    if args.run_training:
        if not ultralytics_available:
            raise RuntimeError(
                "YOLO training was requested, but the 'ultralytics' package is not "
                "available. Install the declared backend dependency before enabling "
                "--run-training."
            )
        training_result = _run_ultralytics_training(
            run_config,
            model_config,
            dataset_yaml_path,
            epochs_override=args.epochs,
            device_override=args.device,
        )
        print("execution_mode=run_training")
        print("training_status=completed")
        print("run_training_enabled=true")
        print(f"training_output_project={training_result['training_plan']['output_project']}")
        print(f"training_output_name={training_result['training_plan']['output_name']}")
        print(f"training_model_source={training_result['model_source']}")
        return 0

    split_counts = {
        "train": expected_counts["train"],
        "validation": expected_counts["validation"],
        "test": expected_counts["test"],
    }

    print(f"execution_mode=validate_only")
    print(f"training_status=not_started")
    print("run_training_enabled=false")
    print(f"run_config_path={_repo_relative(run_config_path)}")
    print(f"dataset_yaml_path={_repo_relative(dataset_yaml_path)}")
    print(f"dataset_id={manifest['dataset_id']}")
    print(f"dataset_version={manifest['dataset_version']}")
    print(f"model_name={run_config['model_identity']['model_name']}")
    print(f"backend={model_config['backend']}")
    print(f"backend_package={model_config['backend_package']}")
    print(f"ultralytics_available={str(ultralytics_available).lower()}")
    print(f"requirements_ultralytics_declared={str(requirements_has_ultralytics).lower()}")
    if requirements_ultralytics_line is not None:
        print(f"requirements_ultralytics_line={requirements_ultralytics_line}")
    print(
        "split_counts="
        f"train:{split_counts['train']},"
        f"validation:{split_counts['validation']},"
        f"test:{split_counts['test']}"
    )
    print(f"class_count={len(class_labels)}")
    print(f"planned_epochs={training_plan['epochs']}")
    print(f"planned_batch_size={training_plan['batch_size']}")
    print(f"planned_learning_rate={training_plan['learning_rate']}")
    print(f"planned_optimizer={training_plan['optimizer']}")
    print(f"planned_device={training_plan['device']}")
    print(f"planned_output_project={training_plan['output_project']}")
    print(f"planned_output_name={training_plan['output_name']}")
    print(f"planned_model_source={training_plan['model_source'] or 'unset'}")
    print(f"dataset_yaml_nc={dataset_yaml['nc']}")
    print(f"dataset_yaml_names={','.join(dataset_yaml['names'])}")
    print(
        "validated_split_counts="
        f"train:{split_validation['train']['images']},"
        f"validation:{split_validation['validation']['images']},"
        f"test:{split_validation['test']['images']}"
    )
    print(
        "validated_label_counts="
        f"train:{split_validation['train']['labels']},"
        f"validation:{split_validation['validation']['labels']},"
        f"test:{split_validation['test']['labels']}"
    )
    print(
        "validated_empty_label_files="
        f"train:{split_validation['train']['empty_label_files']},"
        f"validation:{split_validation['validation']['empty_label_files']},"
        f"test:{split_validation['test']['empty_label_files']}"
    )
    print(
        "validated_total_boxes="
        f"train:{split_validation['train']['total_boxes']},"
        f"validation:{split_validation['validation']['total_boxes']},"
        f"test:{split_validation['test']['total_boxes']}"
    )
    print(f"split_manifest_path={_repo_relative(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
