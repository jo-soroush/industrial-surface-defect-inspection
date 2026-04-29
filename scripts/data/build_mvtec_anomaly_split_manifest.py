from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MVTEC_ROOT = REPO_ROOT / "data/raw/mvtec"
OUTPUT_PATH = REPO_ROOT / "data/manifests/split_mvtec_anomaly.yaml"
PREPROCESSING_POLICY_PATH = "configs/data/preprocessing_mvtec.yaml"
IMAGE_SUFFIX = ".png"
NON_CATEGORY_ROOT_DIRS = {"ground_truth", "test", "train"}


def _repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _sorted_pngs(path: Path) -> list[Path]:
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() == IMAGE_SUFFIX
    )


def _require_dir(path: Path, message: str) -> None:
    if not path.is_dir():
        raise ValueError(message)


def _build_normal_entry(image_path: Path, category: str, split: str) -> dict[str, Any]:
    return {
        "image_path": _repo_relative(image_path),
        "category": category,
        "split": split,
        "label": "normal",
        "label_id": 0,
        "defect_type": "good",
        "mask_path": None,
    }


def _build_anomaly_entry(image_path: Path, category: str, defect_type: str) -> dict[str, Any]:
    mask_path = (
        MVTEC_ROOT
        / category
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )
    if not mask_path.is_file():
        raise ValueError(
            "Missing anomaly mask for "
            f"{_repo_relative(image_path)}: expected {_repo_relative(mask_path)}"
        )

    return {
        "image_path": _repo_relative(image_path),
        "category": category,
        "split": "test",
        "label": "anomaly",
        "label_id": 1,
        "defect_type": defect_type,
        "mask_path": _repo_relative(mask_path),
    }


def build_manifest() -> dict[str, Any]:
    _require_dir(MVTEC_ROOT, f"MVTec root does not exist: {_repo_relative(MVTEC_ROOT)}")

    categories = sorted(
        child
        for child in MVTEC_ROOT.iterdir()
        if child.is_dir() and child.name not in NON_CATEGORY_ROOT_DIRS
    )
    if not categories:
        raise ValueError(f"No MVTec categories found under {_repo_relative(MVTEC_ROOT)}")

    train_entries: list[dict[str, Any]] = []
    test_entries: list[dict[str, Any]] = []
    anomaly_test_count = 0
    normal_test_count = 0

    for category_path in categories:
        category = category_path.name
        train_good_dir = category_path / "train" / "good"
        test_good_dir = category_path / "test" / "good"
        test_dir = category_path / "test"

        _require_dir(
            train_good_dir,
            f"Category {category} lacks required train/good directory: "
            f"{_repo_relative(train_good_dir)}",
        )
        _require_dir(
            test_good_dir,
            f"Category {category} lacks required test/good directory: "
            f"{_repo_relative(test_good_dir)}",
        )

        for image_path in _sorted_pngs(train_good_dir):
            train_entries.append(_build_normal_entry(image_path, category, "train"))

        for image_path in _sorted_pngs(test_good_dir):
            test_entries.append(_build_normal_entry(image_path, category, "test"))
            normal_test_count += 1

        defect_dirs = sorted(
            child
            for child in test_dir.iterdir()
            if child.is_dir() and child.name != "good"
        )
        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            for image_path in _sorted_pngs(defect_dir):
                test_entries.append(_build_anomaly_entry(image_path, category, defect_type))
                anomaly_test_count += 1

    if not train_entries:
        raise ValueError("No anomaly train entries found.")
    if not test_entries:
        raise ValueError("No anomaly test entries found.")

    return {
        "dataset_id": "mvtec_anomaly",
        "dataset_version": "mvtec_1.0",
        "task_type": "anomaly_detection",
        "purpose": "official MVTec anomaly detection workflow",
        "source": "original MVTec train/test/ground_truth directory structure",
        "split_version": "mvtec_anomaly_split_1.0",
        "split_strategy": "official_mvtec_train_good_test_good_and_defect",
        "preprocessing_policy_path": PREPROCESSING_POLICY_PATH,
        "label_policy": (
            "train/good and test/good map to normal; test non-good folders map to anomaly"
        ),
        "mask_policy": (
            "anomaly test samples require matching ground_truth masks; normal samples use null mask_path"
        ),
        "train_entries": train_entries,
        "validation_entries": [],
        "test_entries": test_entries,
        "_counts": {
            "train_count": len(train_entries),
            "validation_count": 0,
            "test_count": len(test_entries),
            "anomaly_test_count": anomaly_test_count,
            "normal_test_count": normal_test_count,
        },
    }


def main() -> int:
    manifest = build_manifest()
    counts = manifest.pop("_counts")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    print(f"output_path={_repo_relative(OUTPUT_PATH)}")
    print(f"train_count={counts['train_count']}")
    print(f"validation_count={counts['validation_count']}")
    print(f"test_count={counts['test_count']}")
    print(f"anomaly_test_count={counts['anomaly_test_count']}")
    print(f"normal_test_count={counts['normal_test_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
