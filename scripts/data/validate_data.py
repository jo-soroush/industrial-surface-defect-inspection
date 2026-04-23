from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import imghdr
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/manifests/dataset_registry.yaml"
REPORTS_DIR = REPO_ROOT / "artifacts/reports/data_validation"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "null":
        return None
    if value.isdigit():
        return int(value)
    return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if data is not None else {}
    except ModuleNotFoundError:
        pass

    lines = path.read_text(encoding="utf-8").splitlines()
    data: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            if current_key is None:
                continue
            data.setdefault(current_key, []).append(_parse_scalar(stripped[2:]))
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                current_key = key
                data[key] = []
            else:
                current_key = key
                data[key] = _parse_scalar(value)

    return data


def _load_registry(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("datasets", [])
    except ModuleNotFoundError:
        pass

    datasets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        stripped = line.strip()
        if stripped == "datasets:":
            continue

        if stripped.startswith("- "):
            if current is not None:
                datasets.append(current)
            current = {}
            stripped = stripped[2:]
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                current[key.strip()] = _parse_scalar(value)
            continue

        if current is None:
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = _parse_scalar(value)

    if current is not None:
        datasets.append(current)

    return datasets


def _readable_image(path: Path) -> bool:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            image.verify()
        return True
    except ModuleNotFoundError:
        try:
            with path.open("rb") as handle:
                header = handle.read(512)
            return imghdr.what(None, h=header) is not None
        except OSError:
            return False
    except Exception:
        return False


def _expected_structure_path(dataset: dict[str, Any]) -> Path:
    structure_id = dataset["expected_structure_id"]
    return REPO_ROOT / f"data/manifests/{structure_id}.yaml"


def _scan_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    root = REPO_ROOT / str(dataset["storage_root"])
    expected = _load_simple_yaml(_expected_structure_path(dataset))
    required_subfolders = expected.get("required_subfolders", [])

    missing = 0
    for folder_name in required_subfolders:
        folder_path = root / str(folder_name)
        if not folder_path.is_dir():
            missing += 1

    total_seen = 0
    total_readable = 0
    total_unreadable = 0

    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            total_seen += 1
            if _readable_image(path):
                total_readable += 1
            else:
                total_unreadable += 1

    validation_status = "pass" if total_unreadable == 0 and missing == 0 else "fail"

    return {
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["dataset_version"],
        "validation_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "total_files_seen": total_seen,
        "total_files_readable": total_readable,
        "total_files_unreadable": total_unreadable,
        "total_missing_files": missing,
        "validation_status": validation_status,
    }


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"dataset_id: {report['dataset_id']}",
        f"dataset_version: {report['dataset_version']}",
        f"validation_timestamp: \"{report['validation_timestamp']}\"",
        f"total_files_seen: {report['total_files_seen']}",
        f"total_files_readable: {report['total_files_readable']}",
        f"total_files_unreadable: {report['total_files_unreadable']}",
        f"total_missing_files: {report['total_missing_files']}",
        f"validation_status: {report['validation_status']}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    datasets = _load_registry(REGISTRY_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        if dataset.get("expected_structure_id") is None or dataset.get("validation_report_path") is None:
            print(
                f"Skipping dataset {dataset['dataset_id']}: "
                "incomplete governance (missing expected_structure_id or validation_report_path)"
            )
            continue

        report = _scan_dataset(dataset)
        report_path = REPO_ROOT / str(dataset["validation_report_path"])
        _write_report(report_path, report)
        print(
            f"{report['dataset_id']}: "
            f"seen={report['total_files_seen']} "
            f"readable={report['total_files_readable']} "
            f"unreadable={report['total_files_unreadable']} "
            f"missing={report['total_missing_files']} "
            f"status={report['validation_status']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
