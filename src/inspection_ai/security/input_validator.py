from __future__ import annotations

from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/data/input_policy.yaml"


def _load_policy() -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if data is not None else {}
    except ModuleNotFoundError:
        policy: dict[str, Any] = {}
        current_key: str | None = None
        for raw_line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- "):
                if current_key is not None:
                    policy.setdefault(current_key, []).append(line[2:].strip())
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                current_key = key
                policy[key] = []
            elif value.isdigit():
                current_key = key
                policy[key] = int(value)
            elif value == "[]":
                current_key = key
                policy[key] = []
            else:
                current_key = key
                policy[key] = value
        return policy


def validate_image_input(file_path: str | Path) -> bool:
    policy = _load_policy()
    path = Path(file_path)

    if not path.exists():
        return False

    if not path.is_file():
        return False

    extension = path.suffix.lower().lstrip(".")
    allowed_extensions = set(policy.get("allowed_extensions", []))
    if extension not in allowed_extensions and policy.get("invalid_type_policy") == "reject":
        return False

    max_file_size_bytes = int(policy.get("max_file_size_bytes", 0))
    try:
        file_size = path.stat().st_size
    except OSError:
        return False

    if file_size > max_file_size_bytes and policy.get("invalid_size_policy") == "reject":
        return False

    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            image.verify()
    except Exception:
        if policy.get("decode_failure_policy") == "reject":
            return False

    return True
