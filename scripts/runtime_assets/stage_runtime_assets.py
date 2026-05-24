"""Validate and stage the curated runtime asset manifest."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "configs" / "runtime_assets" / "manifest.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime_assets"
IGNORED_DIRECTORY_NAMES = {"__pycache__", ".ipynb_checkpoints", ".git"}


@dataclass(frozen=True)
class RuntimeAssetEntry:
    source: Path
    target: Path
    required: bool
    role: str
    consumer: str


@dataclass(frozen=True)
class RuntimeAssetSummary:
    assets_checked: int
    assets_staged: int
    missing_required_assets: tuple[str, ...]
    skipped_optional_assets: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or stage curated runtime assets.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the runtime asset manifest YAML file.",
    )
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Repository root used to resolve manifest source paths.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for staged runtime assets.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the manifest and source availability without copying files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = _resolve_path(args.manifest)
    source_root = _resolve_path(args.root)
    output_dir = _resolve_path(args.output)

    try:
        manifest = load_manifest(manifest_path)
        summary = stage_runtime_assets(
            manifest,
            source_root=source_root,
            output_dir=output_dir,
            check_only=args.check,
        )
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"runtime_asset_stage_status=FAIL")
        print(f"failure_reason={exc}")
        return 1

    _print_summary(summary, check_only=args.check, output_dir=output_dir)
    if summary.missing_required_assets:
        return 1
    return 0


def load_manifest(manifest_path: Path) -> list[RuntimeAssetEntry]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Manifest YAML is invalid: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Manifest YAML must contain a mapping at the top level.")

    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("Manifest YAML must contain a non-empty 'assets' list.")

    entries: list[RuntimeAssetEntry] = []
    for index, item in enumerate(assets, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest asset entry {index} must be a mapping.")

        source = _parse_relative_path(item.get("source"), f"assets[{index}].source")
        target = _parse_relative_path(item.get("target"), f"assets[{index}].target")
        required = _parse_bool(item.get("required"), f"assets[{index}].required")
        role = _parse_nonempty_string(item.get("role"), f"assets[{index}].role")
        consumer = _parse_nonempty_string(item.get("consumer"), f"assets[{index}].consumer")

        entries.append(
            RuntimeAssetEntry(
                source=source,
                target=target,
                required=required,
                role=role,
                consumer=consumer,
            )
        )

    return entries


def stage_runtime_assets(
    entries: list[RuntimeAssetEntry],
    *,
    source_root: Path,
    output_dir: Path,
    check_only: bool,
) -> RuntimeAssetSummary:
    assets_checked = 0
    assets_staged = 0
    missing_required_assets: list[str] = []
    skipped_optional_assets: list[str] = []

    resolved_entries = list(entries)
    for entry in resolved_entries:
        assets_checked += 1
        source_path = source_root / entry.source
        if not source_path.exists():
            if entry.required:
                missing_required_assets.append(str(entry.source))
            else:
                skipped_optional_assets.append(str(entry.source))

    if missing_required_assets:
        return RuntimeAssetSummary(
            assets_checked=assets_checked,
            assets_staged=0,
            missing_required_assets=tuple(missing_required_assets),
            skipped_optional_assets=tuple(skipped_optional_assets),
        )

    if check_only:
        return RuntimeAssetSummary(
            assets_checked=assets_checked,
            assets_staged=0,
            missing_required_assets=(),
            skipped_optional_assets=tuple(skipped_optional_assets),
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path must be a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in resolved_entries:
        source_path = source_root / entry.source
        if not source_path.exists():
            continue

        target_path = output_dir / entry.target
        _copy_runtime_asset(source_path, target_path)
        assets_staged += 1

    return RuntimeAssetSummary(
        assets_checked=assets_checked,
        assets_staged=assets_staged,
        missing_required_assets=(),
        skipped_optional_assets=tuple(skipped_optional_assets),
    )


def _copy_runtime_asset(source_path: Path, target_path: Path) -> None:
    if source_path.is_dir():
        if target_path.exists() and not target_path.is_dir():
            raise ValueError(
                f"Cannot stage directory {source_path} onto existing file target {target_path}."
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source_path,
            target_path,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORED_DIRECTORY_NAMES),
        )
        return

    if target_path.exists() and target_path.is_dir():
        raise ValueError(
            f"Cannot stage file {source_path} onto existing directory target {target_path}."
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _print_summary(
    summary: RuntimeAssetSummary,
    *,
    check_only: bool,
    output_dir: Path,
) -> None:
    print(f"runtime_asset_stage_status={'PASS' if not summary.missing_required_assets else 'FAIL'}")
    print(f"assets_checked={summary.assets_checked}")
    print(f"assets_staged={summary.assets_staged}")
    print(f"missing_required_assets={len(summary.missing_required_assets)}")
    print(f"skipped_optional_assets={len(summary.skipped_optional_assets)}")
    if check_only:
        print("check_only=true")
    else:
        print(f"output_dir={output_dir}")
    for missing in summary.missing_required_assets:
        print(f"missing_required_asset={missing}")
    for skipped in summary.skipped_optional_assets:
        print(f"skipped_optional_asset={skipped}")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return Path.cwd() / path
    return path


def _parse_relative_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    path = Path(value.strip())
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative to the repository root: {value!r}")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{field_name} must not escape the repository root: {value!r}")
    if path == Path("artifacts"):
        raise ValueError(f"{field_name} must not point to the raw artifacts root.")
    return path


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean.")


def _parse_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
