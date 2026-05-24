from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs" / "runtime_assets" / "manifest.yaml"
SCRIPT_PATH = REPO_ROOT / "scripts" / "runtime_assets" / "stage_runtime_assets.py"


def test_manifest_exists_and_parses() -> None:
    assert MANIFEST_PATH.is_file()
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "assets" in payload
    assert isinstance(payload["assets"], list)
    assert payload["assets"]


def test_manifest_sources_and_targets_are_safe() -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assets = payload["assets"]

    for item in assets:
        assert isinstance(item, dict)
        source = Path(item["source"])
        target = Path(item["target"])
        assert source != Path("artifacts")
        assert not source.is_absolute()
        assert not target.is_absolute()
        assert ".." not in source.parts
        assert ".." not in target.parts
        assert "notebooks" not in source.parts
        assert "__pycache__" not in source.parts
        assert "scratch" not in source.parts
        assert "notebooks" not in target.parts
        assert "__pycache__" not in target.parts
        assert "scratch" not in target.parts


def test_required_manifest_sources_exist() -> None:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    missing = []
    for item in payload["assets"]:
        if not item.get("required", False):
            continue
        source = REPO_ROOT / item["source"]
        if not source.exists():
            missing.append(item["source"])
    assert not missing, f"Missing required manifest sources: {missing}"


def test_manifest_check_mode_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--check",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime_asset_stage_status=PASS" in result.stdout
    assert "check_only=true" in result.stdout


def test_check_mode_does_not_create_output_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--check",
            "--output",
            str(tmp_path / "runtime_assets"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "runtime_assets").exists()


def test_staging_with_temporary_fixture_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    nested_dir = source_dir / "nested"
    nested_dir.mkdir()
    (source_dir / "a.json").write_text('{"ok": true}', encoding="utf-8")
    (nested_dir / "b.json").write_text('{"value": 1}', encoding="utf-8")

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "assets:",
                "  - source: source/a.json",
                "    target: bundle/a.json",
                "    required: true",
                "    role: test file",
                "    consumer: test",
                "  - source: source/nested",
                "    target: bundle/nested",
                "    required: true",
                "    role: test directory",
                "    consumer: test",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(manifest_path),
            "--root",
            str(tmp_path),
            "--output",
            str(tmp_path / "runtime_assets"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    staged_root = tmp_path / "runtime_assets"
    assert (staged_root / "bundle" / "a.json").is_file()
    assert (staged_root / "bundle" / "nested" / "b.json").is_file()
    assert "assets_staged=2" in result.stdout
