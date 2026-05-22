"""Helpers for loading frontend JSON evidence bundles.

This module keeps the frontend dashboard read-only. It validates that the
expected JSON bundle files exist, parses them safely, and returns structured
dictionary payloads for the dashboard shell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
TRACK_A_BUNDLE_DIR = ROOT_DIR / "artifacts" / "frontend" / "track_a"
TRACK_B_BUNDLE_DIR = ROOT_DIR / "artifacts" / "frontend" / "track_b"
DETECTION_BUNDLE_DIR = ROOT_DIR / "artifacts" / "frontend" / "detection" / "yolo_train_v0_2_0"

TRACK_A_REQUIRED_FILES = (
    "metric_cards.json",
    "confusion_matrix_chart_data.json",
    "per_class_bar_chart_data.json",
    "threshold_curve_chart_data.json",
    "sample_predictions_gallery.json",
    "frontend_model_recommendation.json",
    "quality_decision_summary.json",
    "model_comparison_table.json",
    "artifact_inventory_frontend.json",
    "error_distribution_pie_data.json",
)

TRACK_B_REQUIRED_FILES = (
    "metric_cards.json",
    "anomaly_score_summary.json",
    "frontend_anomaly_summary.json",
    "reconstruction_loss_summary.json",
    "threshold_behavior.json",
    "sample_predictions.json",
    "sample_anomaly_gallery.json",
    "quality_decision_summary.json",
    "artifact_inventory_frontend.json",
)

DETECTION_REQUIRED_FILES = (
    "detection_overview.json",
    "detection_model_metadata.json",
    "detection_metric_cards.json",
    "detection_confidence_chart.json",
    "detection_class_summary.json",
    "detection_sample_gallery.json",
    "detection_artifact_lineage.json",
    "detection_quality_decision_summary.json",
    "frontend_detection_recommendation.json",
    "frontend_bundle_manifest.json",
)


def load_json_file(path: str | Path) -> dict[str, Any] | list[Any] | None:
    """Load a JSON file safely.

    Returns None when the file does not exist or cannot be parsed.
    """
    json_path = Path(path)
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_bundle(bundle_name: str, bundle_dir: Path, required_files: tuple[str, ...]) -> dict[str, Any]:
    """Load and validate a frontend evidence bundle.

    Raises:
        FileNotFoundError: If one or more required bundle files are missing.
        ValueError: If a required JSON file cannot be parsed.
    """
    missing_files = [name for name in required_files if not (bundle_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"{bundle_name} bundle is missing required files: {missing_files}")

    files: dict[str, Any] = {}
    for filename in required_files:
        payload = load_json_file(bundle_dir / filename)
        if payload is None:
            raise ValueError(f"{bundle_name} bundle contains invalid JSON in: {bundle_dir / filename}")
        files[filename] = payload

    if bundle_name == "detection":
        manifest = files["frontend_bundle_manifest.json"]
        if not isinstance(manifest, dict):
            raise ValueError(f"{bundle_name} bundle manifest must be a JSON object: {bundle_dir / 'frontend_bundle_manifest.json'}")
        manifest_files = set(manifest.get("bundle_files", []))
        required_file_set = set(required_files)
        if manifest_files != required_file_set:
            raise ValueError(
                f"{bundle_name} bundle manifest file list does not match required files: "
                f"{sorted(required_file_set - manifest_files)} missing, {sorted(manifest_files - required_file_set)} extra"
            )

    return files


def load_track_a_bundle() -> dict[str, Any]:
    """Load the surface defect frontend evidence bundle."""
    return _load_bundle("track_a", TRACK_A_BUNDLE_DIR, TRACK_A_REQUIRED_FILES)


def load_track_b_bundle() -> dict[str, Any]:
    """Load the surface anomaly frontend evidence bundle."""
    return _load_bundle("track_b", TRACK_B_BUNDLE_DIR, TRACK_B_REQUIRED_FILES)


def load_detection_bundle() -> dict[str, Any]:
    """Load the Detection frontend evidence bundle."""
    return _load_bundle("detection", DETECTION_BUNDLE_DIR, DETECTION_REQUIRED_FILES)


def load_all_frontend_bundles() -> dict[str, dict[str, Any]]:
    """Load all frontend evidence bundles into a structured dictionary."""
    return {
        "track_a": load_track_a_bundle(),
        "track_b": load_track_b_bundle(),
        "detection": load_detection_bundle(),
    }
