"""Tests for governed Track B frontend bundle generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.evaluation import generate_track_b_frontend_bundle as bundle


def test_track_b_frontend_bundle_uses_governed_anomaly_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_track_b_frontend_bundle.py",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert bundle.main() == 0

    metric_cards = _load(tmp_path / "metric_cards.json")
    metric_text = json.dumps(metric_cards).lower()
    assert "pr auc" in metric_text
    assert "unavailable" not in metric_text
    assert metric_cards["production_ready"] is False
    assert metric_cards["deployment_safe"] is False

    threshold = _load(tmp_path / "threshold_behavior.json")
    assert len(threshold["rows"]) > 1
    assert threshold["selected_threshold"] == threshold["selected_threshold_metrics"]["threshold"]

    score_summary = _load(tmp_path / "anomaly_score_summary.json")
    assert "histograms" in score_summary
    assert "all" in score_summary["histograms"]
    assert "true_normal" in score_summary["histograms"]
    assert "true_anomaly" in score_summary["histograms"]

    reconstruction = _load(tmp_path / "reconstruction_loss_summary.json")
    mapping = reconstruction["sample_level_reconstruction_loss"]["mapping"]
    assert "reconstruction_loss is equal to anomaly_score" in mapping

    quality = _load(tmp_path / "quality_decision_summary.json")
    quality_text = json.dumps(quality).lower()
    assert "production-canonical" not in quality_text
    assert quality["production_ready"] is False
    assert quality["deployment_safe"] is False

    summary = _load(tmp_path / "frontend_anomaly_summary.json")
    summary_text = json.dumps(summary).lower()
    assert "pr auc" in summary_text
    assert "unavailable" not in summary_text

    inventory = _load(tmp_path / "artifact_inventory_frontend.json")
    assert "anomaly_pr_curve" in json.dumps(inventory["source_artifact_paths"])
    assert inventory["missing_optional_files"] == []
    assert inventory["generation_script"] == "scripts/evaluation/generate_track_b_frontend_bundle.py"
    assert inventory["regeneration_command"] == (
        "PYTHONPATH=src python scripts/evaluation/generate_track_b_frontend_bundle.py"
    )
    assert inventory["frontend_bundle_update_source"] == "governed anomaly evidence"
    assert inventory["no_new_inference"] is True
    assert inventory["no_retraining"] is True
    assert inventory["frontend_ui_code_modified"] is False


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
