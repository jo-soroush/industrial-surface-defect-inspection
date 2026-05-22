from __future__ import annotations

from pathlib import Path

import pytest

from frontend.data_loader import load_track_b_bundle
from frontend.streamlit_app import (
    _extract_histogram_series,
    _extract_pr_auc,
    _extract_sample_prediction_rows,
    _extract_threshold_rows,
    _friendly_status_label,
)


def test_track_b_bundle_exposes_governed_anomaly_evidence() -> None:
    bundle = load_track_b_bundle()

    frontend_summary = bundle["frontend_anomaly_summary.json"]
    metric_cards = bundle["metric_cards.json"]
    anomaly_summary = bundle["anomaly_score_summary.json"]
    reconstruction = bundle["reconstruction_loss_summary.json"]["sample_level_reconstruction_loss"]
    threshold_behavior = bundle["threshold_behavior.json"]
    sample_predictions = bundle["sample_predictions.json"]
    quality = bundle["quality_decision_summary.json"]

    assert _extract_pr_auc(frontend_summary, metric_cards) == pytest.approx(0.7182909909021874)

    labels, hist_series = _extract_histogram_series(anomaly_summary["histograms"])
    assert len(labels) == 20
    assert set(hist_series) == {"All", "True Normal", "True Anomaly"}
    assert sum(hist_series["All"]) == anomaly_summary["summary"]["all"]["count"]

    reconstruction_labels, reconstruction_series = _extract_histogram_series(reconstruction["histograms"])
    assert len(reconstruction_labels) == 20
    assert sum(reconstruction_series["All"]) == reconstruction["summary"]["all"]["count"]

    rows = _extract_threshold_rows(threshold_behavior)
    assert len(rows) > 1
    assert rows[0]["threshold"] < rows[-1]["threshold"]

    sample_rows = _extract_sample_prediction_rows(sample_predictions)
    assert len(sample_rows) == sample_predictions["sample_count"] == 1725
    assert sample_rows[0]["reconstruction_loss"] == sample_rows[0]["anomaly_score"]

    assert _friendly_status_label(quality["quality_status"]) == "Review required: weak evidence"
    assert _friendly_status_label(quality["dashboard_usage_recommendation"]) == "Review-only supporting signal"


def test_anomaly_page_source_no_longer_shows_stale_unavailable_copy() -> None:
    source = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    assert "PR AUC is unavailable" not in source
    assert "No anomaly score data is available in the governed bundle" not in source
