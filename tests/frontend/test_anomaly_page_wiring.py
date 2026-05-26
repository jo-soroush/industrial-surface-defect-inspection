from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from frontend.data_loader import load_track_b_bundle
from frontend import streamlit_app as app
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
    anomaly_source = inspect.getsource(app._render_track_b)
    assert "PR AUC is unavailable" not in source
    assert "No anomaly score data is available in the governed bundle" not in source
    assert "Sample evidence count" in anomaly_source
    assert "Sample predictions" in anomaly_source
    assert "Summary-only view; the full governed sample prediction evidence remains in the artifact bundle." in anomaly_source
    assert "Sample prediction preview" in anomaly_source
    assert "Preview (first 5 rows)" in anomaly_source
    assert "replace(\"_\", \" \").title()" in anomaly_source
    assert "_render_mini_metric_tile(" in anomaly_source
    assert "_render_premium_info_card(\n                    label," not in anomaly_source
    assert "_render_chart_mini_tile(" in anomaly_source


def test_anomaly_threshold_agent_request_is_component_aware() -> None:
    request = app._build_anomaly_threshold_agent_request(
        threshold_behavior={
            "selected_threshold": 0.2043,
            "run_id": "run-456",
            "rows": [{"threshold": 0.1}, {"threshold": 0.2}],
        },
        frontend_summary={
            "model_type": "autoencoder",
            "model_version": "0.1.0",
            "key_metrics": {"threshold": 0.2043, "pr_auc": 0.718},
        },
        metric_cards={},
        quality={
            "quality_status": "review_required_weak_evidence",
            "dashboard_usage_recommendation": "review_only_supporting_signal",
        },
    )

    assert request["page_id"] == "anomaly"
    assert request["section_id"] == "visual_evidence"
    assert request["component_id"] == "anomaly_threshold_behavior_chart"
    assert request["question"] == "What does this anomaly threshold behavior chart mean?"
    assert request["inspection_response"] == {}
    assert request["include_raw_evidence"] is False
    assert request["visible_context"]["page_title"] == app.SURFACE_ANOMALY_DETECTION_PAGE_LABEL
    assert request["visible_context"]["component_label"] == "Surface anomaly threshold behavior"
    assert request["visible_context"]["selected_threshold"] == pytest.approx(0.2043)
    assert request["visible_context"]["quality_status"] == "review_required_weak_evidence"
    assert "localhost" not in str(request)


def test_anomaly_threshold_agent_panel_is_scoped_to_threshold_chart() -> None:
    source = inspect.getsource(app._render_track_b)

    assert "_render_anomaly_threshold_agent_panel" in source
    assert "Explain anomaly behavior" not in source
    assert "no backend agent implemented yet" not in source
    assert "planned / not active" not in source.lower()
    assert source.count("_render_anomaly_threshold_agent_panel") == 1
    assert source.index("_render_anomaly_threshold_agent_panel") > source.index("Threshold behavior")
    assert source.index("_render_anomaly_threshold_agent_panel") > source.index("with visual_cols[2]:")
    assert source.index("_render_anomaly_threshold_agent_panel") < source.index('st.markdown("### Sample evidence summary")')


def test_anomaly_threshold_active_panel_copy_is_mock_and_review_only() -> None:
    source = inspect.getsource(app._render_anomaly_threshold_agent_panel).lower()

    assert "_render_component_agent_explanation_panel" in source
    assert "explain this anomaly threshold behavior chart" in source
    assert "mock evidence-grounded explanation" in source
    assert "external llm not connected" in source
    assert "anomaly evidence is review-only" in source
    assert "manual review still applies" in source
    assert "planned / not active" not in source
    assert "no backend agent implemented yet" not in source
    assert "st.button" not in source
    assert "_call_agent_explain_api" not in source


def test_all_chart_agent_panels_use_shared_component_helper() -> None:
    detection_source = inspect.getsource(app._render_detection_confidence_agent_panel)
    classification_source = inspect.getsource(app._render_classification_threshold_agent_panel)
    anomaly_source = inspect.getsource(app._render_anomaly_threshold_agent_panel)

    for source in (detection_source, classification_source, anomaly_source):
        assert "_render_component_agent_explanation_panel" in source
        assert source.count("_render_component_agent_explanation_panel") == 1
