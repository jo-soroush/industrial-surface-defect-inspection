import inspect

from frontend import streamlit_app as app


def test_classification_page_copy_is_professional() -> None:
    source = inspect.getsource(app._render_track_a).lower()

    assert "surface defect classification" in source
    assert "image inspection" in source
    assert "decision boundary" in source
    assert "not production-ready" in source
    assert "not deployment-safe" in source
    assert "track a classification" not in source
    assert "prototype" not in source
    assert "scaffold" not in source


def test_classification_quality_label_is_user_friendly() -> None:
    assert app._friendly_status_label("TRACK_A_STRONG_CANDIDATE") == "Strong classification candidate"
    assert app._friendly_metric_display("TRACK_A_STRONG_CANDIDATE") == "Strong classification candidate"
    assert (
        app._friendly_metric_display("Selected Track A candidate, not production-ready.")
        == "Selected governed classification candidate; local review/demo only, not production use."
    )


def test_classification_threshold_agent_request_is_component_aware() -> None:
    request = app._build_classification_threshold_agent_request(
        threshold_curve={
            "chart_title": "Surface defect threshold behavior",
            "chart_explanation": "Validation-derived threshold tradeoffs.",
            "baseline_threshold": 0.5,
            "recommended_threshold": 0.65,
            "run_id": "run-123",
            "rows": [{"threshold": 0.5}, {"threshold": 0.65}],
        },
        recommendation={
            "selected_model_name": "resnet18",
            "selected_model_version": "0.4.0",
            "selected_run_id": "run-123",
        },
        metric_cards={"validation_samples": 803},
    )

    assert request["page_id"] == "classification"
    assert request["section_id"] == "detailed_metrics"
    assert request["component_id"] == "classification_threshold_curve_chart"
    assert request["question"] == "What does this classification threshold chart mean?"
    assert request["inspection_response"] == {}
    assert request["include_raw_evidence"] is False
    assert request["visible_context"]["page_title"] == app.SURFACE_DEFECT_CLASSIFICATION_PAGE_LABEL
    assert request["visible_context"]["component_label"] == "Surface defect threshold behavior"
    assert request["visible_context"]["recommended_threshold"] == 0.65
    assert "localhost" not in str(request)


def test_classification_threshold_agent_panel_is_scoped_to_threshold_chart() -> None:
    source = inspect.getsource(app._render_track_a)

    assert "_render_classification_threshold_agent_panel" in source
    assert "Explain these classification charts" not in source
    assert "no backend agent implemented yet" not in source
    assert "planned / not active" not in source.lower()
    assert source.count("_render_classification_threshold_agent_panel") == 1
    assert source.index("_render_classification_threshold_agent_panel") > source.index("Threshold behavior")
    assert source.index("_render_classification_threshold_agent_panel") > source.index("with visual_cols[2]:")
    assert source.index("_render_classification_threshold_agent_panel") < source.index('st.markdown("### Sample evidence summary")')


def test_classification_threshold_active_panel_copy_is_mock_and_not_planned() -> None:
    source = inspect.getsource(app._render_classification_threshold_agent_panel).lower()

    assert "_render_component_agent_explanation_panel" in source
    assert "explain this classification threshold chart" in source
    assert "mock evidence-grounded explanation" in source
    assert "external llm not connected" in source
    assert "manual review still applies" in source
    assert "planned / not active" not in source
    assert "no backend agent implemented yet" not in source
    assert "st.button" not in source
    assert "_call_agent_explain_api" not in source
