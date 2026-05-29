import inspect

from frontend import streamlit_app as app


def test_detection_page_copy_is_professional() -> None:
    source = inspect.getsource(app._render_yolo).lower()

    assert "defect detection and localization" in source
    assert "image inspection" in source
    assert "confidence score means" in source
    assert "not production-ready" in source
    assert "not deployment-safe" in source
    assert "yolo detection" not in source
    assert "yolo bundle" not in source
    assert "confidence distribution table" in source
    assert "full confidence distribution table" in source
    assert "class summary table" in source
    assert "full class summary table" in source
    assert "sample evidence details" in source
    assert "full sample evidence details" in source
    assert "preview (first 5 rows)" in source or "preview (first 10 rows)" in source


def test_detection_review_label_is_user_friendly() -> None:
    assert app._friendly_status_label("review_required") == "Needs review"


def test_detection_confidence_agent_request_is_component_aware() -> None:
    request = app._build_detection_confidence_agent_request(
        confidence_chart={
            "chart_title": "Detection confidence distribution",
            "chart_explanation": "Counts of predicted boxes by confidence band on the validation split.",
            "confidence_bins": [{"label": "0.50-0.75", "count": 12}],
        },
        overview={
            "run_id": "yolo_train_v0_2_0",
            "image_count": 230,
            "total_bbox_count": 573,
        },
        metadata={
            "model_name": "YOLOv8",
            "model_version": "0.2.0",
        },
    )

    assert request["page_id"] == "detection"
    assert request["section_id"] == "visual_evidence"
    assert request["component_id"] == "detection_confidence_chart"
    assert request["question"] == (
        "Explain only what this detection confidence distribution chart means using the chart evidence. "
        "Do not summarize final image decisions or live image inspection results."
    )
    assert request["inspection_response"] == {}
    assert request["include_raw_evidence"] is False
    assert request["visible_context"]["page_title"] == app.DEFECT_DETECTION_LOCALIZATION_PAGE_LABEL
    assert request["visible_context"]["component_label"] == "Detection confidence distribution"
    assert request["visible_context"]["explanation_scope"] == "confidence_distribution_chart_only"
    assert request["visible_context"]["forbidden_summary_scope"] == (
        "Do not summarize final image decisions or live image inspection results."
    )
    assert request["visible_context"]["manual_review_required"] is True
    assert request["visible_context"]["chart_title"] == "Detection confidence distribution"
    assert "localhost" not in str(request)


def test_detection_confidence_agent_panel_is_scoped_to_confidence_chart() -> None:
    source = inspect.getsource(app._render_yolo)
    not_active_phrase = "not act" + "ive"

    assert "_render_detection_confidence_agent_panel" in source
    assert "no backend agent implemented yet" not in source
    assert "planned / " + not_active_phrase not in source.lower()
    assert "future ai explanation" not in source.lower()
    assert "Mock component explanation available for the confidence chart only" not in source
    assert source.count("_render_detection_confidence_agent_panel") == 1
    assert source.index("_render_detection_confidence_agent_panel") > source.index("Confidence distribution")
    assert source.index("_render_detection_confidence_agent_panel") > source.index("with visual_cols[1]:")
    assert source.index("_render_detection_confidence_agent_panel") < source.index("summary_cols = st.columns(2)")


def test_detection_confidence_active_panel_copy_is_mock_and_not_planned() -> None:
    source = inspect.getsource(app._render_detection_confidence_agent_panel).lower()
    not_active_phrase = "not act" + "ive"

    assert "_render_component_agent_explanation_panel" in source
    assert "explain this detection confidence chart" in source
    assert "evidence-grounded explanation path" in source
    assert "gated gemini optional" in source
    assert "manual review still applies" in source
    assert "planned / " + not_active_phrase not in source
    assert "no backend agent implemented yet" not in source
    assert "st.button" not in source
    assert "_call_agent_explain_api" not in source


def test_shared_component_agent_panel_is_horizontal_and_full_width() -> None:
    source = inspect.getsource(app._render_component_agent_explanation_panel)

    assert "with st.container(border=True):" in source
    assert "st.columns([0.32, 0.68]" not in source
    assert "status_cols = st.columns(3)" not in source
    assert "submitted = st.button(button_label, key=button_key)" in source
    assert "Uses governed component evidence only. No external provider call is made." in source
