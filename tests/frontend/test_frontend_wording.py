import inspect

from frontend import streamlit_app as app


def test_frontend_page_labels_are_professional() -> None:
    assert app.OVERVIEW_PAGE_LABEL == "Overview"
    assert app.SURFACE_DEFECT_CLASSIFICATION_PAGE_LABEL == "Surface Defect Classification"
    assert app.SURFACE_ANOMALY_DETECTION_PAGE_LABEL == "Surface Anomaly Detection"
    assert app.DEFECT_DETECTION_LOCALIZATION_PAGE_LABEL == "Defect Detection & Localization"
    assert app.IMAGE_INSPECTION_PAGE_LABEL == "Image Inspection"
    assert app.SAFETY_LIMITATIONS_PAGE_LABEL == "Safety & Limitations"
    assert app.AI_EXPLANATION_ASSISTANT_PAGE_LABEL == "AI Explanation Assistant"
    assert app.INSPECTION_CAPABILITY_SUMMARY_LABEL == "Recommended review path"


def test_frontend_status_labels_are_user_friendly() -> None:
    assert app._friendly_status_label("review_required") == "Needs review"
    assert app._friendly_status_label("review_required_weak_evidence") == "Review required: weak evidence"
    assert app._friendly_status_label("review_only_signal") == "Review-only supporting signal"
    assert app._friendly_status_label("frontend_bundle_ready_for_review") == "Frontend evidence bundle ready for review"
    assert app._friendly_status_label("strong_track_a_candidate_selected_not_production_ready") == "Strong classification candidate"
    assert app._friendly_status_label("production-canonical") == "Governed review evidence"


def test_frontend_source_no_longer_shows_global_ai_placeholder_sentence() -> None:
    source = inspect.getsource(app)
    lower = source.lower()
    assert "the future ai explanation assistant is a placeholder only." not in lower
