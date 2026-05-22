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
