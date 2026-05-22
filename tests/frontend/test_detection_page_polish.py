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


def test_detection_review_label_is_user_friendly() -> None:
    assert app._friendly_status_label("review_required") == "Needs review"
