import inspect

from frontend import streamlit_app as app


def test_safety_page_copy_is_current() -> None:
    source = inspect.getsource(app._render_limitations).lower()

    assert app.SAFETY_LIMITATIONS_PAGE_LABEL == "Safety & Limitations"
    assert "local image inspection" in source
    assert "classification" in source
    assert "localization" in source or "boxes" in source
    assert "anomaly" in source
    assert "final rule-based decision" in source
    assert "not production-ready" in source
    assert "not deployment-safe" in source
    assert "manual review" in source or "expert/manual review" in source
    assert "planned / not active" in source
    assert "track a only" not in source
    assert "prototype" not in source
    assert "scaffold" not in source
    assert "fake ai" in source
