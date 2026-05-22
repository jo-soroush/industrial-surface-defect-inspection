from pathlib import Path


def test_frontend_visual_presentation_copy_and_helper_text() -> None:
    source = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")

    required_phrases = [
        "local review/demo",
        "Docker/release validation",
        "planned / not active",
        "Image Inspection",
        "Evidence view only",
        "Not claimed means local review/demo only, not factory production use.",
    ]
    for phrase in required_phrases:
        assert phrase in source, phrase

    blocked_visible_phrases = [
        "Unified image inspection UI is not yet connected here",
        "Partial-failure warnings were returned",
        "Track A Classification",
        "Track B Anomaly Detection",
        "YOLO Detection",
        "Upload / Predict",
        "Visual Status",
        "prototype",
        "scaffold",
        "Sample gallery snapshot",
    ]
    for phrase in blocked_visible_phrases:
        assert phrase not in source, phrase

    assert "\" | \".join(str(item) for item in decision.get(\"supporting_signals\"" not in source

