from pathlib import Path


def test_frontend_visual_presentation_copy_and_helper_text() -> None:
    source = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")

    required_phrases = [
        "local review/demo",
        "Docker/release validation",
        "Previous EC2 demo completed",
        "Gemini-gated responses can be used when enabled",
        "Image Inspection",
        "Not claimed means local review/demo only, not factory production use.",
        "Recommended review path",
        "Run Image Inspection",
        "Check Safety & Limitations",
        "Classification validation bundle",
        "Anomaly validation bundle",
        "Detection validation bundle",
        "Unified inspection workflow",
    ]
    for phrase in required_phrases:
        assert phrase in source, phrase

    blocked_visible_phrases = [
        "Evidence/dashboard view only",
        "Evidence view only — local review/demo, not production use.",
        "Local inspection workflow. not production-ready. not deployment-safe.",
        "Unified image inspection UI is not yet connected here",
        "Partial-failure warnings were returned",
        "Selected Track A candidate",
        "Track A Classification",
        "Track B Anomaly Detection",
        "YOLO Detection",
        "Upload / Predict",
        "Visual Status",
        "Inspection capability summary",
        "prototype",
        "scaffold",
        "Sample gallery snapshot",
    ]
    for phrase in blocked_visible_phrases:
        assert phrase not in source, phrase

    assert "\" | \".join(str(item) for item in decision.get(\"supporting_signals\"" not in source
    assert "Selected governed classification candidate" in source
    assert "Strong classification candidate" in source
