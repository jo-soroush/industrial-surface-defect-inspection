import inspect

from frontend import streamlit_app as app


def test_overview_copy_is_professional_and_grounded() -> None:
    source = inspect.getsource(app).lower()
    blocked_word_one = "pro" "totype"
    blocked_word_two = "scaf" "fold"

    assert "image inspection" in source
    assert "rule-based decision" in source
    assert "not production-ready" in source
    assert "not deployment-safe" in source
    assert "inspection capability summary" not in source
    assert "gated ai explanation status" in source
    assert "evidence-grounded explanations are available for selected components" in source
    assert "uses /agent/explain" in source
    assert "safe fallback remains available" in source
    assert "ai explanation assistant" in source
    assert "evidence-grounded explanations are active for selected components" in source
    assert "does not contain an autonomous ai agent" in source
    assert "current explanations are gated, evidence-grounded, and review-only" in source
    assert "recommended review path" in source
    assert "run image inspection" in source
    assert "check safety & limitations" in source
    assert blocked_word_one not in source
    assert blocked_word_two not in source
