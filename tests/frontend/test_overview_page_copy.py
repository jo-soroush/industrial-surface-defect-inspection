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
    assert "mock agent status" in source
    assert "external llm providers are not connected" in source
    assert "ai explanation assistant" in source
    assert "mock backend agent is available for selected evidence-grounded explanations" in source
    assert "recommended review path" in source
    assert "run image inspection" in source
    assert "check safety & limitations" in source
    assert blocked_word_one not in source
    assert blocked_word_two not in source
