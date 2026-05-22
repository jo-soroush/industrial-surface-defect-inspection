import inspect

from frontend import streamlit_app as app


def test_ai_assistant_page_copy_is_placeholder_only() -> None:
    source = inspect.getsource(app._render_ai_assistant).lower()

    assert app.AI_EXPLANATION_ASSISTANT_PAGE_LABEL == "AI Explanation Assistant"
    assert "planned / not active" in source
    assert "no backend agent" in source
    assert "no llm call" in source
    assert "future ai explanation assistant" in source
    assert "governed frontend bundles" in source
    assert "image inspection" in source
    assert "classification results" in source or "classification result" in source
    assert "defect localization boxes" in source or "localization boxes" in source or "boxes" in source
    assert "anomaly" in source
    assert "final rule-based decisions" in source or "final rule-based decision" in source
    assert "confidence" in source
    assert "manual review needs" in source
    assert "limitations" in source
    assert "traceability" in source
    assert "explanation_context" in source or "explanation context" in source
    assert "invent metrics or predictions" in source
    assert "claim production readiness" in source
    assert "claim deployment safety" in source
    assert "replace reviewer approval" in source
    assert "modify artifacts" in source
    assert "update registries" in source
    assert "silently recompute evidence" in source
    assert "active assistant" not in source
    assert "prototype" not in source
    assert "scaffold" not in source
