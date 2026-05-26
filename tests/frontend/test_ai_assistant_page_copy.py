import inspect

from frontend import streamlit_app as app


def test_ai_assistant_page_copy_reflects_mock_agent_and_future_llm_boundary() -> None:
    source = inspect.getsource(app._render_ai_assistant).lower()

    assert app.AI_EXPLANATION_ASSISTANT_PAGE_LABEL == "AI Explanation Assistant"
    assert "mock backend agent active" in source
    assert "mock component explanations are active" in source
    assert "external llm providers are not connected" in source
    assert "real llm providers are not connected" in source
    assert "gemini/grok/openai are not active" in source
    assert "mock/pre-gemini explanation layer" in source
    assert "broader natural-language llm assistance remains planned" in source
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
    assert "no backend agent" not in source
    assert "planned / not active" not in source
    assert "no llm call" not in source
    assert "gemini is connected" not in source
    assert "grok is connected" not in source
    assert "openai is connected" not in source
    assert "prototype" not in source
    assert "scaffold" not in source
