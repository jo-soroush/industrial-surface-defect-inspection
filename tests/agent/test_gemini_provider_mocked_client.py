"""Tests for the Gemini Phase G2 mocked client seam."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.inspection_ai.agent.context_builder import build_grounding_context
import src.inspection_ai.agent.gemini_provider as gemini_provider_module
from src.inspection_ai.agent.gemini_provider import (
    GeminiClientResult,
    GeminiGenerationRequest,
    GeminiProviderError,
    GeminiProviderEmptyResponseError,
    GeminiProviderMalformedResponseError,
    GeminiProviderRateLimitError,
    GeminiRealProvider,
    GeminiRealProviderConfig,
    GeminiSdkLoadResult,
    GeminiProviderSkeleton,
    GeminiProviderTimeoutError,
    GeminiProviderStub,
)
from src.inspection_ai.agent.provider_contracts import build_provider_request
from src.inspection_ai.agent.provider_router import AgentProviderRouter


REPO_ROOT = Path(__file__).resolve().parents[2]
GEMINI_PROVIDER_PATH = REPO_ROOT / "src/inspection_ai/agent/gemini_provider.py"


def test_gemini_mocked_client_success_translates_to_gemini_response() -> None:
    stub = GeminiProviderStub()
    request = _build_gemini_provider_request(
        question="What does this classification threshold chart mean?",
    )

    evaluation = stub.evaluate_mocked_client_result(
        request,
        GeminiClientResult(
            text=(
                "This threshold chart summarizes validation evidence and threshold behavior. "
                "Manual review still applies."
            ),
        ),
    )

    assert evaluation.provider_response.provider_used == "gemini"
    assert evaluation.provider_response.fallback_used is False
    assert evaluation.provider_response.raw_provider_response_allowed is False
    assert evaluation.provider_response.safety_status in {"pass", "limited"}
    assert evaluation.safe_to_send is True
    assert evaluation.safe_to_display is True
    assert "manual review" in evaluation.provider_response.answer.lower()
    assert "validation evidence" in evaluation.provider_response.answer.lower()


def test_gemini_mocked_client_timeout_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(error_kind="timeout"),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert evaluation.provider_response.raw_provider_response_allowed is False
    assert evaluation.provider_error is not None
    assert "timeout" in evaluation.provider_error.lower()
    assert evaluation.fallback_reason is not None
    assert "mock fallback" in evaluation.fallback_reason.lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_provider_error_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(error_kind="provider_error"),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert "provider error" in (evaluation.provider_error or "").lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_rate_limit_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(error_kind="rate_limit"),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert "rate limited" in (evaluation.provider_error or "").lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_empty_response_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(text=""),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert "empty response" in (evaluation.provider_error or "").lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_malformed_response_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(payload=["not", "a", "dict"]),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert "malformed" in (evaluation.provider_error or "").lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_unsafe_output_is_blocked_by_safety_guard() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(
            text=(
                "This model is production-ready, deployment-safe, and Gemini is active. "
                "Manual review is not required."
            ),
        ),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert evaluation.provider_response.safety_status == "blocked"
    assert evaluation.safe_to_display is False
    assert "manual review" in evaluation.provider_response.answer.lower()
    assert "production-ready" not in evaluation.provider_response.answer.lower()


def test_gemini_mocked_client_invented_metric_like_output_is_blocked_by_safety_guard() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(
            text="The threshold is 0.99 and the F1 score is 0.87. Manual review still applies.",
        ),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert evaluation.provider_response.safety_status == "blocked"
    assert evaluation.safe_to_display is False
    assert evaluation.provider_error is not None


def test_gemini_mocked_client_readiness_claims_are_blocked_separately() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(
            text=(
                "This model is production-ready, deployment-safe, and Gemini is active. "
                "Manual review is not required."
            ),
        ),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert evaluation.provider_response.safety_status == "blocked"
    assert evaluation.safe_to_display is False
    assert "production-ready" not in evaluation.provider_response.answer.lower()


def test_gemini_mocked_client_sanitizes_secret_like_questions_before_handling() -> None:
    request = _build_gemini_provider_request(
        question="Explain /Users/jo.soroush/secret.key",
    )

    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        request,
        GeminiClientResult(
            text="This is a safe mocked Gemini answer. Manual review still applies.",
        ),
    )

    assert request.provider_name == "gemini"
    assert request.question == "[REDACTED_SECRET]"
    assert request.sanitized_context["question"] == "[REDACTED_SECRET]"
    assert request.grounding_context["question"] == "[REDACTED_SECRET]"
    assert "/Users/jo.soroush/secret.key" not in request.question
    assert "/Users/jo.soroush/secret.key" not in evaluation.provider_response.answer
    assert evaluation.provider_response.provider_used == "gemini"
    assert evaluation.safe_to_display is True


def test_gemini_provider_module_does_not_import_provider_sdks_or_network_libraries() -> None:
    tree = ast.parse(GEMINI_PROVIDER_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    banned_roots = {"google", "openai", "requests", "httpx", "urllib"}
    assert imported_roots.isdisjoint(banned_roots)


def test_gemini_provider_skeleton_without_injected_client_returns_not_implemented() -> None:
    skeleton = GeminiProviderSkeleton()
    request = _build_gemini_provider_request()

    result = skeleton.generate(request)

    assert result.status == "not_implemented"
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.safe_to_send is False
    assert result.safe_to_display is True
    assert result.provider_error is not None
    assert "not implemented" in result.provider_error.lower()
    assert result.fallback_reason is not None
    assert "mock fallback" in result.fallback_reason.lower()


def test_gemini_provider_skeleton_accepts_generation_request_envelope() -> None:
    skeleton = GeminiProviderSkeleton(client=_FakeGeminiClient("safe"))
    request = GeminiGenerationRequest(provider_request=_build_gemini_provider_request())

    result = skeleton.generate(request)

    assert result.status in {"pass", "limited"}
    assert result.provider_response.provider_used == "gemini"
    assert result.provider_response.fallback_used is False
    assert result.safe_to_display is True


def test_gemini_provider_skeleton_with_fake_safe_client_returns_safe_provider_result() -> None:
    skeleton = GeminiProviderSkeleton(client=_FakeGeminiClient("safe"))
    request = _build_gemini_provider_request()

    result = skeleton.generate(request)

    assert result.status in {"pass", "limited"}
    assert result.provider_response.provider_used == "gemini"
    assert result.provider_response.fallback_used is False
    assert result.safe_to_send is True
    assert result.safe_to_display is True
    assert result.provider_response.raw_provider_response_allowed is False
    assert "manual review" in result.provider_response.answer.lower()


def test_gemini_provider_skeleton_blocks_fake_unsafe_output_by_safety_guard() -> None:
    skeleton = GeminiProviderSkeleton(client=_FakeGeminiClient("unsafe"))
    request = _build_gemini_provider_request()

    result = skeleton.generate(request)

    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_response.safety_status == "blocked"
    assert result.safe_to_display is False
    assert result.provider_error is not None
    assert "safety guard" in result.provider_error.lower()


def test_gemini_provider_skeleton_blocks_fake_invented_metric_output() -> None:
    skeleton = GeminiProviderSkeleton(client=_FakeGeminiClient("metric"))
    request = _build_gemini_provider_request()

    result = skeleton.generate(request)

    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_response.safety_status == "blocked"
    assert result.safe_to_display is False
    assert result.provider_error is not None


def test_gemini_provider_skeleton_blocks_fake_path_and_secret_like_output() -> None:
    skeleton = GeminiProviderSkeleton(client=_FakeGeminiClient("path"))
    request = _build_gemini_provider_request(question="Explain /Users/jo.soroush/secret.key")

    result = skeleton.generate(request)

    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_response.safety_status == "blocked"
    assert result.safe_to_display is False
    assert "/Users/jo.soroush/secret.key" not in result.provider_response.answer
    assert result.provider_error is not None


def test_gemini_provider_skeleton_handles_timeout_provider_error_rate_limit_and_empty_or_malformed_outputs() -> None:
    request = _build_gemini_provider_request()

    timeout_result = GeminiProviderSkeleton(client=_FakeExceptionClient(GeminiProviderTimeoutError("timeout"))).generate(request)
    provider_error_result = GeminiProviderSkeleton(client=_FakeExceptionClient(GeminiProviderError("error"))).generate(request)
    rate_limit_result = GeminiProviderSkeleton(client=_FakeExceptionClient(GeminiProviderRateLimitError("rate"))).generate(request)
    empty_exception_result = GeminiProviderSkeleton(client=_FakeExceptionClient(GeminiProviderEmptyResponseError("empty"))).generate(request)
    malformed_exception_result = GeminiProviderSkeleton(client=_FakeExceptionClient(GeminiProviderMalformedResponseError("malformed"))).generate(request)
    empty_result = GeminiProviderSkeleton(client=_FakeGeminiClient("empty")).generate(request)
    malformed_result = GeminiProviderSkeleton(client=_FakeGeminiClient("malformed")).generate(request)

    assert timeout_result.status == "timeout"
    assert timeout_result.provider_response.provider_used == "mock"
    assert timeout_result.provider_response.fallback_used is True
    assert timeout_result.safe_to_display is True

    assert provider_error_result.status == "provider_error"
    assert provider_error_result.provider_response.provider_used == "mock"
    assert provider_error_result.provider_response.fallback_used is True

    assert rate_limit_result.status == "rate_limit"
    assert rate_limit_result.provider_response.provider_used == "mock"
    assert rate_limit_result.provider_response.fallback_used is True

    assert empty_exception_result.status == "empty"
    assert empty_exception_result.provider_response.provider_used == "mock"
    assert empty_exception_result.provider_response.fallback_used is True

    assert malformed_exception_result.status == "malformed"
    assert malformed_exception_result.provider_response.provider_used == "mock"
    assert malformed_exception_result.provider_response.fallback_used is True

    assert empty_result.status == "empty"
    assert empty_result.provider_response.provider_used == "mock"
    assert empty_result.provider_response.fallback_used is True

    assert malformed_result.status == "malformed"
    assert malformed_result.provider_response.provider_used == "mock"
    assert malformed_result.provider_response.fallback_used is True


def test_existing_agent_provider_router_normal_mock_path_remains_unchanged() -> None:
    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Why is this image defective?",
        visible_context={},
        inspection_response={
            "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert "manual review" in response.answer.lower()


def _build_gemini_provider_request(*, question: str = "What does this chart mean?"):
    context = build_grounding_context(
        page_id="classification",
        section_id="detailed_metrics",
        component_id="classification_threshold_curve_chart",
        question=question,
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )
    sanitized_context = {
        "page_id": context.page_id,
        "section_id": context.section_id,
        "component_id": context.component_id,
        "question": "[REDACTED_SECRET]"
        if "secret.key" in question
        else context.question,
        "evidence_used": context.evidence_used,
        "limitations": context.limitations,
        "grounding_status": context.grounding_status,
    }
    return build_provider_request(
        provider_name="gemini",
        grounding_context=context,
        sanitized_context=sanitized_context,
        safety_status="pass",
        llm_enabled=True,
    )


class _FakeGeminiClient:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def generate(self, request: GeminiGenerationRequest):
        if self.mode == "safe":
            return (
                "This threshold chart summarizes validation evidence and threshold behavior. "
                "Manual review still applies."
            )
        if self.mode == "unsafe":
            return (
                "This model is production-ready, deployment-safe, and Gemini is active. "
                "Manual review is not required."
            )
        if self.mode == "metric":
            return "The threshold is 0.99 and the F1 score is 0.87. Manual review still applies."
        if self.mode == "path":
            return "Here is the file path: /Users/jo.soroush/secret.key"
        if self.mode == "empty":
            return ""
        if self.mode == "malformed":
            return {"unexpected": "structure"}
        return GeminiClientResult(text="This is a safe mocked Gemini answer. Manual review still applies.")


class _FakeExceptionClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def generate(self, request: GeminiGenerationRequest):
        raise self.exc


def test_gemini_real_provider_without_sdk_import_gate_returns_not_implemented_without_lazy_import(monkeypatch) -> None:
    calls: list[str] = []

    def fake_load_google_genai_module():
        calls.append("called")
        raise AssertionError("lazy SDK import should not be called when the import gate is disabled")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=True, gemini_key_present=True),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=False,
            api_key_resolver=lambda: "fake-key",
        ),
        sdk_loader=_RecordingSdkLoader(
            GeminiSdkLoadResult(checked=True, sdk_available=True, status="available", reason="available")
        ).loader,
    )

    result = provider.generate(_build_gemini_provider_request())

    assert calls == []
    assert result.status == "not_implemented"
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.safe_to_display is True


def test_gemini_real_provider_without_injected_sdk_client_returns_not_implemented_without_lazy_import(monkeypatch) -> None:
    calls: list[str] = []

    def fake_load_google_genai_module():
        calls.append("called")
        raise AssertionError("lazy SDK import should not be called when no injected SDK/client seam is provided")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    loader = _RecordingSdkLoader(
        GeminiSdkLoadResult(checked=True, sdk_available=True, status="available", reason="available")
    )
    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=True, gemini_key_present=True),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=True,
            api_key_resolver=lambda: "fake-key",
        ),
        sdk_loader=loader.loader,
    )

    result = provider.generate(_build_gemini_provider_request())

    assert calls == []
    assert loader.calls == 0
    assert result.status == "not_implemented"
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert "injected sdk/client seam" in (result.provider_error or "").lower()


def test_gemini_real_provider_missing_key_skips_sdk_loader_and_falls_back(monkeypatch) -> None:
    calls: list[str] = []

    def fake_load_google_genai_module():
        calls.append("called")
        raise AssertionError("lazy SDK import should not be called when the key is missing")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    loader = _RecordingSdkLoader(
        GeminiSdkLoadResult(checked=True, sdk_available=True, status="available", reason="available")
    )
    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=True, gemini_key_present=False),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=True,
            api_key_resolver=lambda: None,
        ),
        sdk_loader=loader.loader,
        sdk_module_loader=lambda: _FakeRealSdkModule("safe"),
    )

    result = provider.generate(_build_gemini_provider_request())

    assert calls == []
    assert loader.calls == 0
    assert result.status == "unavailable"
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_error is not None


def test_gemini_real_provider_disabled_by_default_skips_sdk_loading_even_with_fake_key(monkeypatch) -> None:
    sdk_module_loader_calls: list[str] = []

    def fake_load_google_genai_module():
        sdk_module_loader_calls.append("called")
        raise AssertionError("lazy SDK import should not be called when AGENT_ENABLE_LLM is disabled")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    loader = _RecordingSdkLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="available",
        )
    )
    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=False, gemini_key_present=True),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=True,
            api_key_resolver=lambda: "fake-key",
        ),
        sdk_loader=loader.loader,
        sdk_module_loader=lambda: sdk_module_loader_calls.append("called") or _FakeRealSdkModule("safe"),
    )

    result = provider.generate(_build_gemini_provider_request())

    assert sdk_module_loader_calls == []
    assert loader.calls == 0
    assert result.status == "disabled"
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_response.raw_provider_response_allowed is False
    assert result.provider_error is not None
    assert "disabled while ag" in result.provider_error.lower()
    assert result.readiness is not None
    assert result.readiness.status == "disabled"
    assert result.sdk_load_result is not None
    assert result.sdk_load_result.checked is False
    assert result.sdk_load_result.sdk_available is False
    assert result.sdk_load_result.status == "not_checked"


def test_gemini_real_provider_sdk_missing_returns_fallback_without_lazy_import(monkeypatch) -> None:
    calls: list[str] = []

    def fake_load_google_genai_module():
        calls.append("called")
        raise AssertionError("lazy SDK import should not be called when the SDK is missing")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    loader = _RecordingSdkLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=False,
            status="missing",
            reason="google-genai is missing in the injected test seam.",
            error_category="missing",
        )
    )
    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=True, gemini_key_present=True),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=True,
            api_key_resolver=lambda: "fake-key",
        ),
        sdk_loader=loader.loader,
        sdk_module_loader=lambda: _FakeRealSdkModule("safe"),
    )

    result = provider.generate(_build_gemini_provider_request())

    assert calls == []
    assert loader.calls == 1
    assert result.status == "sdk_missing"
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_error is not None
    assert "not available" in result.provider_error.lower()


def test_gemini_real_provider_with_fake_safe_sdk_client_returns_safe_provider_result(monkeypatch) -> None:
    calls: list[str] = []

    def fake_load_google_genai_module():
        calls.append("called")
        raise AssertionError("lazy SDK import should not be called when a fake SDK module is injected")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    loader = _RecordingSdkLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai is available in the injected test seam.",
        )
    )
    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=True, gemini_key_present=True),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=True,
            api_key_resolver=lambda: "fake-key",
        ),
        sdk_loader=loader.loader,
        sdk_module_loader=lambda: _FakeRealSdkModule("safe"),
    )

    result = provider.generate(_build_gemini_provider_request())

    assert calls == []
    assert loader.calls == 1
    assert result.status in {"pass", "limited"}
    assert result.provider_response.provider_used == "gemini"
    assert result.provider_response.fallback_used is False
    assert result.provider_response.raw_provider_response_allowed is False
    assert result.safe_to_display is True
    assert "manual review" in result.provider_response.answer.lower()


@pytest.mark.parametrize(
    "mode,expected_status",
    [
        ("unsafe", "blocked"),
        ("metric", "blocked"),
        ("path", "blocked"),
        ("empty", "empty"),
        ("malformed", "malformed"),
    ],
)
def test_gemini_real_provider_blocks_unsafe_or_invalid_fake_sdk_outputs(
    monkeypatch,
    mode: str,
    expected_status: str,
) -> None:
    calls: list[str] = []

    def fake_load_google_genai_module():
        calls.append("called")
        raise AssertionError("lazy SDK import should not be called when a fake SDK module is injected")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    loader = _RecordingSdkLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai is available in the injected test seam.",
        )
    )
    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=True, gemini_key_present=True),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=True,
            api_key_resolver=lambda: "fake-key",
        ),
        sdk_loader=loader.loader,
        sdk_module_loader=lambda: _FakeRealSdkModule(mode),
    )

    result = provider.generate(_build_gemini_provider_request())

    assert calls == []
    assert loader.calls == 1
    assert result.status == expected_status
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_response.provider_error is not None
    assert result.safe_to_display is (expected_status not in {"blocked"})


@pytest.mark.parametrize(
    "mode,expected_status",
    [
        ("timeout", "timeout"),
        ("rate_limit", "rate_limit"),
        ("provider_error", "provider_error"),
    ],
)
def test_gemini_real_provider_handles_fake_sdk_errors_safely(
    monkeypatch,
    mode: str,
    expected_status: str,
) -> None:
    calls: list[str] = []

    def fake_load_google_genai_module():
        calls.append("called")
        raise AssertionError("lazy SDK import should not be called when a fake SDK module is injected")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    loader = _RecordingSdkLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai is available in the injected test seam.",
        )
    )
    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=True, gemini_key_present=True),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=True,
            api_key_resolver=lambda: "fake-key",
        ),
        sdk_loader=loader.loader,
        sdk_module_loader=lambda: _FakeRealSdkModule(mode),
    )

    result = provider.generate(_build_gemini_provider_request())

    assert calls == []
    assert loader.calls == 1
    assert result.status == expected_status
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_error is not None
    assert result.safe_to_display is True


@pytest.mark.parametrize(
    "mode,expected_status,expected_phrase",
    [
        ("service_unavailable", "provider_error", "service unavailable"),
        ("too_many_requests", "rate_limit", "rate limited"),
        ("deadline_exceeded", "timeout", "timeout"),
        ("unknown_exception", "provider_error", "provider error"),
    ],
)
def test_gemini_real_provider_classifies_provider_sdk_exceptions_safely(
    monkeypatch,
    mode: str,
    expected_status: str,
    expected_phrase: str,
) -> None:
    calls: list[str] = []

    def fake_load_google_genai_module():
        calls.append("called")
        raise AssertionError("lazy SDK import should not be called when a fake SDK module is injected")

    monkeypatch.setattr(gemini_provider_module, "_load_google_genai_module", fake_load_google_genai_module)

    loader = _RecordingSdkLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai is available in the injected test seam.",
        )
    )
    provider = GeminiRealProvider(
        settings=_build_real_provider_settings(enable_llm=True, gemini_key_present=True),
        config=GeminiRealProviderConfig(
            real_provider_implemented=True,
            sdk_import_allowed=True,
            api_key_resolver=lambda: "fake-key",
        ),
        sdk_loader=loader.loader,
        sdk_module_loader=lambda: _FakeRealSdkModule(mode),
    )

    result = provider.generate(_build_gemini_provider_request())

    assert calls == []
    assert loader.calls == 1
    assert result.status == expected_status
    assert result.provider_response.provider_used == "mock"
    assert result.provider_response.fallback_used is True
    assert result.provider_error is not None
    assert expected_phrase in result.provider_error.lower()
    assert "503" not in result.provider_error
    assert "429" not in result.provider_error
    assert "deadline exceeded" not in result.provider_error.lower()
    assert result.safe_to_display is True


def _build_real_provider_settings(*, enable_llm: bool, gemini_key_present: bool):
    from src.inspection_ai.agent.provider_contracts import ProviderRuntimeSettings

    return ProviderRuntimeSettings(
        enable_llm=enable_llm,
        default_provider="mock",
        provider_order=("mock", "gemini", "grok"),
        enable_fallback=True,
        gemini_key_present=gemini_key_present,
        grok_key_present=False,
        openai_key_present=False,
    )


class _RecordingSdkLoader:
    def __init__(self, result: GeminiSdkLoadResult) -> None:
        self.result = result
        self.calls = 0

    def loader(self) -> GeminiSdkLoadResult:
        self.calls += 1
        return self.result


class _FakeRealSdkModule:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def Client(self, api_key: str):  # noqa: N802 - mimic SDK factory shape
        return _FakeRealGeminiClient(self.mode, api_key=api_key)


class _FakeRealGeminiClient:
    def __init__(self, mode: str, *, api_key: str) -> None:
        self.mode = mode
        self.api_key = api_key
        self.models = _FakeRealGeminiModels(mode)


class _FakeRealGeminiModels:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def generate_content(self, *, model: str, contents: str):
        if self.mode == "safe":
            return (
                "This threshold chart summarizes validation evidence and threshold behavior. "
                "Manual review still applies."
            )
        if self.mode == "unsafe":
            return (
                "This model is production-ready, deployment-safe, and Gemini is active. "
                "Manual review is not required."
            )
        if self.mode == "metric":
            return "The threshold is 0.99 and the F1 score is 0.87. Manual review still applies."
        if self.mode == "path":
            return "Here is the file path: /Users/jo.soroush/secret.key"
        if self.mode == "empty":
            return ""
        if self.mode == "malformed":
            return {"unexpected": "structure"}
        if self.mode == "timeout":
            raise GeminiProviderTimeoutError("timeout")
        if self.mode == "rate_limit":
            raise GeminiProviderRateLimitError("rate")
        if self.mode == "service_unavailable":
            raise ServiceUnavailable("503 ServiceUnavailable")
        if self.mode == "too_many_requests":
            raise TooManyRequests("429 TooManyRequests")
        if self.mode == "deadline_exceeded":
            raise DeadlineExceeded("deadline exceeded")
        if self.mode == "unknown_exception":
            raise RuntimeError("unexpected external SDK failure")
        if self.mode == "provider_error":
            raise GeminiProviderError("provider error")
        return GeminiClientResult(text="This is a safe mocked Gemini answer. Manual review still applies.")


class ServiceUnavailable(Exception):
    pass


class TooManyRequests(Exception):
    pass


class DeadlineExceeded(Exception):
    pass
