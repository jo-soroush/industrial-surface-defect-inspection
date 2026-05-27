"""Tests for the Gemini Phase G3 readiness scaffolding."""

from __future__ import annotations

import ast
from pathlib import Path

from src.inspection_ai.agent.context_builder import build_grounding_context
from src.inspection_ai.agent.gemini_provider import (
    GeminiSdkLoadResult,
    GeminiSdkLoader,
    check_gemini_sdk_available,
    evaluate_gemini_g3_readiness,
    load_gemini_sdk_status,
)
from src.inspection_ai.agent.provider_contracts import ProviderRuntimeSettings
from src.inspection_ai.agent.provider_router import AgentProviderRouter


REPO_ROOT = Path(__file__).resolve().parents[2]
GEMINI_PROVIDER_PATH = REPO_ROOT / "src/inspection_ai/agent/gemini_provider.py"


def test_gemini_g3_readiness_is_disabled_when_llm_disabled_and_key_missing() -> None:
    loader = _RecordingLoader(GeminiSdkLoadResult(status="missing", sdk_available=False))
    readiness = evaluate_gemini_g3_readiness(
        ProviderRuntimeSettings(
            enable_llm=False,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=False,
            grok_key_present=False,
            openai_key_present=False,
        ),
        sdk_loader=loader.loader,
    )

    assert readiness.status == "disabled"
    assert readiness.available is False
    assert readiness.gates.llm_enabled is False
    assert readiness.gates.api_key_present is False
    assert readiness.gates.sdk_available is False
    assert readiness.gates.activation_allowed is False
    assert readiness.availability.available is False
    assert readiness.fallback_policy.allow_mock_fallback is True
    assert readiness.fallback_policy.fallback_provider == "mock"
    assert "disabled" in readiness.reason.lower()
    assert "key" not in readiness.reason.lower()
    assert any("mock fallback" in warning.lower() for warning in readiness.warnings)
    assert loader.calls == 0


def test_gemini_g3_readiness_stays_disabled_when_key_present_but_llm_disabled() -> None:
    loader = _RecordingLoader(GeminiSdkLoadResult(status="available", sdk_available=True))
    readiness = evaluate_gemini_g3_readiness(
        ProviderRuntimeSettings(
            enable_llm=False,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=False,
            openai_key_present=False,
        ),
        sdk_loader=loader.loader,
    )

    assert readiness.status == "disabled"
    assert readiness.available is False
    assert readiness.gates.api_key_present is True
    assert readiness.gates.llm_enabled is False
    assert readiness.gates.activation_allowed is False
    assert readiness.availability.configured is True
    assert "disabled" in readiness.availability.reason.lower()
    assert "mock fallback" in readiness.fallback_policy.fallback_reason.lower()
    assert loader.calls == 0


def test_gemini_g3_readiness_is_unavailable_when_key_missing_and_llm_enabled() -> None:
    loader = _RecordingLoader(GeminiSdkLoadResult(status="available", sdk_available=True))
    readiness = evaluate_gemini_g3_readiness(
        ProviderRuntimeSettings(
            enable_llm=True,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=False,
            grok_key_present=False,
            openai_key_present=False,
        ),
        sdk_loader=loader.loader,
    )

    assert readiness.status == "unavailable"
    assert readiness.available is False
    assert readiness.gates.llm_enabled is True
    assert readiness.gates.api_key_present is False
    assert readiness.gates.activation_allowed is False
    assert readiness.availability.status == "unavailable"
    assert "missing" in readiness.reason.lower()
    assert "google-genai" not in readiness.reason.lower()
    assert loader.calls == 0


def test_gemini_g3_readiness_is_sdk_missing_when_sdk_is_not_available() -> None:
    loader = _RecordingLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=False,
            status="missing",
            reason="google-genai is missing in the test seam.",
            error_category="missing",
        )
    )
    readiness = evaluate_gemini_g3_readiness(
        ProviderRuntimeSettings(
            enable_llm=True,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=False,
            openai_key_present=False,
        ),
        sdk_loader=loader.loader,
    )

    assert readiness.status == "sdk_missing"
    assert readiness.available is False
    assert readiness.gates.llm_enabled is True
    assert readiness.gates.api_key_present is True
    assert readiness.gates.sdk_available is False
    assert readiness.gates.activation_allowed is False
    assert readiness.availability.status == "unavailable"
    assert "sdk" in readiness.reason.lower()
    assert "google-genai" in readiness.warnings[0].lower()
    assert loader.calls == 1


def test_gemini_g3_readiness_reports_load_error_safely() -> None:
    loader = _RecordingLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=False,
            status="load_error",
            reason="SDK import failed in the injected test seam.",
            error_category="import_error",
        )
    )
    readiness = evaluate_gemini_g3_readiness(
        ProviderRuntimeSettings(
            enable_llm=True,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=False,
            openai_key_present=False,
        ),
        sdk_loader=loader.loader,
    )

    assert readiness.status == "load_error"
    assert readiness.available is False
    assert readiness.gates.sdk_checked is True
    assert readiness.gates.sdk_status == "load_error"
    assert readiness.gates.activation_allowed is False
    assert readiness.availability.status == "unavailable"
    assert "load failed" in readiness.reason.lower()
    assert loader.calls == 1


def test_gemini_g3_readiness_uses_loader_available_but_stays_not_implemented() -> None:
    loader = _RecordingLoader(
        GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google-genai is available in the injected test seam.",
        )
    )
    readiness = evaluate_gemini_g3_readiness(
        ProviderRuntimeSettings(
            enable_llm=True,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=False,
            openai_key_present=False,
        ),
        sdk_loader=loader.loader,
    )

    assert readiness.status == "not_implemented"
    assert readiness.available is False
    assert readiness.gates.sdk_available is True
    assert readiness.gates.real_provider_implemented is False
    assert readiness.gates.activation_allowed is False
    assert readiness.availability.available is False
    assert loader.calls == 1


def test_gemini_g3_readiness_remains_not_implemented_even_when_all_gates_pass() -> None:
    readiness = evaluate_gemini_g3_readiness(
        ProviderRuntimeSettings(
            enable_llm=True,
            default_provider="mock",
            provider_order=("mock", "gemini", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=False,
            openai_key_present=False,
        ),
        sdk_available=True,
    )

    assert readiness.status == "not_implemented"
    assert readiness.available is False
    assert readiness.gates.llm_enabled is True
    assert readiness.gates.api_key_present is True
    assert readiness.gates.sdk_available is True
    assert readiness.gates.provider_allowed is True
    assert readiness.gates.real_provider_implemented is False
    assert readiness.gates.activation_allowed is False
    assert readiness.availability.available is False
    assert "not implemented" in readiness.reason.lower()
    assert "mock fallback" in readiness.fallback_policy.fallback_reason.lower()


def test_gemini_g3_readiness_is_gated_when_provider_order_disallows_gemini() -> None:
    readiness = evaluate_gemini_g3_readiness(
        ProviderRuntimeSettings(
            enable_llm=True,
            default_provider="mock",
            provider_order=("mock", "grok"),
            enable_fallback=True,
            gemini_key_present=True,
            grok_key_present=False,
            openai_key_present=False,
        ),
        sdk_available=True,
    )

    assert readiness.status == "gated"
    assert readiness.available is False
    assert readiness.gates.provider_allowed is False
    assert readiness.gates.activation_allowed is False
    assert readiness.availability.available is False
    assert "gated" in readiness.reason.lower()


def test_gemini_g3_readiness_module_stays_offline_only() -> None:
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


def test_gemini_g3_readiness_does_not_change_mock_router_path() -> None:
    router = AgentProviderRouter()
    response = router.explain(
        build_grounding_context(
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
    )

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert "manual review" in response.answer.lower()


def test_gemini_g3_sdk_loader_status_helpers_are_available() -> None:
    explicit_available = check_gemini_sdk_available(sdk_available=True)
    explicit_missing = load_gemini_sdk_status(sdk_available=False)

    assert explicit_available.checked is True
    assert explicit_available.sdk_available is True
    assert explicit_available.status == "available"
    assert explicit_missing.checked is True
    assert explicit_missing.sdk_available is False
    assert explicit_missing.status == "missing"


class _RecordingLoader:
    def __init__(self, result: GeminiSdkLoadResult) -> None:
        self.result = result
        self.calls = 0

    def loader(self) -> GeminiSdkLoadResult:
        self.calls += 1
        return self.result
