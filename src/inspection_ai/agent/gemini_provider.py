"""Gemini provider offline test seam for Phases G1 and G2.

This module is offline-only and intentionally does not implement real Gemini
execution. It contains the Phase G1 provider config/stub and the Phase G2
mocked-client seam so the repository can define provider behavior before any
network-enabled Gemini integration is considered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol

from .safety_guard import guard_post_generation_text
from .provider_contracts import (
    AgentProviderRequest,
    AgentProviderResponse,
    AgentProviderStatus,
    ProviderFallbackPolicy,
    ProviderReadinessResult,
    ProviderRuntimeSettings,
    build_provider_response,
)

GeminiClientErrorKind = Literal[
    "timeout",
    "provider_error",
    "rate_limit",
    "empty",
    "malformed",
]


@dataclass(frozen=True, slots=True)
class GeminiProviderConfig:
    """Offline Gemini configuration flags for the G1 stub."""

    enabled: bool = False
    api_key_present: bool = False
    model_name: str = "gemini-g1-stub"
    stub_only: bool = True

    @classmethod
    def from_runtime_settings(cls, settings: ProviderRuntimeSettings) -> "GeminiProviderConfig":
        return cls(
            enabled=settings.enable_llm,
            api_key_present=settings.gemini_key_present,
        )


class GeminiProviderDisabledError(RuntimeError):
    """Raised when the G1 Gemini stub is asked to generate an answer."""


class GeminiProviderError(RuntimeError):
    """Base error for the offline Gemini provider test seam."""


class GeminiProviderTimeoutError(GeminiProviderError):
    """Raised when a mocked Gemini client times out."""


class GeminiProviderRateLimitError(GeminiProviderError):
    """Raised when a mocked Gemini client is rate limited."""


class GeminiProviderMalformedResponseError(GeminiProviderError):
    """Raised when a mocked Gemini client payload is malformed."""


class GeminiProviderEmptyResponseError(GeminiProviderError):
    """Raised when a mocked Gemini client returns no answer text."""


@dataclass(frozen=True, slots=True)
class GeminiClientResult:
    """Offline Gemini client output used only by G2 tests."""

    text: str | None = None
    payload: Any = None
    error_kind: GeminiClientErrorKind | None = None
    error_message: str | None = None


class GeminiClientProtocol(Protocol):
    """Minimal offline protocol for future Gemini client tests."""

    def generate(self, request: AgentProviderRequest) -> GeminiClientResult:
        ...


@dataclass(frozen=True, slots=True)
class GeminiMockedClientEvaluation:
    """Safe evaluation result for mocked Gemini client outputs."""

    provider_response: AgentProviderResponse
    status: str
    safe_to_send: bool
    safe_to_display: bool
    provider_error: str | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GeminiProviderStub:
    """Non-network Gemini stub that always remains unavailable in G1."""

    config: GeminiProviderConfig = field(default_factory=GeminiProviderConfig)

    @classmethod
    def from_runtime_settings(
        cls, settings: ProviderRuntimeSettings
    ) -> "GeminiProviderStub":
        return cls(config=GeminiProviderConfig.from_runtime_settings(settings))

    def readiness(self) -> ProviderReadinessResult:
        reason = "Gemini provider stub exists in Phase G1; real execution is not implemented."
        if self.config.api_key_present:
            reason = (
                "Gemini provider stub exists in Phase G1; API key presence is recorded but real execution is not implemented."
            )
        availability = AgentProviderStatus(
            provider_name="gemini",
            configured=self.config.api_key_present,
            available=False,
            status="disabled",
            reason=reason,
            warnings=(),
        )
        return ProviderReadinessResult(
            provider_name="gemini",
            availability=availability,
            fallback_policy=ProviderFallbackPolicy(
                allow_mock_fallback=True,
                fallback_provider="mock",
                fallback_reason="Gemini Phase G1 stub is not available; mock fallback remains the safe path.",
            ),
            warnings=(
                "Gemini provider stub is present in Phase G1, but real execution is not implemented.",
            ),
        )

    def explain(self, request: AgentProviderRequest) -> AgentProviderResponse:
        raise GeminiProviderDisabledError(
            "Gemini provider stub is not implemented in Phase G1 and cannot generate responses."
        )

    generate = explain

    def evaluate_mocked_client_result(
        self,
        request: AgentProviderRequest,
        client_result: GeminiClientResult,
        *,
        allowed_evidence_values: Iterable[Any] | None = None,
    ) -> GeminiMockedClientEvaluation:
        """Translate a mocked Gemini client result into a safe provider response.

        This method exists only for G2 tests. It never performs network access
        and it never changes normal runtime routing.
        """

        request_evidence_values = _collect_request_evidence_values(request)
        allowed_values = list(request_evidence_values)
        if allowed_evidence_values is not None:
            allowed_values.extend(list(allowed_evidence_values))

        if client_result.error_kind == "timeout":
            return _build_mock_fallback_evaluation(
                request=request,
                fallback_reason="Gemini mocked client timed out; mock fallback remains the safe path.",
                provider_error="Gemini mocked client timeout.",
            )
        if client_result.error_kind == "provider_error":
            return _build_mock_fallback_evaluation(
                request=request,
                fallback_reason="Gemini mocked client errored; mock fallback remains the safe path.",
                provider_error="Gemini mocked client raised a provider error.",
            )
        if client_result.error_kind == "rate_limit":
            return _build_mock_fallback_evaluation(
                request=request,
                fallback_reason="Gemini mocked client was rate limited; mock fallback remains the safe path.",
                provider_error="Gemini mocked client was rate limited.",
            )
        if client_result.error_kind == "empty":
            return _build_mock_fallback_evaluation(
                request=request,
                fallback_reason="Gemini mocked client returned no usable text; mock fallback remains the safe path.",
                provider_error="Gemini mocked client returned an empty response.",
            )
        if client_result.error_kind == "malformed" or not isinstance(client_result.payload, (dict, type(None))):
            return _build_mock_fallback_evaluation(
                request=request,
                fallback_reason="Gemini mocked client returned a malformed payload; mock fallback remains the safe path.",
                provider_error="Gemini mocked client returned a malformed response.",
            )

        text = (client_result.text or "").strip()
        if not text:
            return _build_mock_fallback_evaluation(
                request=request,
                fallback_reason="Gemini mocked client returned no usable text; mock fallback remains the safe path.",
                provider_error="Gemini mocked client returned an empty response.",
            )

        safety_result = guard_post_generation_text(text, allowed_evidence_values=allowed_values)
        if safety_result.blocked:
            blocked_response = build_provider_response(
                answer=safety_result.sanitized_text
                or "I can’t provide that answer because the mocked Gemini output is unsafe. Manual review still applies.",
                provider_used="mock",
                fallback_used=True,
                fallback_reason="Gemini mocked client output was blocked by the safety guard; mock fallback remains the safe path.",
                grounding_status=_request_grounding_status(request),
                safety_status=safety_result.status,
                limitations=_request_limitations(request, safety_result.limitations),
                evidence_used=_request_evidence_items(request),
                provider_error="Gemini mocked client output was blocked by the safety guard.",
            )
            return GeminiMockedClientEvaluation(
                provider_response=blocked_response,
                status=safety_result.status,
                safe_to_send=False,
                safe_to_display=False,
                provider_error="Gemini mocked client output was blocked by the safety guard.",
                fallback_reason=blocked_response.fallback_reason,
            )

        response = build_provider_response(
            answer=safety_result.sanitized_text or text,
            provider_used="gemini",
            fallback_used=False,
            fallback_reason=None,
            grounding_status=_request_grounding_status(request),
            safety_status=safety_result.status,
            limitations=_request_limitations(request, safety_result.limitations),
            evidence_used=_request_evidence_items(request),
            provider_error=None,
        )
        return GeminiMockedClientEvaluation(
            provider_response=response,
            status=safety_result.status,
            safe_to_send=safety_result.safe_to_send,
            safe_to_display=safety_result.safe_to_display,
        )

    translate_mocked_client_result = evaluate_mocked_client_result


def evaluate_gemini_provider_readiness(
    settings: ProviderRuntimeSettings,
) -> ProviderReadinessResult:
    """Return the G1 Gemini stub readiness snapshot without any network access."""
    return GeminiProviderStub.from_runtime_settings(settings).readiness()


def _collect_request_evidence_values(request: AgentProviderRequest) -> list[Any]:
    evidence_values: list[Any] = []
    evidence_items = request.grounding_context.get("evidence_used", [])
    if isinstance(evidence_items, list):
        for item in evidence_items:
            if isinstance(item, dict):
                evidence_values.append(item.get("value"))
            else:
                evidence_values.append(item)
    return evidence_values


def _request_grounding_status(request: AgentProviderRequest) -> str:
    grounding_status = request.grounding_context.get("grounding_status")
    if isinstance(grounding_status, str) and grounding_status.strip():
        return grounding_status
    return "insufficient_evidence"


def _request_limitations(
    request: AgentProviderRequest,
    extra_limitations: Iterable[str] | None = None,
) -> list[str]:
    limitations: list[str] = []
    grounding_limitations = request.grounding_context.get("limitations", [])
    if isinstance(grounding_limitations, list):
        limitations.extend(str(item) for item in grounding_limitations if str(item).strip())
    if extra_limitations is not None:
        limitations.extend(str(item) for item in extra_limitations if str(item).strip())
    return list(dict.fromkeys(limitations))


def _request_evidence_items(request: AgentProviderRequest) -> list[dict[str, Any]]:
    evidence_items: list[dict[str, Any]] = []
    raw_items = request.grounding_context.get("evidence_used", [])
    if not isinstance(raw_items, list):
        return evidence_items
    for item in raw_items:
        if isinstance(item, dict):
            evidence_items.append(dict(item))
    return evidence_items


def _build_mock_fallback_evaluation(
    *,
    request: AgentProviderRequest,
    fallback_reason: str,
    provider_error: str,
) -> GeminiMockedClientEvaluation:
    response = build_provider_response(
        answer=(
            "I can’t provide a Gemini answer in the mocked-client validation path. "
            "Mock fallback remains the safe path. Manual review still applies."
        ),
        provider_used="mock",
        fallback_used=True,
        fallback_reason=fallback_reason,
        grounding_status=_request_grounding_status(request),
        safety_status="pass",
        limitations=_request_limitations(request, [fallback_reason]),
        evidence_used=_request_evidence_items(request),
        provider_error=provider_error,
    )
    return GeminiMockedClientEvaluation(
        provider_response=response,
        status="limited",
        safe_to_send=True,
        safe_to_display=True,
        provider_error=provider_error,
        fallback_reason=fallback_reason,
    )
