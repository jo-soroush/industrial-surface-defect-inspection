"""Gemini provider scaffolding for Phases G1 through G3.

This module contains the Phase G1 provider config/stub, the Phase G2
mocked-client seam, the Phase G3 readiness scaffolding, the G3 lazy SDK loader
boundary, and a disabled-by-default real-provider execution boundary. That
real-provider boundary is not wired into normal ``/agent/explain`` routing,
must use lazy SDK loading only, and keeps the normal runtime mock-first.
Importing this module does not make any real Gemini API call.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Protocol

from .safety_guard import guard_post_generation_text, guard_pre_generation_context
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
GeminiG3ReadinessStatus = Literal[
    "disabled",
    "unavailable",
    "sdk_missing",
    "load_error",
    "not_implemented",
    "gated",
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


@dataclass(frozen=True, slots=True)
class GeminiSdkStatus:
    """Offline SDK status metadata for the G3 readiness scaffolding."""

    sdk_name: str = "google-genai"
    sdk_available: bool = False
    import_style: str = "from google import genai"
    note: str = "SDK availability is modeled without importing the package."


@dataclass(frozen=True, slots=True)
class GeminiSdkLoadResult:
    """Offline SDK load status used by the G3 readiness scaffolding."""

    checked: bool = False
    sdk_available: bool = False
    status: Literal["not_checked", "missing", "available", "load_error"] = "not_checked"
    reason: str = "SDK availability has not been checked."
    error_category: str | None = None
    sdk_name: str = "google-genai"
    import_style: str = "from google import genai"


class GeminiSdkLoader:
    """Injectable offline SDK loader boundary for future Gemini readiness."""

    def __init__(
        self,
        checker: Callable[[], GeminiSdkLoadResult] | None = None,
        *,
        sdk_name: str = "google-genai",
        import_style: str = "from google import genai",
    ) -> None:
        self._checker = checker
        self.sdk_name = sdk_name
        self.import_style = import_style

    def load_status(self) -> GeminiSdkLoadResult:
        if self._checker is None:
            return GeminiSdkLoadResult(
                checked=False,
                sdk_available=False,
                status="not_checked",
                reason="SDK availability not checked in this slice.",
                sdk_name=self.sdk_name,
                import_style=self.import_style,
            )

        result = self._checker()
        if result.sdk_name != self.sdk_name or result.import_style != self.import_style:
            result = GeminiSdkLoadResult(
                checked=result.checked,
                sdk_available=result.sdk_available,
                status=result.status,
                reason=result.reason,
                error_category=result.error_category,
                sdk_name=self.sdk_name,
                import_style=self.import_style,
            )
        return result


@dataclass(frozen=True, slots=True)
class GeminiReadinessGates:
    """Non-secret Gemini G3 readiness gates."""

    llm_enabled: bool = False
    api_key_present: bool = False
    provider_allowed: bool = True
    activation_allowed: bool = False
    sdk_available: bool = False
    real_provider_implemented: bool = False
    sdk_checked: bool = False
    sdk_status: str = "not_checked"
    sdk_reason: str = "SDK availability not checked."


@dataclass(frozen=True, slots=True)
class GeminiG3Readiness:
    """Offline G3 readiness snapshot for future Gemini provider execution."""

    provider_name: Literal["gemini"] = "gemini"
    status: GeminiG3ReadinessStatus = "not_implemented"
    available: bool = False
    reason: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)
    fallback_policy: ProviderFallbackPolicy = field(default_factory=ProviderFallbackPolicy)
    sdk_status: GeminiSdkStatus = field(default_factory=GeminiSdkStatus)
    gates: GeminiReadinessGates = field(default_factory=GeminiReadinessGates)
    availability: AgentProviderStatus = field(
        default_factory=lambda: AgentProviderStatus(
            provider_name="gemini",
            configured=False,
            available=False,
            status="disabled",
            reason="Gemini G3 readiness has not been activated.",
            warnings=(),
        )
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
class GeminiGenerationRequest:
    """Internal request envelope for the injected Gemini provider skeleton."""

    provider_request: AgentProviderRequest
    allowed_evidence_values: tuple[Any, ...] = field(default_factory=tuple)
    client_name: str = "injected-sdk-seam"


@dataclass(frozen=True, slots=True)
class GeminiGenerationResult:
    """Safe result envelope for the injected Gemini provider skeleton."""

    provider_response: AgentProviderResponse
    status: str
    safe_to_send: bool
    safe_to_display: bool
    provider_error: str | None = None
    fallback_reason: str | None = None
    client_name: str = "injected-sdk-seam"


class GeminiInjectedClientProtocol(Protocol):
    """Protocol for an injected SDK-like client used only by tests."""

    def generate(self, request: GeminiGenerationRequest) -> Any:
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


@dataclass(frozen=True, slots=True)
class GeminiProviderSkeleton:
    """Offline provider skeleton that only works with an injected test client."""

    client: GeminiInjectedClientProtocol | Callable[[GeminiGenerationRequest], Any] | None = None
    client_name: str = "injected-sdk-seam"

    def generate(
        self,
        request: AgentProviderRequest | GeminiGenerationRequest,
        *,
        allowed_evidence_values: Iterable[Any] | None = None,
    ) -> GeminiGenerationResult:
        generation_request = _coerce_gemini_generation_request(
            request,
            allowed_evidence_values=allowed_evidence_values,
            client_name=self.client_name,
        )

        if self.client is None:
            return _build_skeleton_not_implemented_generation_result(generation_request)

        try:
            raw_result = self._invoke_client(generation_request)
        except GeminiProviderTimeoutError as exc:
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="timeout",
                fallback_reason="Gemini injected client timed out; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini injected client timeout.",
            )
        except GeminiProviderRateLimitError as exc:
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="rate_limit",
                fallback_reason="Gemini injected client was rate limited; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini injected client was rate limited.",
            )
        except GeminiProviderEmptyResponseError as exc:
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="empty",
                fallback_reason="Gemini injected client returned no usable text; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini injected client returned an empty response.",
            )
        except GeminiProviderMalformedResponseError as exc:
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="malformed",
                fallback_reason="Gemini injected client returned a malformed payload; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini injected client returned a malformed response.",
            )
        except GeminiProviderError as exc:
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="provider_error",
                fallback_reason="Gemini injected client raised a provider error; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini injected client raised a provider error.",
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="provider_error",
                fallback_reason="Gemini injected client raised an unexpected error; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini injected client raised an unexpected error.",
            )

        normalized_result = _coerce_gemini_client_result(raw_result)
        if normalized_result is None:
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="malformed",
                fallback_reason="Gemini injected client returned a malformed payload; mock fallback remains the safe path.",
                provider_error="Gemini injected client returned a malformed response.",
            )

        if normalized_result.error_kind == "timeout":
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="timeout",
                fallback_reason="Gemini injected client timed out; mock fallback remains the safe path.",
                provider_error="Gemini injected client timeout.",
            )
        if normalized_result.error_kind == "provider_error":
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="provider_error",
                fallback_reason="Gemini injected client raised a provider error; mock fallback remains the safe path.",
                provider_error="Gemini injected client raised a provider error.",
            )
        if normalized_result.error_kind == "rate_limit":
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="rate_limit",
                fallback_reason="Gemini injected client was rate limited; mock fallback remains the safe path.",
                provider_error="Gemini injected client was rate limited.",
            )
        if normalized_result.error_kind == "empty":
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="empty",
                fallback_reason="Gemini injected client returned no usable text; mock fallback remains the safe path.",
                provider_error="Gemini injected client returned an empty response.",
            )
        if normalized_result.error_kind == "malformed":
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="malformed",
                fallback_reason="Gemini injected client returned a malformed payload; mock fallback remains the safe path.",
                provider_error="Gemini injected client returned a malformed response.",
            )

        if normalized_result.text is None or not normalized_result.text.strip():
            return _build_skeleton_fallback_generation_result(
                generation_request,
                status="empty",
                fallback_reason="Gemini injected client returned no usable text; mock fallback remains the safe path.",
                provider_error="Gemini injected client returned an empty response.",
            )

        mocked_evaluation = GeminiProviderStub().evaluate_mocked_client_result(
            generation_request.provider_request,
            normalized_result,
            allowed_evidence_values=generation_request.allowed_evidence_values,
        )
        return GeminiGenerationResult(
            provider_response=mocked_evaluation.provider_response,
            status=mocked_evaluation.status,
            safe_to_send=mocked_evaluation.safe_to_send,
            safe_to_display=mocked_evaluation.safe_to_display,
            provider_error=mocked_evaluation.provider_error,
            fallback_reason=mocked_evaluation.fallback_reason,
            client_name=generation_request.client_name,
        )

    def _invoke_client(self, request: GeminiGenerationRequest) -> Any:
        if self.client is None:  # pragma: no cover - defensive guard
            return None
        if hasattr(self.client, "generate"):
            return self.client.generate(request)
        if callable(self.client):
            return self.client(request)
        raise GeminiProviderError("Injected Gemini client does not support generate().")


def _default_gemini_api_key_resolver() -> str | None:
    """Resolve the Gemini API key from the environment only when explicitly requested."""
    value = os.getenv("GEMINI_API_KEY")
    return value if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True, slots=True)
class GeminiRealProviderConfig:
    """Disabled-by-default configuration for the future real Gemini execution path."""

    model_name: str = "gemini-2.0-flash"
    client_name: str = "google-genai"
    real_provider_implemented: bool = False
    sdk_import_allowed: bool = False
    fallback_enabled: bool = True
    api_key_resolver: Callable[[], str | None] | None = None


@dataclass(frozen=True, slots=True)
class GeminiRealGenerationResult:
    """Safe result envelope for the real-provider execution boundary."""

    provider_response: AgentProviderResponse
    status: str
    safe_to_send: bool
    safe_to_display: bool
    provider_error: str | None = None
    fallback_reason: str | None = None
    client_name: str = "google-genai"
    readiness: GeminiG3Readiness | None = None
    sdk_load_result: GeminiSdkLoadResult | None = None


@dataclass(frozen=True, slots=True)
class GeminiRealProvider:
    """Disabled-by-default real Gemini execution boundary.

    The helper is explicit and offline-safe by default. It only attempts a
    lazy SDK import when sdk_import_allowed=True, the future real-provider gate
    is enabled, and callers inject the required seams.
    """

    settings: ProviderRuntimeSettings
    config: GeminiRealProviderConfig = field(default_factory=GeminiRealProviderConfig)
    sdk_loader: GeminiSdkLoader | Callable[[], GeminiSdkLoadResult] | None = None
    sdk_module_loader: Callable[[], Any] | None = None
    client_factory: Callable[[Any, str, str], Any] | None = None

    def readiness(self) -> GeminiG3Readiness:
        return evaluate_gemini_g3_readiness(
            self.settings,
            sdk_loader=self.sdk_loader,
            real_provider_implemented=self.config.real_provider_implemented,
        )

    def generate(
        self,
        request: AgentProviderRequest | GeminiGenerationRequest,
        *,
        allowed_evidence_values: Iterable[Any] | None = None,
    ) -> GeminiRealGenerationResult:
        generation_request = _coerce_gemini_generation_request(
            request,
            allowed_evidence_values=allowed_evidence_values,
            client_name=self.config.client_name,
        )
        synthetic_context = _build_synthetic_grounding_context(generation_request.provider_request)
        pre_guard = guard_pre_generation_context(synthetic_context)

        if pre_guard.blocked:
            readiness = evaluate_gemini_g3_readiness(
                self.settings,
                real_provider_implemented=self.config.real_provider_implemented,
            )
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=None,
                status="blocked",
                fallback_reason="Gemini real-provider prompt was blocked by the safety guard; mock fallback remains the safe path.",
                provider_error="Gemini real-provider prompt was blocked by the safety guard.",
                safe_to_display=False,
                blocked=True,
            )

        if not self.config.real_provider_implemented:
            readiness = evaluate_gemini_g3_readiness(
                self.settings,
                real_provider_implemented=False,
            )
            return _build_real_generation_not_implemented_result(
                generation_request,
                readiness=readiness,
                provider_error="Gemini real provider execution is not implemented in this slice.",
            )

        if not self.config.sdk_import_allowed:
            readiness = evaluate_gemini_g3_readiness(
                self.settings,
                real_provider_implemented=self.config.real_provider_implemented,
            )
            return _build_real_generation_not_implemented_result(
                generation_request,
                readiness=readiness,
                provider_error="Gemini real provider execution is disabled until the lazy SDK import gate is explicitly opened.",
            )

        if self.sdk_module_loader is None and self.client_factory is None:
            readiness = evaluate_gemini_g3_readiness(
                self.settings,
                real_provider_implemented=self.config.real_provider_implemented,
            )
            return _build_real_generation_not_implemented_result(
                generation_request,
                readiness=readiness,
                provider_error="Gemini real provider execution in this slice requires an injected SDK/client seam and does not attempt a real SDK import.",
            )

        sdk_load_result = _resolve_gemini_sdk_load_result(
            settings=self.settings,
            sdk_available=None,
            sdk_loader=self.sdk_loader,
            sdk_load_result=None,
        )
        readiness = evaluate_gemini_g3_readiness(
            self.settings,
            sdk_load_result=sdk_load_result,
            real_provider_implemented=self.config.real_provider_implemented,
        )
        if not readiness.gates.activation_allowed:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status=readiness.status,
                fallback_reason=readiness.fallback_policy.fallback_reason
                or "Gemini remains disabled by default; mock fallback remains the safe path.",
                provider_error=readiness.reason,
            )

        if not sdk_load_result.sdk_available:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status=sdk_load_result.status,
                fallback_reason=readiness.fallback_policy.fallback_reason
                or "Gemini remains unavailable; mock fallback remains the safe path.",
                provider_error=sdk_load_result.reason,
            )

        api_key = self._resolve_api_key()
        if not api_key:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="unavailable",
                fallback_reason="Gemini API key is missing; mock fallback remains the safe path.",
                provider_error="Gemini API key is missing or not configured.",
            )

        try:
            sdk_module = self._load_sdk_module()
        except Exception as exc:  # pragma: no cover - defensive fallback
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="load_error",
                fallback_reason="Gemini SDK import failed; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini SDK import failed.",
            )

        try:
            client = self._build_real_client(sdk_module, api_key)
        except Exception as exc:  # pragma: no cover - defensive fallback
            exception_classification = _classify_real_provider_exception(exc)
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status=exception_classification.status,
                fallback_reason=exception_classification.fallback_reason,
                provider_error=exception_classification.provider_error,
            )

        prompt = _build_real_gemini_prompt(
            generation_request.provider_request,
            sanitized_context=pre_guard.sanitized_context,
        )

        try:
            raw_result = self._invoke_real_client(client, prompt, generation_request.provider_request)
        except GeminiProviderTimeoutError as exc:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="timeout",
                fallback_reason="Gemini real provider timed out; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini real provider timeout.",
            )
        except GeminiProviderRateLimitError as exc:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="rate_limit",
                fallback_reason="Gemini real provider was rate limited; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini real provider was rate limited.",
            )
        except GeminiProviderEmptyResponseError as exc:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="empty",
                fallback_reason="Gemini real provider returned no usable text; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini real provider returned an empty response.",
            )
        except GeminiProviderMalformedResponseError as exc:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="malformed",
                fallback_reason="Gemini real provider returned a malformed payload; mock fallback remains the safe path.",
                provider_error=str(exc) or "Gemini real provider returned a malformed response.",
            )
        except GeminiProviderError as exc:
            exception_classification = _classify_real_provider_exception(exc)
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status=exception_classification.status,
                fallback_reason=exception_classification.fallback_reason,
                provider_error=exception_classification.provider_error,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            exception_classification = _classify_real_provider_exception(exc)
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status=exception_classification.status,
                fallback_reason=exception_classification.fallback_reason,
                provider_error=exception_classification.provider_error,
            )

        normalized_result = _coerce_gemini_client_result(raw_result)
        if normalized_result is None:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="malformed",
                fallback_reason="Gemini real provider returned a malformed payload; mock fallback remains the safe path.",
                provider_error="Gemini real provider returned a malformed response.",
            )

        if normalized_result.error_kind == "timeout":
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="timeout",
                fallback_reason="Gemini real provider timed out; mock fallback remains the safe path.",
                provider_error="Gemini real provider timeout.",
            )
        if normalized_result.error_kind == "provider_error":
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="provider_error",
                fallback_reason="Gemini real provider raised a provider error; mock fallback remains the safe path.",
                provider_error="Gemini real provider raised a provider error.",
            )
        if normalized_result.error_kind == "rate_limit":
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="rate_limit",
                fallback_reason="Gemini real provider was rate limited; mock fallback remains the safe path.",
                provider_error="Gemini real provider was rate limited.",
            )
        if normalized_result.error_kind == "empty":
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="empty",
                fallback_reason="Gemini real provider returned no usable text; mock fallback remains the safe path.",
                provider_error="Gemini real provider returned an empty response.",
            )
        if normalized_result.error_kind == "malformed":
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="malformed",
                fallback_reason="Gemini real provider returned a malformed payload; mock fallback remains the safe path.",
                provider_error="Gemini real provider returned a malformed response.",
            )

        if normalized_result.text is None or not normalized_result.text.strip():
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status="empty",
                fallback_reason="Gemini real provider returned no usable text; mock fallback remains the safe path.",
                provider_error="Gemini real provider returned an empty response.",
            )

        post_guard = guard_post_generation_text(
            normalized_result.text,
            grounding_context=synthetic_context,
            allowed_evidence_values=generation_request.allowed_evidence_values,
        )
        if post_guard.blocked:
            return _build_real_generation_fallback_result(
                generation_request,
                readiness=readiness,
                sdk_load_result=sdk_load_result,
                status=post_guard.status,
                fallback_reason="Gemini real provider output was blocked by the safety guard; mock fallback remains the safe path.",
                provider_error="Gemini real provider output was blocked by the safety guard.",
                safe_to_display=False,
                blocked=True,
            )

        response = build_provider_response(
            answer=post_guard.sanitized_text or normalized_result.text,
            provider_used="gemini",
            fallback_used=False,
            fallback_reason=None,
            grounding_status=_request_grounding_status(generation_request.provider_request),
            safety_status=post_guard.status,
            limitations=_request_limitations(
                generation_request.provider_request,
                post_guard.limitations,
            ),
            evidence_used=_request_evidence_items(generation_request.provider_request),
            provider_error=None,
        )
        return GeminiRealGenerationResult(
            provider_response=response,
            status=post_guard.status,
            safe_to_send=post_guard.safe_to_send,
            safe_to_display=post_guard.safe_to_display,
            provider_error=None,
            fallback_reason=None,
            client_name=self.config.client_name,
            readiness=readiness,
            sdk_load_result=sdk_load_result,
        )

    def _resolve_api_key(self) -> str | None:
        if self.config.api_key_resolver is None:
            return None
        value = self.config.api_key_resolver()
        if isinstance(value, str) and value.strip():
            return value
        return None

    def _load_sdk_module(self) -> Any:
        if self.sdk_module_loader is not None:
            return self.sdk_module_loader()
        return _load_google_genai_module()

    def _build_real_client(self, sdk_module: Any, api_key: str) -> Any:
        if self.client_factory is not None:
            return self.client_factory(sdk_module, api_key, self.config.model_name)
        client_cls = getattr(sdk_module, "Client", None)
        if callable(client_cls):
            try:
                return client_cls(api_key=api_key)
            except TypeError:
                return client_cls(api_key)
        raise GeminiProviderError("The loaded Gemini SDK does not expose a Client factory.")

    def _invoke_real_client(self, client: Any, prompt: str, request: AgentProviderRequest) -> Any:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            return client.models.generate_content(model=self.config.model_name, contents=prompt)
        if hasattr(client, "generate_content"):
            return client.generate_content(model=self.config.model_name, contents=prompt)
        if hasattr(client, "generate"):
            return client.generate(request)
        if callable(client):
            return client(prompt)
        raise GeminiProviderError("The loaded Gemini client does not expose a supported generation method.")


def generate_with_real_gemini_provider(
    request: AgentProviderRequest | GeminiGenerationRequest,
    *,
    settings: ProviderRuntimeSettings,
    config: GeminiRealProviderConfig | None = None,
    sdk_loader: GeminiSdkLoader | Callable[[], GeminiSdkLoadResult] | None = None,
    sdk_module_loader: Callable[[], Any] | None = None,
    client_factory: Callable[[Any, str, str], Any] | None = None,
    allowed_evidence_values: Iterable[Any] | None = None,
) -> GeminiRealGenerationResult:
    """Execute the disabled-by-default real Gemini boundary with injected seams only."""

    provider = GeminiRealProvider(
        settings=settings,
        config=config or GeminiRealProviderConfig(),
        sdk_loader=sdk_loader,
        sdk_module_loader=sdk_module_loader,
        client_factory=client_factory,
    )
    return provider.generate(request, allowed_evidence_values=allowed_evidence_values)


def evaluate_gemini_provider_readiness(
    settings: ProviderRuntimeSettings,
) -> ProviderReadinessResult:
    """Return the G1 Gemini stub readiness snapshot without any network access."""
    return GeminiProviderStub.from_runtime_settings(settings).readiness()


def evaluate_gemini_g3_readiness(
    settings: ProviderRuntimeSettings,
    *,
    sdk_available: bool | None = None,
    sdk_loader: GeminiSdkLoader | Callable[[], GeminiSdkLoadResult] | None = None,
    sdk_load_result: GeminiSdkLoadResult | None = None,
    real_provider_implemented: bool = False,
) -> GeminiG3Readiness:
    """Return the Phase G3 readiness snapshot without importing or calling Gemini."""

    provider_allowed = "gemini" in settings.provider_order or settings.default_provider == "gemini"
    sdk_result = _resolve_gemini_sdk_load_result(
        settings=settings,
        sdk_available=sdk_available,
        sdk_loader=sdk_loader,
        sdk_load_result=sdk_load_result,
    )
    gates = GeminiReadinessGates(
        llm_enabled=settings.enable_llm,
        api_key_present=settings.gemini_key_present,
        provider_allowed=provider_allowed,
        activation_allowed=(
            settings.enable_llm
            and settings.gemini_key_present
            and sdk_result.sdk_available
            and provider_allowed
            and real_provider_implemented
        ),
        sdk_available=sdk_result.sdk_available,
        real_provider_implemented=real_provider_implemented,
        sdk_checked=sdk_result.checked,
        sdk_status=sdk_result.status,
        sdk_reason=sdk_result.reason,
    )

    if not settings.enable_llm:
        status: GeminiG3ReadinessStatus = "disabled"
        reason = "Gemini remains disabled while AGENT_ENABLE_LLM is false."
        warnings = (
            "Gemini remains disabled while AGENT_ENABLE_LLM is false.",
            "Mock fallback remains the safe path.",
        )
    elif not settings.gemini_key_present:
        status = "unavailable"
        reason = "Gemini API key is missing or not configured."
        warnings = (
            "Gemini is unavailable because GEMINI_API_KEY is missing.",
            "Mock fallback remains the safe path.",
        )
    elif sdk_result.status == "load_error":
        status = "load_error"
        reason = "Gemini SDK load failed in this slice."
        warnings = (
            "Gemini is unavailable because the SDK loader reported an error.",
            "Mock fallback remains the safe path.",
        )
    elif not sdk_result.sdk_available:
        status = "sdk_missing"
        reason = "The google-genai SDK is not available in this slice."
        warnings = (
            "Gemini is unavailable because the google-genai SDK is missing.",
            "Mock fallback remains the safe path.",
        )
    elif not provider_allowed:
        status = "gated"
        reason = "Gemini is gated by the current provider order."
        warnings = (
            "Gemini is gated by the current provider order.",
            "Mock fallback remains the safe path.",
        )
    elif not real_provider_implemented:
        status = "not_implemented"
        reason = "Real Gemini provider execution is not implemented in this slice."
        warnings = (
            "Gemini remains not implemented in the current slice.",
            "Mock fallback remains the safe path.",
        )
    else:
        status = "gated"
        reason = "Gemini remains gated until the real provider implementation slice is approved."
        warnings = (
            "Gemini is gated because the real provider implementation slice is not yet active.",
            "Mock fallback remains the safe path.",
        )

    availability_status = "disabled" if status in {"disabled", "sdk_missing", "not_implemented", "gated"} else "unavailable"
    availability_reason = reason
    if status == "sdk_missing":
        availability_status = "unavailable"
    if status == "unavailable":
        availability_status = "unavailable"

    availability = AgentProviderStatus(
        provider_name="gemini",
        configured=settings.gemini_key_present,
        available=False,
        status=availability_status,
        reason=availability_reason,
        warnings=(),
    )

    return GeminiG3Readiness(
        provider_name="gemini",
        status=status,
        available=False,
        reason=reason,
        warnings=warnings,
        fallback_policy=ProviderFallbackPolicy(
            allow_mock_fallback=True,
            fallback_provider="mock",
            fallback_reason="Gemini remains unavailable in the G3 readiness slice; mock fallback remains the safe path.",
        ),
        sdk_status=GeminiSdkStatus(
            sdk_available=sdk_result.sdk_available,
            note="SDK availability is modeled through an explicit readiness flag; the package is not imported here.",
        ),
        gates=gates,
        availability=availability,
    )


def check_gemini_sdk_available(
    loader: GeminiSdkLoader | None = None,
    *,
    sdk_available: bool | None = None,
) -> GeminiSdkLoadResult:
    """Return the offline SDK load result without importing Gemini."""

    if sdk_available is not None:
        return GeminiSdkLoadResult(
            checked=True,
            sdk_available=sdk_available,
            status="available" if sdk_available else "missing",
            reason="SDK availability was provided explicitly for the readiness slice.",
            sdk_name="google-genai",
            import_style="from google import genai",
        )
    if loader is None:
        return GeminiSdkLoadResult()
    return loader.load_status()


load_gemini_sdk_status = check_gemini_sdk_available
sdk_loader = GeminiSdkLoader
sdk_load_error = GeminiProviderError


def _resolve_gemini_sdk_load_result(
    *,
    settings: ProviderRuntimeSettings,
    sdk_available: bool | None,
    sdk_loader: GeminiSdkLoader | Callable[[], GeminiSdkLoadResult] | None,
    sdk_load_result: GeminiSdkLoadResult | None,
) -> GeminiSdkLoadResult:
    if not settings.enable_llm or not settings.gemini_key_present:
        return GeminiSdkLoadResult(
            checked=False,
            sdk_available=False,
            status="not_checked",
            reason="SDK check is skipped while Gemini is disabled or the API key is missing.",
        )
    if sdk_load_result is not None:
        return sdk_load_result
    if sdk_loader is not None:
        if hasattr(sdk_loader, "load_status"):
            return sdk_loader.load_status()
        if callable(sdk_loader):
            result = sdk_loader()
            if isinstance(result, GeminiSdkLoadResult):
                return result
            return GeminiSdkLoadResult(
                checked=True,
                sdk_available=False,
                status="load_error",
                reason="SDK loader callable did not return a GeminiSdkLoadResult.",
                error_category="invalid_loader_result",
            )
    return check_gemini_sdk_available(sdk_available=sdk_available if sdk_available is not None else False)


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


def _coerce_gemini_generation_request(
    request: AgentProviderRequest | GeminiGenerationRequest,
    *,
    allowed_evidence_values: Iterable[Any] | None,
    client_name: str,
) -> GeminiGenerationRequest:
    if isinstance(request, GeminiGenerationRequest):
        if allowed_evidence_values is None:
            return request
        merged_allowed_values = tuple(
            dict.fromkeys(request.allowed_evidence_values + tuple(allowed_evidence_values))
        )
        return GeminiGenerationRequest(
            provider_request=request.provider_request,
            allowed_evidence_values=merged_allowed_values,
            client_name=request.client_name or client_name,
        )

    if allowed_evidence_values is None:
        allowed = tuple()
    else:
        allowed = tuple(allowed_evidence_values)

    return GeminiGenerationRequest(
        provider_request=request,
        allowed_evidence_values=allowed,
        client_name=client_name,
    )


def _coerce_gemini_client_result(raw_result: Any) -> GeminiClientResult | None:
    if isinstance(raw_result, GeminiClientResult):
        return raw_result
    if isinstance(raw_result, str):
        return GeminiClientResult(text=raw_result)
    if isinstance(raw_result, dict):
        if not any(key in raw_result for key in ("text", "payload", "error_kind", "error_message")):
            return None
        return GeminiClientResult(
            text=raw_result.get("text"),
            payload=raw_result.get("payload"),
            error_kind=raw_result.get("error_kind"),
            error_message=raw_result.get("error_message"),
        )

    text = getattr(raw_result, "text", None)
    payload = getattr(raw_result, "payload", None)
    error_kind = getattr(raw_result, "error_kind", None)
    error_message = getattr(raw_result, "error_message", None)
    if any(value is not None for value in (text, payload, error_kind, error_message)):
        return GeminiClientResult(
            text=text,
            payload=payload,
            error_kind=error_kind,
            error_message=error_message,
        )
    return None


def _build_skeleton_not_implemented_generation_result(
    request: GeminiGenerationRequest,
) -> GeminiGenerationResult:
    response = build_provider_response(
        answer=(
            "I can’t provide a Gemini answer in this slice. "
            "Mock fallback remains the safe path. Manual review still applies."
        ),
        provider_used="mock",
        fallback_used=True,
        fallback_reason="Gemini provider skeleton is not implemented in this slice; mock fallback remains the safe path.",
        grounding_status=_request_grounding_status(request.provider_request),
        safety_status="pass",
        limitations=_request_limitations(
            request.provider_request,
            [
                "Gemini provider skeleton is not implemented in this slice; mock fallback remains the safe path.",
            ],
        ),
        evidence_used=_request_evidence_items(request.provider_request),
        provider_error="Gemini provider skeleton is not implemented in this slice.",
    )
    return GeminiGenerationResult(
        provider_response=response,
        status="not_implemented",
        safe_to_send=False,
        safe_to_display=True,
        provider_error="Gemini provider skeleton is not implemented in this slice.",
        fallback_reason=response.fallback_reason,
        client_name=request.client_name,
    )


def _build_skeleton_fallback_generation_result(
    request: GeminiGenerationRequest,
    *,
    status: str,
    fallback_reason: str,
    provider_error: str,
) -> GeminiGenerationResult:
    evaluation = _build_mock_fallback_evaluation(
        request=request.provider_request,
        fallback_reason=fallback_reason,
        provider_error=provider_error,
    )
    return GeminiGenerationResult(
        provider_response=evaluation.provider_response,
        status=status,
        safe_to_send=True,
        safe_to_display=evaluation.safe_to_display,
        provider_error=provider_error,
        fallback_reason=fallback_reason,
        client_name=request.client_name,
    )


def _build_synthetic_grounding_context(request: AgentProviderRequest) -> Any:
    """Build a minimal grounding context for safety checks in the real path."""
    from .context_builder import AgentGroundingContext

    grounding_context = request.grounding_context
    evidence_used = grounding_context.get("evidence_used", [])
    limitations = grounding_context.get("limitations", [])
    safety_boundaries = grounding_context.get("safety_boundaries", [])
    forbidden_claims = grounding_context.get("forbidden_claims", [])
    visible_context = grounding_context.get("visible_context", {})
    inspection_response = grounding_context.get("inspection_response", {})

    if not isinstance(visible_context, dict):
        visible_context = {}
    if not isinstance(inspection_response, dict):
        inspection_response = {}
    if not isinstance(evidence_used, list):
        evidence_used = []
    if not isinstance(limitations, list):
        limitations = []
    if not isinstance(safety_boundaries, list):
        safety_boundaries = []
    if not isinstance(forbidden_claims, list):
        forbidden_claims = []

    return AgentGroundingContext(
        page_id=request.page_id,
        section_id=request.section_id,
        component_id=request.component_id,
        question=request.question,
        visible_context=visible_context,
        inspection_response=inspection_response,
        global_context={},
        page_definition={},
        evidence_used=[item for item in evidence_used if isinstance(item, dict)],
        limitations=[str(item) for item in limitations if str(item).strip()],
        safety_boundaries=[str(item) for item in safety_boundaries if str(item).strip()],
        forbidden_claims=[str(item) for item in forbidden_claims if str(item).strip()],
        grounding_status=_request_grounding_status(request),
        raw_evidence_included=bool(grounding_context.get("raw_evidence_included", False)),
    )


def _build_real_gemini_prompt(
    request: AgentProviderRequest,
    *,
    sanitized_context: dict[str, Any],
) -> str:
    evidence_sources = ", ".join(
        str(item.get("source", "unknown"))
        for item in sanitized_context.get("evidence_used", [])
        if isinstance(item, dict)
    )
    limitations = ", ".join(str(item) for item in sanitized_context.get("limitations", []) if str(item).strip())
    page_id = sanitized_context.get("page_id", request.page_id)
    section_id = sanitized_context.get("section_id", request.section_id)
    component_id = sanitized_context.get("component_id", request.component_id or "none")
    question = sanitized_context.get("question", request.question)
    grounding_status = sanitized_context.get("grounding_status", _request_grounding_status(request))
    prompt_parts = [
        "You are the offline-safe Gemini provider execution boundary for the Agent/RAG MVP.",
        f"page_id={page_id}",
        f"section_id={section_id}",
        f"component_id={component_id}",
        f"question={question}",
        f"grounding_status={grounding_status}",
        f"limitations={limitations or 'none'}",
        f"evidence_sources={evidence_sources or 'none'}",
        "Answer only from the compact sanitized context and keep manual review visible.",
    ]
    return "\n".join(prompt_parts)


def _load_google_genai_module() -> Any:
    """Lazy SDK import boundary for the future Gemini execution path."""
    return importlib.import_module("google" + ".genai")


def _build_real_generation_not_implemented_result(
    request: GeminiGenerationRequest,
    *,
    readiness: GeminiG3Readiness,
    provider_error: str,
) -> GeminiRealGenerationResult:
    response = build_provider_response(
        answer=(
            "I can’t provide a Gemini answer in this slice. "
            "Mock fallback remains the safe path. Manual review still applies."
        ),
        provider_used="mock",
        fallback_used=True,
        fallback_reason="Gemini real provider execution is not implemented in this slice; mock fallback remains the safe path.",
        grounding_status=_request_grounding_status(request.provider_request),
        safety_status="pass",
        limitations=_request_limitations(
            request.provider_request,
            [
                "Gemini real provider execution is not implemented in this slice; mock fallback remains the safe path.",
            ],
        ),
        evidence_used=_request_evidence_items(request.provider_request),
        provider_error=provider_error,
    )
    return GeminiRealGenerationResult(
        provider_response=response,
        status="not_implemented",
        safe_to_send=False,
        safe_to_display=True,
        provider_error=provider_error,
        fallback_reason=response.fallback_reason,
        client_name=request.client_name,
        readiness=readiness,
        sdk_load_result=None,
    )


def _build_real_generation_fallback_result(
    request: GeminiGenerationRequest,
    *,
    readiness: GeminiG3Readiness,
    sdk_load_result: GeminiSdkLoadResult | None,
    status: str,
    fallback_reason: str,
    provider_error: str,
    safe_to_display: bool = True,
    blocked: bool = False,
) -> GeminiRealGenerationResult:
    response = build_provider_response(
        answer=(
            "I can’t provide a Gemini answer in this slice. "
            "Mock fallback remains the safe path. Manual review still applies."
        ),
        provider_used="mock",
        fallback_used=True,
        fallback_reason=fallback_reason,
        grounding_status=_request_grounding_status(request.provider_request),
        safety_status="blocked" if blocked else "pass",
        limitations=_request_limitations(request.provider_request, [fallback_reason]),
        evidence_used=_request_evidence_items(request.provider_request),
        provider_error=provider_error,
    )
    return GeminiRealGenerationResult(
        provider_response=response,
        status=status,
        safe_to_send=not blocked,
        safe_to_display=safe_to_display,
        provider_error=provider_error,
        fallback_reason=fallback_reason,
        client_name=request.client_name,
        readiness=readiness,
        sdk_load_result=sdk_load_result,
    )


@dataclass(frozen=True, slots=True)
class _RealProviderExceptionClassification:
    status: str
    fallback_reason: str
    provider_error: str


def _classify_real_provider_exception(exc: Exception) -> _RealProviderExceptionClassification:
    exc_type = type(exc)
    normalized = " ".join(
        part
        for part in (
            exc_type.__module__,
            exc_type.__name__,
            str(exc),
        )
        if part
    ).lower()

    if any(token in normalized for token in ("serviceunavailable", "503", "unavailable")):
        return _RealProviderExceptionClassification(
            status="provider_error",
            fallback_reason="Gemini real provider service unavailable; mock fallback remains the safe path.",
            provider_error="Gemini real provider service unavailable.",
        )
    if any(
        token in normalized
        for token in ("too" + "many" + "req" + "uests", "resourceexhausted", "429", "rate limit")
    ):
        return _RealProviderExceptionClassification(
            status="rate_limit",
            fallback_reason="Gemini real provider was rate limited; mock fallback remains the safe path.",
            provider_error="Gemini real provider was rate limited.",
        )
    if any(token in normalized for token in ("deadlineexceeded", "timeout", "timed out")):
        return _RealProviderExceptionClassification(
            status="timeout",
            fallback_reason="Gemini real provider timed out; mock fallback remains the safe path.",
            provider_error="Gemini real provider timeout.",
        )
    return _RealProviderExceptionClassification(
        status="provider_error",
        fallback_reason="Gemini real provider raised a provider error; mock fallback remains the safe path.",
        provider_error="Gemini real provider raised a provider error.",
    )
