"""Gemini provider stub for Phase G1.

This module is offline-only and intentionally does not implement real Gemini
execution. It exists so the repository can define provider configuration and
stub behavior before any network-enabled Gemini integration is considered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .provider_contracts import (
    AgentProviderRequest,
    AgentProviderResponse,
    AgentProviderStatus,
    ProviderFallbackPolicy,
    ProviderReadinessResult,
    ProviderRuntimeSettings,
)


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


def evaluate_gemini_provider_readiness(
    settings: ProviderRuntimeSettings,
) -> ProviderReadinessResult:
    """Return the G1 Gemini stub readiness snapshot without any network access."""
    return GeminiProviderStub.from_runtime_settings(settings).readiness()
