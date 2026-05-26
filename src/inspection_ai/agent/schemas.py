"""Schemas for the Agent/RAG MVP contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .context_builder import validate_page_section_component


AgentPageId = Literal[
    "overview",
    "classification",
    "anomaly",
    "detection",
    "image_inspection",
    "safety",
    "ai_assistant",
]
GroundingStatus = Literal["grounded", "partially_grounded", "insufficient_evidence", "unsupported"]
AgentProviderName = Literal["mock", "gemini", "grok"]


class AgentEvidenceItem(BaseModel):
    """One evidence item referenced by an explanation response."""

    source: str
    value: Any


class AgentExplainRequest(BaseModel):
    """Structured input for a page-aware explanation request."""

    model_config = ConfigDict(extra="forbid")

    page_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    component_id: str | None = Field(default=None, min_length=1)
    question: str = Field(min_length=1)
    visible_context: dict[str, Any] = Field(default_factory=dict)
    inspection_response: dict[str, Any] = Field(default_factory=dict)
    include_raw_evidence: bool = False

    @model_validator(mode="after")
    def validate_page_section(self) -> "AgentExplainRequest":
        validate_page_section_component(self.page_id, self.section_id, self.component_id)
        return self


class AgentExplainResponse(BaseModel):
    """Structured output for an evidence-grounded explanation."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    evidence_used: list[AgentEvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provider_used: AgentProviderName
    fallback_used: bool
    grounding_status: GroundingStatus
    page_id: AgentPageId
    section_id: str
    component_id: str | None = None


class AgentHealthResponse(BaseModel):
    """Health signal for the Agent/RAG MVP."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    service: str
    agent_ready: bool
    llm_enabled: bool
    default_provider: AgentProviderName
    provider_order: list[AgentProviderName]
    available_providers: list[AgentProviderName]
    fallback_available: bool
    grounding_ready: bool
    warnings: list[str] = Field(default_factory=list)
