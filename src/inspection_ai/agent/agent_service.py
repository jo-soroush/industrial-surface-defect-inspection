"""Service layer for the Agent/RAG MVP."""

from __future__ import annotations

from .context_builder import build_grounding_context
from .provider_router import AgentProviderRouter
from .schemas import AgentExplainRequest, AgentExplainResponse, AgentHealthResponse


class AgentService:
    """Coordinate request validation, grounding, and provider selection."""

    def __init__(self, router: AgentProviderRouter | None = None) -> None:
        self.router = router or AgentProviderRouter()

    def health(self) -> AgentHealthResponse:
        """Return a safe agent health snapshot."""
        return self.router.health()

    def explain(self, request: AgentExplainRequest) -> AgentExplainResponse:
        """Return a grounded explanation for a page/section request."""
        grounding_context = build_grounding_context(
            page_id=request.page_id,
            section_id=request.section_id,
            component_id=request.component_id,
            question=request.question,
            visible_context=request.visible_context,
            inspection_response=request.inspection_response,
            include_raw_evidence=request.include_raw_evidence,
        )
        return self.router.explain(grounding_context)
