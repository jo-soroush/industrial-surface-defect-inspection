"""Routes for the Agent/RAG MVP."""

from __future__ import annotations

from fastapi import APIRouter

from api.app.schemas.agent import AgentExplainRequest, AgentExplainResponse, AgentHealthResponse
from src.inspection_ai.agent.agent_service import AgentService


router = APIRouter(prefix="/agent", tags=["agent"])


def _service() -> AgentService:
    """Build the agent service at request time so env-driven settings stay current."""
    return AgentService()


@router.get("/health", response_model=AgentHealthResponse)
def agent_health() -> AgentHealthResponse:
    """Return a safe readiness signal for the agent layer."""
    return _service().health()


@router.post("/explain", response_model=AgentExplainResponse)
def agent_explain(request: AgentExplainRequest) -> AgentExplainResponse:
    """Return an evidence-grounded explanation for the requested page/section."""
    return _service().explain(request)
