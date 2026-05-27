"""Deterministic safety guard for the Agent/RAG MVP.

This module stays offline and provider-agnostic. It validates the context that
would be sent to a future LLM and the text that would come back from a future
LLM or the current mock provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Iterable, Literal

from .context_builder import AgentGroundingContext


SafetyGuardStatus = Literal["pass", "limited", "blocked"]


SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z_-]{10,}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]+\b", re.IGNORECASE),
    re.compile(r"\bgh[pousr]_[0-9A-Za-z]{20,}\b", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|secret|token|password|passwd|private[_-]?key)\b", re.IGNORECASE),
)

ABSOLUTE_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<!\w)/(?:Users|home|private|var|tmp)/[^\s'\"`<>]+"),
    re.compile(r"\b[A-Za-z]:\\\\[^\s'\"`<>]+"),
)

PRODUCTION_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bproduction[- ]ready\b", re.IGNORECASE),
    re.compile(r"\bdeployment[- ]safe\b", re.IGNORECASE),
    re.compile(r"\bcertify\b.*\b(readiness|safe|deploy|production)\b", re.IGNORECASE),
    re.compile(r"\bautonomous\b.*\b(decision|decisions|final)\b", re.IGNORECASE),
    re.compile(r"\b(?:replace|replaces|replacing)\b.*\b(?:manual|human|expert) review\b", re.IGNORECASE),
    re.compile(r"\bmanual review\b.*\b(?:not required|not needed|unnecessary)\b", re.IGNORECASE),
    re.compile(r"\b(?:human|expert|manual) review\b.*\b(?:not required|not needed|unnecessary)\b", re.IGNORECASE),
    re.compile(r"\b(?:go|ship|move)\s+to\s+production\b", re.IGNORECASE),
    re.compile(r"\bready\s+for\s+production\b", re.IGNORECASE),
    re.compile(r"\bready\s+for\s+deployment\b", re.IGNORECASE),
)

POSITIVE_PROVIDER_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgemini\b.*\b(?:connected|active|enabled|integrated|available|running)\b", re.IGNORECASE),
    re.compile(r"\bgrok\b.*\b(?:connected|active|enabled|integrated|available|running)\b", re.IGNORECASE),
    re.compile(r"\bopenai\b.*\b(?:connected|active|enabled|integrated|available|running)\b", re.IGNORECASE),
    re.compile(r"\breal llm\b.*\b(?:connected|active|enabled|integrated|available|running)\b", re.IGNORECASE),
    re.compile(r"\bexternal llm\b.*\b(?:connected|active|enabled|integrated|available|running)\b", re.IGNORECASE),
)

METRIC_LANGUAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bthreshold\b", re.IGNORECASE),
    re.compile(r"\baccuracy\b", re.IGNORECASE),
    re.compile(r"\bprecision\b", re.IGNORECASE),
    re.compile(r"\brecall\b", re.IGNORECASE),
    re.compile(r"\bf1\b", re.IGNORECASE),
    re.compile(r"\bf-?score\b", re.IGNORECASE),
    re.compile(r"\bscore\b", re.IGNORECASE),
    re.compile(r"\bconfidence\b", re.IGNORECASE),
    re.compile(r"\bprobability\b", re.IGNORECASE),
    re.compile(r"\bmetric\b", re.IGNORECASE),
    re.compile(r"\bauc\b", re.IGNORECASE),
    re.compile(r"\bcount\b", re.IGNORECASE),
    re.compile(r"\bbox(?:es)?\b", re.IGNORECASE),
    re.compile(r"\bdecision\b", re.IGNORECASE),
)

SAFE_NEGATED_REVIEW_PHRASES: tuple[str, ...] = (
    "does not replace manual review",
    "does not replace human review",
    "does not replace expert review",
    "does not replace review",
    "does not replace reviewer approval",
)

SAFE_NEGATED_PROVIDER_PHRASES: tuple[str, ...] = (
    "gemini is not active",
    "gemini is not connected",
    "gemini is not enabled",
    "gemini is not integrated",
    "gemini is not available",
    "grok is not active",
    "grok is not connected",
    "grok is not enabled",
    "grok is not integrated",
    "grok is not available",
    "openai is not active",
    "openai is not connected",
    "openai is not enabled",
    "openai is not integrated",
    "openai is not available",
    "real llm is not active",
    "real llm is not connected",
    "real llm is not enabled",
    "real llm is not integrated",
    "real llm is not available",
    "external llm is not active",
    "external llm is not connected",
    "external llm is not enabled",
    "external llm is not integrated",
    "external llm is not available",
    "external llm providers are not active",
    "external llm providers are not connected",
    "external llm providers are not enabled",
    "external llm providers are not integrated",
    "external llm providers are not available",
    "external llm not connected",
    "external llm not active",
    "external llm not enabled",
    "external llm not integrated",
    "external llm not available",
    "gemini/grok/openai are not active",
    "gemini/grok/openai are not connected",
    "gemini/grok/openai are not enabled",
    "gemini/grok/openai are not integrated",
)

UNSAFE_READINESS_PHRASES: tuple[str, ...] = (
    "production-ready",
    "production ready",
    "deployment-safe",
    "deployment safe",
    "safe for deployment",
    "ready for production",
    "ready for deployment",
    "production readiness",
    "autonomous decision",
    "autonomous decisions",
    "autonomous final decision",
    "replace human review",
    "replace manual review",
    "replace expert review",
    "replaces human review",
    "replaces manual review",
    "replaces expert review",
    "human review is not required",
    "manual review is not required",
    "expert review is not required",
    "human review not required",
    "manual review not required",
    "expert review not required",
    "human review is not needed",
    "manual review is not needed",
    "expert review is not needed",
    "human review not needed",
    "manual review not needed",
    "expert review not needed",
    "no human review needed",
    "no manual review needed",
    "no expert review needed",
)

NUMERIC_TOKEN_PATTERN = re.compile(r"(?<!\w)-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")


@dataclass(frozen=True, slots=True)
class SafetyGuardResult:
    """Structured outcome for pre- and post-generation safety checks."""

    status: SafetyGuardStatus
    blocked: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    sanitized_context: dict[str, Any] = field(default_factory=dict)
    sanitized_text: str | None = None
    safe_to_send: bool = True
    safe_to_display: bool = True


def guard_pre_generation_context(grounding_context: AgentGroundingContext) -> SafetyGuardResult:
    """Validate and sanitize the context before any future LLM call."""
    sanitized_context = _build_sanitized_context(grounding_context)
    reasons: list[str] = []
    warnings: list[str] = []
    limitations = list(dict.fromkeys(grounding_context.limitations))

    if _contains_raw_evidence(grounding_context):
        reasons.append("Raw evidence is not allowed in future LLM prompts.")
        sanitized_context = _drop_raw_evidence(sanitized_context)

    prompt_surface = {
        "question": grounding_context.question,
        "visible_context": grounding_context.visible_context,
        "inspection_response": grounding_context.inspection_response,
        "evidence_used": sanitized_context.get("evidence_used", []),
    }

    secret_hits = _collect_matches(prompt_surface, _contains_secret_like_text)
    if secret_hits:
        warnings.append("Secret-like values were redacted from the prompt context.")
        sanitized_context = _sanitize_recursive(sanitized_context)

    path_hits = _collect_matches(prompt_surface, _contains_absolute_path_text)
    if path_hits:
        warnings.append("Local absolute paths were redacted from the prompt context.")
        sanitized_context = _sanitize_recursive(sanitized_context)

    if _contains_production_or_autonomy_request(prompt_surface):
        reasons.append(
            "The request asks for production/deployment/autonomous certification, which is not allowed."
        )

    status: SafetyGuardStatus
    if reasons:
        status = "blocked"
    elif warnings:
        status = "limited"
    else:
        status = "pass"

    return SafetyGuardResult(
        status=status,
        blocked=status == "blocked",
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        limitations=tuple(dict.fromkeys(limitations)),
        sanitized_context=sanitized_context,
        sanitized_text=None,
        safe_to_send=status != "blocked",
        safe_to_display=status != "blocked",
    )


def guard_post_generation_text(
    answer_text: str,
    *,
    grounding_context: AgentGroundingContext | None = None,
    allowed_evidence_values: Iterable[Any] | None = None,
) -> SafetyGuardResult:
    """Validate generated text before it is returned to the user."""
    reasons: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []
    normalized_answer = _normalize_text(answer_text)

    if _contains_secret_like_text(answer_text):
        reasons.append("Generated text exposes a secret-like value.")

    if _contains_absolute_path_text(answer_text):
        reasons.append("Generated text exposes a local absolute path.")

    if _contains_unallowed_readiness_claim(normalized_answer):
        reasons.append(
            "Generated text claims production readiness, deployment safety, or autonomous decision-making."
        )

    if _contains_provider_connected_claim(normalized_answer):
        reasons.append(
            "Generated text claims Gemini, Grok, OpenAI, or a real LLM is connected or active."
        )

    allowed_tokens = _collect_allowed_numeric_tokens(
        grounding_context=grounding_context,
        allowed_evidence_values=allowed_evidence_values,
    )
    invented_numeric_tokens = _invented_numeric_tokens(normalized_answer, allowed_tokens)
    if invented_numeric_tokens:
        reasons.append(
            "Generated text contains metric-like values that are not present in the allowed evidence."
        )

    if not reasons:
        if "manual review" in normalized_answer and (
            "still applies" in normalized_answer
            or "required" in normalized_answer
            or "remains required" in normalized_answer
        ):
            warnings.append("Manual review boundary was preserved.")
        if "not production-ready" in normalized_answer or "not deployment-safe" in normalized_answer:
            warnings.append("Explicit readiness disclaimers were preserved.")

    status: SafetyGuardStatus
    if reasons:
        status = "blocked"
    elif warnings:
        status = "limited"
    else:
        status = "pass"

    if grounding_context is not None:
        limitations.extend(grounding_context.limitations)

    return SafetyGuardResult(
        status=status,
        blocked=status == "blocked",
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        limitations=tuple(dict.fromkeys(limitations)),
        sanitized_context={},
        sanitized_text=answer_text if status != "blocked" else _blocked_answer_text(reasons),
        safe_to_send=status != "blocked",
        safe_to_display=status != "blocked",
    )


def _build_sanitized_context(grounding_context: AgentGroundingContext) -> dict[str, Any]:
    return {
        "page_id": grounding_context.page_id,
        "section_id": grounding_context.section_id,
        "component_id": grounding_context.component_id,
        "question": _sanitize_recursive(grounding_context.question),
        "visible_context": _sanitize_recursive(grounding_context.visible_context),
        "inspection_response": _sanitize_recursive(grounding_context.inspection_response),
        "evidence_used": _sanitize_evidence_items(grounding_context.evidence_used),
        "limitations": list(dict.fromkeys(grounding_context.limitations)),
        "safety_boundaries": list(grounding_context.safety_boundaries),
        "forbidden_claims": list(grounding_context.forbidden_claims),
        "grounding_status": grounding_context.grounding_status,
        "raw_evidence_included": grounding_context.raw_evidence_included,
    }


def _sanitize_recursive(value: Any) -> Any:
    if isinstance(value, str):
        if _contains_secret_like_text(value):
            return "[REDACTED_SECRET]"
        if _contains_absolute_path_text(value):
            return "[REDACTED_PATH]"
        return value
    if isinstance(value, dict):
        return {key: _sanitize_recursive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_recursive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_recursive(item) for item in value)
    return value


def _sanitize_evidence_items(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized_items: list[dict[str, Any]] = []
    for item in evidence_items:
        source = str(item.get("source", ""))
        if "inspection_response.raw" in source or ".raw" in source or source.endswith("raw"):
            continue
        sanitized_items.append(
            {
                "source": source,
                "value": _sanitize_recursive(item.get("value")),
            }
        )
    return sanitized_items


def _drop_raw_evidence(sanitized_context: dict[str, Any]) -> dict[str, Any]:
    context_copy = dict(sanitized_context)
    evidence_used = context_copy.get("evidence_used", [])
    if isinstance(evidence_used, list):
        context_copy["evidence_used"] = [
            item
            for item in evidence_used
            if "inspection_response.raw" not in str(item.get("source", ""))
            and ".raw" not in str(item.get("source", ""))
            and not str(item.get("source", "")).endswith("raw")
        ]
    return context_copy


def _collect_matches(value: Any, predicate: Any) -> list[str]:
    matches: list[str] = []
    for text in _iter_text_values(value):
        if predicate(text):
            matches.append(text)
    return matches


def _iter_text_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        values.append(value)
    elif isinstance(value, (int, float, Decimal)):
        values.append(str(value))
    elif isinstance(value, dict):
        for item in value.values():
            values.extend(_iter_text_values(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            values.extend(_iter_text_values(item))
    return values


def _contains_secret_like_text(value: Any) -> bool:
    text = str(value)
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _contains_absolute_path_text(value: Any) -> bool:
    text = str(value)
    return any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS)


def _contains_raw_evidence(grounding_context: AgentGroundingContext) -> bool:
    if grounding_context.raw_evidence_included:
        return True
    return any(
        "inspection_response.raw" in str(item.get("source", "")) or ".raw" in str(item.get("source", ""))
        for item in grounding_context.evidence_used
    )


def _contains_production_or_autonomy_request(value: Any) -> bool:
    text = _normalize_text(" ".join(_iter_text_values(value)))
    return any(pattern.search(text) for pattern in PRODUCTION_REQUEST_PATTERNS)


def _contains_unallowed_readiness_claim(text: str) -> bool:
    masked_text = _mask_allowed_readiness_negations(text)
    return any(phrase in masked_text for phrase in UNSAFE_READINESS_PHRASES)


def _contains_provider_connected_claim(text: str) -> bool:
    masked_text = _mask_allowed_provider_negations(text)
    return any(pattern.search(masked_text) for pattern in POSITIVE_PROVIDER_CLAIM_PATTERNS)


def _collect_allowed_numeric_tokens(
    *,
    grounding_context: AgentGroundingContext | None,
    allowed_evidence_values: Iterable[Any] | None,
) -> set[str]:
    tokens: set[str] = set()
    if grounding_context is not None:
        for item in grounding_context.evidence_used:
            tokens.update(_extract_numeric_tokens(item.get("value")))
        tokens.update(_extract_numeric_tokens(grounding_context.visible_context))
        tokens.update(_extract_numeric_tokens(grounding_context.inspection_response))
        tokens.update(_extract_numeric_tokens(grounding_context.question))
    if allowed_evidence_values is not None:
        for value in allowed_evidence_values:
            tokens.update(_extract_numeric_tokens(value))
    return tokens


def _invented_numeric_tokens(answer_text: str, allowed_tokens: set[str]) -> list[str]:
    if not _contains_metric_language(answer_text):
        return []
    answer_tokens = [_normalize_numeric_token(token) for token in NUMERIC_TOKEN_PATTERN.findall(answer_text)]
    invented_tokens = [token for token in answer_tokens if token and token not in allowed_tokens]
    return list(dict.fromkeys(invented_tokens))


def _contains_metric_language(text: str) -> bool:
    return any(pattern.search(text) for pattern in METRIC_LANGUAGE_PATTERNS)


def _extract_numeric_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for text in _iter_text_values(value):
        for token in NUMERIC_TOKEN_PATTERN.findall(text):
            normalized = _normalize_numeric_token(token)
            if normalized:
                tokens.add(normalized)
    if isinstance(value, (int, float, Decimal)):
        tokens.add(_normalize_numeric_token(str(value)))
    return tokens


def _normalize_numeric_token(token: str) -> str:
    text = token.strip().replace(",", "")
    if not text:
        return ""
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    try:
        decimal_value = Decimal(text)
    except (InvalidOperation, ValueError):
        return token.lower()
    normalized = format(decimal_value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return f"{normalized}%" if percent else normalized


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _mask_allowed_readiness_negations(text: str) -> str:
    masked = _normalize_text(text)
    for phrase in ("not production-ready", "not deployment-safe", *SAFE_NEGATED_REVIEW_PHRASES):
        masked = masked.replace(phrase, " " * len(phrase))
    return masked


def _mask_allowed_provider_negations(text: str) -> str:
    masked = _normalize_text(text)
    for phrase in SAFE_NEGATED_PROVIDER_PHRASES:
        masked = masked.replace(phrase, " " * len(phrase))
    return masked


def _blocked_answer_text(reasons: list[str]) -> str:
    if reasons:
        return (
            "I can’t provide that answer because the requested text is unsafe under the Agent safety guard. "
            "Manual review still applies."
        )
    return "I can’t provide that answer because it is unsafe under the Agent safety guard."
