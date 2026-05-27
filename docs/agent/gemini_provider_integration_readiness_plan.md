# Gemini Provider Integration Readiness Plan

## Executive Summary

This document describes how Gemini should later be integrated into the Agent/RAG foundation without breaking the current mock-first, safety-guarded, evidence-grounded system.

This is a planning document only.
Gemini is not currently connected.
No real LLM call is currently made.

The current foundation is ready for planning, not implementation.

## Current Baseline

The current repository baseline already provides:

- A deterministic safety guard, wired into the mock provider path.
- Provider request / response / readiness contracts.
- A mock provider that remains the active runtime path.
- A successful LLM-disabled Docker / Compose smoke validation.
- A pre-Gemini active explainability scope that is intentionally limited to four accepted components.
- A formal pre-Gemini requirement-to-test matrix.

## Non-Negotiable Integration Rules

The following rules must hold for any future Gemini integration:

- No raw evidence is sent to Gemini unless explicitly approved for a specific scope.
- No secrets or local paths may appear in provider context.
- The safety guard must run before and after Gemini.
- Manual review must remain visible.
- Gemini must not claim production-ready or deployment-safe status.
- Gemini must not replace expert or manual review.
- Gemini must not invent metrics, thresholds, predictions, or decisions.
- Fallback to mock must remain available.
- Provider output must not expose raw provider response by default.
- Logs must not contain secrets or raw prompts with sensitive values.

## Environment and Secret Handling

Gemini integration must follow these rules:

- `GEMINI_API_KEY` must be read only from environment injection or a secret manager.
- API keys must never be committed.
- API keys must never be printed.
- Docker / Compose should use environment variables or `.env.example` only, without real values.
- EC2 or other deployment environments should use secure secret injection later.
- Local development must fail safely if the key is absent.
- If `AGENT_ENABLE_LLM=false`, Gemini must stay disabled even if a key exists.

## Provider Order and Fallback Policy

The intended provider behavior is:

- The default provider remains `mock` unless explicitly changed.
- If Gemini is requested but disabled or unavailable, fallback must remain safe and mock-first.
- If Gemini errors, rate-limits, times out, or is blocked by the safety guard, fallback behavior must remain safe.
- Fallback reasons must be recorded internally.
- The API response shape must remain compatible.

## Context and Evidence Sent to Gemini

Allowed context for future Gemini requests:

- `page_id`
- `section_id`
- `component_id`
- sanitized question
- compact `visible_context`
- allowlisted compact evidence only
- limitations
- traceability references
- grounding status

Forbidden context:

- raw evidence blobs
- full artifact JSON
- local absolute paths
- environment variables
- API keys
- full logs
- hidden files
- unrelated project files
- unbounded inspection responses

## Safety Guard Placement

The safety guard must remain in both directions:

- Pre-generation: sanitize the provider request before any Gemini call.
- Post-generation: validate Gemini output before it is returned.

If the guard blocks output, that output must not be displayed directly.
A safe refusal or fallback path must be used instead.
Mixed unsafe claims must remain blocked.

## Docker / Compose Readiness

The Docker / Compose defaults must remain mock-first:

- `AGENT_ENABLE_LLM=false` remains the safe default.
- Enabling Gemini must require explicit environment configuration.
- The LLM-disabled Docker / Compose smoke must continue to pass.
- Any future Gemini-enabled smoke must be separate and must not rely on committed secrets.

## EC2 Readiness

For later EC2 deployment:

- Keys must not be hardcoded.
- Environment variables, secure parameter stores, or secret managers should be used.
- Mock fallback must remain available.
- Health endpoints must expose only booleans and status fields, never key values.
- Logs must be sanitized.
- Rollback must be possible by setting `AGENT_ENABLE_LLM=false`.

## Testing Plan Before Implementation

Before any Gemini implementation begins, the following tests must exist or be updated:

- Provider readiness tests with key absent.
- Provider readiness tests with key present but LLM disabled.
- Provider readiness tests with LLM enabled but provider disabled by config.
- Gemini provider unit tests with a mocked client only.
- No-network tests in the normal unit suite.
- Safety guard pre-generation and post-generation tests for the Gemini path.
- Fallback tests for timeout, error, and rate-limit conditions.
- API response compatibility tests.
- Frontend no-regression tests.
- Docker / Compose LLM-disabled regression test.
- Optional separate manual Gemini smoke test only when a key is explicitly available and approved.

## Rollout Phases

Future Gemini work should follow these phases:

- Phase G0 - readiness plan only: complete
- Phase G1 - provider config / stub only, no network: complete
- Phase G2 - mocked Gemini client tests: complete
- Phase G3 - real Gemini provider behind `AGENT_ENABLE_LLM` and key
- Phase G4 - local manual Gemini smoke
- Phase G5 - Docker / Compose Gemini-enabled optional smoke
- Phase G6 - EC2 secret-managed deployment plan
- Phase G7 - monitoring, limits, rollback

## G1 Completion Note

Phase G1 stub/config implementation is complete.

What this means:

- The Gemini stub exists.
- The stub refuses real execution.
- No Gemini SDK is imported.
- No Gemini API call is implemented.
- No real LLM execution is active.
- Gemini runtime activation is not active.
- No network call is made.
- The runtime remains mock-first.

What this does not mean:

- Real Gemini provider integration has not started.
- Gemini is not connected.
- Gemini is not active.

## G2 Completion Note

Phase G2 mocked-client test layer is complete.

What this means:

- The mocked Gemini client seam exists.
- The mocked client seam remains offline-only.
- No Gemini SDK is imported.
- No Gemini API call is implemented.
- No real LLM execution is active.
- Gemini runtime activation is not active.
- No network call is made.
- The runtime remains mock-first.

What this does not mean:

- Real Gemini provider integration has not started.
- Gemini is not connected.
- Gemini is not active.

## Failure Behavior

The Gemini path must fail safely for these cases:

- no key
- invalid key
- network error
- timeout
- rate limit
- provider unsafe output
- invented metric
- safety guard block
- malformed response
- empty output

## Observability and Logging

Future Gemini integration should log only safe metadata:

- `provider_used`
- `fallback_used`
- `fallback_reason`
- `safety_status`
- `grounding_status`
- `component_id`
- `request_id` if available

It must not log raw prompts or secrets by default.

## Acceptance Criteria Before Gemini Implementation Can Start

Gemini implementation must not start until:

- This plan exists in the repository and is reviewed.
- Current tests pass.
- The LLM-disabled Docker / Compose smoke evidence remains PASS.
- Safety guard tests remain PASS.
- Provider contract tests remain PASS.
- The user explicitly approves the next implementation phase.

## Remaining Blockers After This Plan

- User approval to start the Gemini implementation phase.
- Any implementation-specific package or dependency decision.
- Optional: Gemini API key availability for later runtime testing, but not required for planning.

## Explicit Statement

Phase G1 stub/config implementation is complete.
Real Gemini provider integration has not started.
No Gemini API call is implemented.
No real LLM execution is active.
Gemini runtime activation is not active.
Gemini, Grok, and OpenAI remain inactive.
The system remains mock-first and offline by default.
