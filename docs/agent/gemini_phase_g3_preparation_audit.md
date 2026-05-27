# Gemini Phase G3 Preparation Audit

## Executive Summary

Phase G3 is not implemented.
This is a preparation-only audit for the future real Gemini provider integration phase.

Current confirmed state:

- Phase G0 readiness plan is complete.
- Phase G1 provider config/stub is complete.
- Phase G2 mocked-client tests are complete.
- Real Gemini provider integration has not started.
- No Gemini API call is implemented.
- No real LLM execution is active.
- Gemini runtime activation is not active.
- The system remains mock-first.

This audit records the package, environment, API-key, activation, rollback, and testing strategy that should be in place before any first real Gemini call is attempted.
It does not authorize implementation by itself.

## Current Baseline

The current repository baseline already provides:

- A deterministic safety guard wired into the mock provider path.
- Provider request, response, and readiness contracts.
- A mock provider that remains the active runtime path.
- A Gemini G1 stub that remains offline-only.
- A Gemini G2 mocked-client seam that remains offline-only.
- A Gemini G3 package verification artifact that selects `google-genai` as the verified future dependency candidate and requires lazy import only.
- A successful LLM-disabled Docker / Compose smoke validation.
- A pre-Gemini active explainability scope limited to four accepted components.
- A formal pre-Gemini requirement-to-test matrix.
- Passing compile, agent, API, frontend, and docker test suites in the validated baseline.

## Package and Dependency Strategy

Recommended strategy for G3 implementation:

- Keep the Gemini integration dependency out of normal import paths until G3 starts.
- Use `google-genai` as the verified future dependency candidate.
- Use a late-import / optional-import pattern so the mock-first runtime does not require the Gemini package.
- Use `from google import genai` only inside the real provider execution path when G3 implementation is approved.
- Do not add or import the Gemini SDK at module import time in shared Agent code.
- If the SDK is missing and Gemini is disabled, the app must still start and remain mock-first.
- If the SDK is missing and Gemini is enabled, the provider must fail safely and fall back to mock or unavailable status.
- If the SDK is present, the provider must still remain disabled unless explicit G3 activation gates are satisfied.

This audit does not change requirements files.
Any dependency addition should happen only in the real G3 implementation step after approval.

## Environment Variable Strategy

Planned environment variables and defaults:

- `AGENT_ENABLE_LLM=false` by default.
- `AGENT_DEFAULT_PROVIDER=mock` by default.
- `GEMINI_API_KEY` only from environment injection or secret manager.
- `GEMINI_MODEL_NAME` optional, with a safe default chosen later.
- `GEMINI_TIMEOUT_SECONDS` optional.
- `GEMINI_MAX_OUTPUT_TOKENS` optional.
- `GEMINI_TEMPERATURE` optional.
- `AGENT_ENABLE_PROVIDER_FALLBACK=true` optional and default-safe.

Rules:

- If `AGENT_ENABLE_LLM=false`, Gemini must stay disabled even if a key exists.
- If `GEMINI_API_KEY` is missing, Gemini must be unavailable and fallback must remain safe.
- Health endpoints must expose only booleans and status fields, never secret values.
- Keys must never be printed or committed.

## API Key Handling

Local development:

- Use shell environment injection or a local `.env` file only for placeholders.
- Keep real keys out of the repository.
- Fail safely if the key is absent.

Docker / Compose:

- Use environment variables or placeholder-only `.env.example` entries.
- Do not commit real secrets into compose or image build files.
- Keep the LLM-disabled defaults intact.

EC2 / production-like environments:

- Use secure environment injection or a secret manager / parameter store.
- Do not hardcode keys in images or repo files.
- Keep rollback available through `AGENT_ENABLE_LLM=false`.

Logging:

- Never log secret values.
- Never log raw prompts that contain secret-like material.
- Never echo key values in health, readiness, or fallback messages.

## Provider Activation Gates

Before any real Gemini call can happen, all of these gates must be satisfied:

- `AGENT_ENABLE_LLM=true`.
- Gemini is explicitly selected or allowed by the provider policy.
- `GEMINI_API_KEY` is present.
- The Gemini SDK is installed and importable.
- The provider request has passed the pre-generation safety guard.
- The provider request uses sanitized context only.
- Component evidence remains allowlisted and compact.
- Timeout and retry policy is defined.
- Fallback remains enabled and safe.
- The user has explicitly approved the G3 implementation phase.

If any gate fails, the provider must remain disabled or fall back safely to mock.

## Context Boundary for Gemini

Allowed context:

- sanitized question
- `page_id`
- `section_id`
- `component_id`
- compact `visible_context`
- allowlisted compact evidence
- limitations
- traceability references
- grounding status

Forbidden context:

- raw evidence blobs
- full artifact JSON
- local absolute paths
- secrets or environment variables
- full logs
- hidden files
- unrelated files
- unbounded inspection responses

## Safety and Fallback Strategy

Safety must remain in both directions:

- Pre-generation guard sanitizes the provider request before any Gemini call.
- Post-generation guard validates the Gemini output before display.

Fallback rules:

- Timeout, rate-limit, provider error, empty response, malformed response, or unsafe output must fall back safely.
- Blocked output must not be displayed directly.
- Manual review must remain visible.
- Production-ready, deployment-safe, and autonomous-decision claims must remain blocked.
- Invented metrics, thresholds, predictions, and decisions must remain blocked.

## Testing Strategy Before and During G3

The following test coverage should exist before or with G3 implementation:

- no-key tests
- key-present but LLM-disabled tests
- SDK-missing tests
- SDK-available mocked tests
- timeout tests
- generic provider error tests
- rate-limit tests
- malformed response tests
- empty response tests
- unsafe output tests
- invented metric tests
- API shape compatibility tests
- frontend regression tests
- Docker / Compose LLM-disabled regression tests
- optional manual real Gemini smoke only when explicitly approved

## Docker and Compose Strategy

- Keep `AGENT_ENABLE_LLM=false` as the safe default.
- Do not place real API keys in compose files or committed environment files.
- Keep the LLM-disabled smoke path required.
- Any Gemini-enabled smoke must be separate and explicitly approved.
- Compose must remain functional when the SDK is missing and LLM is disabled.

## EC2 Strategy

- Use environment injection or secret manager for keys.
- Do not bake secrets into images.
- Keep mock fallback available.
- Health endpoints should show only status booleans and readiness information.
- Logs should be sanitized.
- Rollback should be possible by setting `AGENT_ENABLE_LLM=false`.

## Observability

Recommended safe telemetry fields:

- `provider_used`
- `fallback_used`
- `fallback_reason`
- `provider_error` category
- `safety_status`
- `grounding_status`
- `component_id`
- `request_id` if available

Do not log raw prompts or secrets by default.

## Rollback and Kill Switch

- `AGENT_ENABLE_LLM=false` disables real provider execution.
- `AGENT_DEFAULT_PROVIDER=mock` keeps the runtime safe by default.
- Mock fallback must remain available.
- The system should degrade safely without requiring a deployment rollback.

## G3 Entry Checklist

Before G3 implementation starts, all of these must be true:

- The user has approved the implementation phase.
- The package verification artifact has been reviewed.
- The package decision has been verified.
- The environment variable names are agreed.
- The dependency strategy is agreed.
- The tests have been planned.
- The rollback path is agreed.
- No secrets have been committed.
- The current baseline tests remain green.

## G3 Implementation Boundaries

When G3 eventually starts, the allowed changes are limited to:

- a real provider implementation module
- an optional dependency update if needed
- provider readiness updates
- mocked and offline tests for the new provider path
- no default frontend behavior changes
- no provider activation by default

## Remaining Decision Points

- Which Gemini model name to adopt
- Exact version pinning for the verified future dependency
- Whether real smoke testing should begin locally before any Docker-enabled smoke
- Whether EC2 Gemini activation is in this project scope

## Decision Summary

G3 is a planning target only.
The safe implementation strategy is:

- optional/lazy SDK import
- environment-gated activation
- strict safety guard before and after provider output
- mock fallback preserved
- no real Gemini call until all gates pass and the user approves G3
