# Gemini G3 Package Verification

## Executive Summary

This document records the verified package decision for the future Gemini provider integration.

The selected future dependency candidate is `google-genai`, based on the current official Google Gemini SDK documentation and the documented migration guidance away from the legacy package.

This is a documentation-only verification artifact.
It does not add a dependency, it does not add a runtime import, and it does not implement a real Gemini call.

## Verified Package Decision

- Selected future dependency candidate: `google-genai`
- Status: verified for planning
- Exact version pinning: to be decided during the implementation phase
- Placement recommendation: add only to the API/backend dependency set when G3 implementation starts, not to the frontend dependency set

Why this decision is recorded now:

- The repository already has a mock-first Agent runtime.
- Gemini remains inactive.
- The future real provider implementation should use the official Google Gen AI SDK path rather than the legacy package.

## Import-Path Decision

The real provider execution path should use a lazy import pattern:

- import inside the Gemini provider execution path only
- use `from google import genai` when the real provider is actually invoked
- do not import `google.genai` or any Gemini SDK at module import time in shared Agent code
- do not make normal mock-first startup depend on the Gemini SDK

The import path is a future implementation detail, not a runtime change in this step.

## Why `google-generativeai` Is Not Selected

`google-generativeai` is not the selected future dependency for this project because the verified official documentation points to `google-genai` as the current Google Gen AI SDK path.

The legacy package should not be introduced for new work in this repository unless a future implementation review explicitly revises the decision.

## Lazy Import Requirement

The Gemini SDK must be imported lazily and guarded:

- normal mock-first runtime must not require `google-genai`
- when Gemini is disabled, the application must still start normally even if the SDK is absent
- when Gemini is enabled but the SDK is absent, the provider must fail safely and fall back to mock or unavailable status
- module import time must stay SDK-free in shared Agent code

## Dependency Placement Recommendation

Recommended placement for the future dependency:

- API/backend requirements only
- not frontend requirements
- not the shared mock-first runtime import path

This keeps the frontend light and avoids coupling normal mock behavior to a provider SDK that is not active by default.

## Runtime Behavior Requirement If Dependency Is Missing

If `google-genai` is missing:

- mock-first runtime must continue to work
- Gemini must remain unavailable/disabled
- health endpoints must report booleans and status only, never secret values
- fallback to mock must remain safe when enabled

Missing dependency must never cause the default mock-first path to become unusable.

## Docker and Compose Implication

- LLM-disabled Docker and Compose defaults must remain intact
- no real Gemini key should be required for the default smoke path
- the current mock-first Docker / Compose smoke behavior must remain passable without the Gemini SDK installed
- any future Gemini-enabled Docker / Compose path must be opt-in and explicitly approved

## EC2 Implication

- keys must be injected through environment or secret manager later
- no key values should be committed to images or repository files
- health endpoints must not expose secret values
- rollback must remain possible by setting `AGENT_ENABLE_LLM=false`

## Required Tests Before the Dependency Is Added

Before any future dependency addition, the following test expectations should exist or remain green:

- `AGENT_ENABLE_LLM=false` keeps Gemini disabled even if a key is present
- missing SDK with LLM disabled remains safe
- missing SDK with LLM enabled falls back safely
- SDK import remains lazy and does not happen at module import time
- provider contract compatibility tests remain green
- safety guard pre/post tests remain green
- API response compatibility tests remain green
- frontend regression tests remain green
- Docker / Compose LLM-disabled smoke remains green

## Remaining Decision Before Real G3 Implementation

The remaining decision is no longer the package family itself.

Remaining implementation decisions are:

- exact version pinning
- model name selection
- whether to keep the first real provider test local-only before any future Docker-enabled Gemini smoke
- whether EC2 Gemini activation is in project scope

## Explicit Statement

This step does not add a real Gemini implementation.
No Gemini API call is implemented.
No real LLM execution is active.
Gemini runtime activation is not active.
