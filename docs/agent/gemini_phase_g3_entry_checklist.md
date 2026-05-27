# Gemini Phase G3 Entry Checklist and First Slice Plan

## Executive Summary

G3 entry is prepared, but real Gemini provider implementation has not started.

This document defines the safest first coding slice for G3. The first slice must be small, reversible, and offline-only. It must not activate Gemini, it must not call the network, and it must keep the normal runtime mock-first.

No SDK is installed in the first readiness-scaffolding slice.
The first readiness-scaffolding slice does not change requirements.
No API call is made in the first readiness-scaffolding slice.
No runtime activation is added in the first readiness-scaffolding slice.

## Current Completed Prerequisites

The following prerequisites are complete:

- Phase G0 readiness plan
- Phase G1 provider config / stub
- Phase G2 mocked Gemini client seam
- Phase G3 preparation audit
- Phase G3 package verification
- Phase G3 execution gate design
- current tests are green
- working tree was clean at the start of the validated baseline

## First Slice Status

The first readiness-scaffolding slice is implemented and tested.

What this means:

- SDK-missing and key-missing behavior is modeled safely.
- Real Gemini provider execution remains inactive.
- No SDK was added.
- No requirements file was changed.
- The normal runtime remains mock-first.

What this does not mean:

- Real Gemini provider integration has started.
- Gemini is connected.
- Gemini is active.
- A Gemini API call is implemented.

## Second Slice Status

The lazy SDK loader boundary is implemented and tested.

What this means:

- SDK availability can be modeled through an injected loader boundary.
- The module still does not import or load the real SDK at import time.
- The normal runtime remains mock-first and offline-only.

What this does not mean:

- Google Gen AI SDK is installed.
- Gemini is active.
- Gemini runtime activation is enabled.
- A real Gemini client has been created.

## Third Slice Status

The health/readiness integration slice is implemented and tested.

What this means:

- Gemini readiness metadata is now threaded into the existing health/readiness surface.
- The health response stays schema-compatible and mock-first.
- SDK presence, key presence, activation gating, and real-provider-implemented state remain safe metadata only.
- No runtime activation is added.

What this does not mean:

- Gemini is connected.
- Gemini is active.
- A Gemini API call is implemented.
- Real Gemini provider execution has started.

## Fourth Slice Status

The provider skeleton with mocked / injected SDK object is implemented and tested.

What this means:

- The future provider boundary can be exercised through injected test clients only.
- The module remains offline-only and does not create a real Gemini client.
- Safe fallback behavior remains in place when no injected client is provided.

What this does not mean:

- Google Gen AI SDK is installed.
- Real Gemini provider execution has started.
- Gemini is connected.
- Gemini is active.

## Fifth Slice Status

The execution gate design for disabled-by-default router activation is implemented and reviewed.

What this means:

- The routing gate strategy is documented before any runtime routing code exists.
- The default runtime remains mock-first and disabled-by-default.
- Gemini cannot become active through dependency installation alone.

What this does not mean:

- Provider routing to Gemini is implemented.
- Gemini is connected.
- Gemini is active.
- A Gemini API call is implemented.

## Pre-Real-Call Audit Status

The pre-real-call audit is implemented and reviewed.

What this means:

- The final activation gates are documented before any real provider work begins.
- The package / requirements decision remains explicit and controlled.
- No runtime activation is added by this audit.

What this does not mean:

- Real Gemini provider implementation has started.
- Gemini is connected.
- Gemini is active.
- A Gemini API call is implemented.

## Dependency Decision Status

The dependency decision artifact is implemented and reviewed.

What this means:

- The repository dependency structure has been inspected.
- `google-genai` is identified as the future package candidate.
- The backend/API requirements slice has been applied in `requirements-api.txt`.
- No other requirements file is changed in this slice.

What this does not mean:

- `google-genai` has been installed.
- Real Gemini provider implementation has started.
- Gemini is connected.
- Gemini is active.

## First Slice Objective

The first G3 coding slice should be:

> Add real-provider readiness scaffolding and SDK-missing/key-missing behavior only, without real SDK import, dependency addition, or API call.

The first slice may cover:

- environment parsing for Gemini-specific config
- SDK-missing behavior modeled without importing the SDK
- key-present / key-missing readiness behavior
- keeping Gemini unavailable unless all gates pass
- tests for disabled / missing dependency behavior
- documentation update

The first slice must not:

- add `google-genai` to requirements
- import `from google import genai`
- call Gemini
- route `/agent/explain` to Gemini
- change frontend behavior
- change Docker / Compose
- require an API key
- enable real LLM execution

## Allowed Files for the First Slice

The first coding slice should stay within these files unless a compatibility test proves a small extension is needed:

- `src/inspection_ai/agent/gemini_provider.py`
- `src/inspection_ai/agent/provider_contracts.py` only if needed for safe readiness metadata
- `src/inspection_ai/agent/provider_router.py` only if readiness metadata needs a safe extension
- `tests/agent/test_gemini_provider_stub.py`
- `tests/agent/test_provider_router.py` if health or readiness wording changes
- `docs/agent/gemini_phase_g3_entry_checklist.md`
- `docs/agent/gemini_provider_integration_readiness_plan.md`
- `docs/agent/pre_gemini_test_matrix.md`

## Forbidden or Discouraged Files for the First Slice

Do not touch these files for the first G3 slice:

- `requirements.txt`
- `requirements-api.txt`
- `requirements-frontend.txt`
- `Dockerfile.api`
- `Dockerfile.frontend`
- `docker-compose.yml`
- `frontend/streamlit_app.py`
- model/runtime assets
- component registry files
- evidence loader files

## Required Tests for the First Slice

The first slice must verify:

- Gemini remains unavailable when `AGENT_ENABLE_LLM=false` even if a key is present.
- Gemini remains unavailable when the key is missing.
- Gemini remains unavailable when the SDK is missing.
- SDK missing does not break the mock-first runtime.
- normal `/agent/explain` still returns `provider_used=mock`.
- no `google`, `openai`, `requests`, `httpx`, or network imports are added to the Gemini module.
- raw key values are never exposed.
- health responses expose only booleans and status.

## Environment Gates for Future G3 Implementation

The future real Gemini implementation may start only if all of these are true:

- `AGENT_ENABLE_LLM=true`
- default provider or requested provider allows Gemini
- `GEMINI_API_KEY` is present
- `google-genai` is installed
- SDK import succeeds lazily
- pre-generation guard passes
- sanitized request only
- fallback remains enabled
- the user has approved the implementation phase

## First Slice Acceptance Criteria

The first slice is accepted only if:

- `python -m compileall frontend tests api src/inspection_ai scripts` passes
- `pytest tests/agent/ -q` passes
- `pytest tests/api/test_agent_endpoint.py -q` passes
- `pytest tests/frontend/ -q` passes
- `pytest tests/docker/ -q` passes
- the working tree shows only the expected documentation or first-slice files before commit
- no SDK import is added
- no requirements file is changed
- no network or Gemini API call is implemented
- the normal runtime remains mock-first

## Rollback Strategy

Rollback should be trivial:

- revert the first-slice commit, or
- set `AGENT_ENABLE_LLM=false`

No deployment rollback should be needed because the first slice does not add runtime activation.

## Next Slice After the First Slice

The next slice after this one is only a placeholder:

- possible future slice: add optional lazy SDK import behind tests
- still no default activation
- real API call only after a separate explicit approval

## Explicit Statement

This is planning only.
G3 real provider implementation has not started.
No Gemini API call is implemented.
No real LLM execution is active.
Gemini runtime activation is not active.
The provider skeleton with mocked / injected SDK object is implemented, but it does not activate Gemini or change runtime behavior.
The health/readiness metadata slice is implemented, but it does not activate Gemini.
The pre-real-call audit is implemented, but it does not activate Gemini or change runtime behavior.
The execution gate design is implemented, but provider routing to Gemini is not implemented and Gemini remains disabled by default.
The dependency decision is documented, and the backend/API requirements slice is applied. Exact version pinning remains a separate review item.
The backend/API requirements slice is applied, but it does not activate Gemini or change runtime behavior.
