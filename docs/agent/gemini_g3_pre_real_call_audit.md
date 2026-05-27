# Gemini G3 Pre-Real-Call Audit / Final Activation Gate Review

## Executive Summary

This document is documentation only.

Real Gemini provider implementation has not started.
No package is installed.
No requirements file is changed.
No SDK import is added.
No Gemini API call is implemented.
No real LLM execution is active.
Gemini runtime activation is not active.
The system remains mock-first.

This audit exists to define the final activation gates and the required evidence before the project may begin any real Gemini implementation.

What is allowed next only after approval:

- a future real-provider runtime activation slice, if explicitly approved
- a later local real-smoke plan, if explicitly approved
- the local real-smoke plan is documented separately and remains planning only
- the local real-smoke harness skeleton exists and remains disabled by default
- the final real-smoke execution checklist exists and remains documentation only
- the harness dry-run verification evidence exists and remains dry-run evidence only

What remains blocked:

- any real Gemini API call
- any runtime activation by default
- any uncontrolled dependency or SDK import
- any provider routing change that makes Gemini active
- any router change that bypasses the disabled-by-default gate

## Current Completed Foundation

The following items are complete and established in the repository:

- Phase G0 readiness plan
- Phase G1 provider config / stub only, no network
- Phase G2 mocked Gemini client seam
- Phase G3 preparation audit
- Phase G3 package verification
- Phase G3 entry checklist
- G3 first coding slice readiness scaffolding
- G3 second coding slice lazy SDK loader boundary
- G3 third coding slice health / readiness metadata
- G3 provider skeleton with mocked / injected SDK object
- G3 real provider execution boundary with lazy import and injected SDK/client only
- G3 execution gate design / disabled-by-default router plan
- G3 router activation tests for disabled-by-default routing
- G3 dependency decision artifact
- G3 backend/API requirements slice (`google-genai`)
- G3 local real-smoke plan
- G3 local real-smoke harness skeleton
- G3 final real-smoke execution checklist
- G3 harness dry-run verification evidence
- The first safe Gemini code slice is implemented as regression coverage only, and it keeps the disabled-by-default Gemini boundary mock-first.
- The second safe Gemini code slice is implemented as regression coverage only, and it keeps provider routing, `/agent/explain`, and `/agent/health` mock-first and secret-safe even when fake key-like values are present.
- Phase 12 LLM-disabled Docker / Compose readiness audit
- Phase 12 LLM-disabled Docker / Compose validation evidence
- Phase 12 LLM-disabled Docker / Compose validation is PASS and remains local-only
- EC2 LLM-disabled readiness planning is documented for the next deployment step
- EC2 LLM-disabled validation evidence is documented and records the actual EC2 PASS
- deterministic safety guard
- provider contract and readiness layer
- mock-first runtime
- LLM-disabled Docker / Compose validation
- current tests green

## Final Activation Gates Before Real Gemini Implementation

All of the following gates must be true before any real Gemini API call can exist:

- explicit user approval for real provider implementation
- exact package / version decision
- requirements change approved
- lazy import design approved
- `AGENT_ENABLE_LLM=true` required for runtime activation
- `AGENT_DEFAULT_PROVIDER` or provider order explicitly allows Gemini
- `GEMINI_API_KEY` present only through environment or secret manager
- `google-genai` installed in API / backend environment
- SDK import only inside a guarded execution path
- pre-generation safety guard passes
- sanitized provider request only
- context boundary is enforced
- post-generation safety guard passes
- fallback to mock on any error
- provider routing tests exist for the disabled-by-default gate
- health reports safe status only
- no frontend behavior change by default
- rollback path is defined
- test suite is green before and after the slice
- no secrets are committed

## Package and Requirements Gate

- The future package candidate remains `google-genai`.
- The backend/API requirements slice has been applied in `requirements-api.txt`.
- Exact version pinning is still pending.
- Requirements must not be changed silently in any other file.
- Dependency placement remains API/backend only, not frontend.
- Package installation remains a separate approved action; it was not run in this step.
- If the SDK is missing and LLM is disabled, the app must still work.
- If the SDK is missing and LLM is enabled, a fallback-safe unavailable status must be returned.

## Environment and Secret Gate

- `AGENT_ENABLE_LLM=false` remains the default.
- `AGENT_DEFAULT_PROVIDER=mock` remains the default.
- `GEMINI_API_KEY` must never be committed.
- Health may expose only a `key_present` boolean, never the key value.
- Logs must never expose the key value.
- Docker / Compose must not contain a real key.
- EC2 must receive the key later through environment injection or a secret manager.

## Runtime Activation Gate

- `real_provider_implemented` must remain `False` until real provider implementation is explicitly approved.
- `activation_allowed` must remain `False` unless all gates pass.
- `available` must remain `False` until real provider code is implemented and explicitly enabled.
- `available_providers` must not include Gemini unless it is genuinely available.
- `/agent/explain` must remain mock unless explicit provider routing is approved later.

## Context Boundary Gate

Allowed future Gemini input:

- sanitized user question
- `page_id`, `section_id`, `component_id`
- compact visible context
- compact allowlisted evidence
- limitations
- traceability references
- grounding status

Forbidden future Gemini input:

- raw evidence dumps
- full artifact JSON
- local file paths
- secrets / env vars
- logs
- hidden files
- unrelated files
- unbounded inspection response
- training artifacts not needed for the answer

## Safety Gate

- Pre-generation safety guard is required.
- Post-generation safety guard is required.
- Claims of production-ready, deployment-ready, or manual-review-not-needed must be blocked.
- Raw evidence, path, and secret leaks must be blocked.
- Unsupported metric invention must be blocked.
- Blocked output must never be displayed directly.
- Fallback to mock must remain available.
- Manual review must remain visible.

## Test Gate Before Real Implementation

Required tests before or with real implementation:

- no-key test
- key-present but LLM-disabled test
- SDK missing test
- SDK import lazy test
- SDK load error test
- provider timeout test
- provider rate-limit test
- provider malformed / empty response test
- unsafe output test
- invented metric output test
- no secret logging test
- health schema compatibility test
- `/agent/explain` mock-first regression
- frontend regression
- Docker / Compose LLM-disabled regression

## Local Real-Smoke Gate

The local real-smoke plan is documented in `docs/agent/gemini_g3_local_real_smoke_plan.md`.
The local real-smoke harness skeleton is documented in `scripts/agent/run_gemini_local_smoke.py`.
The final real-smoke execution checklist is documented in `docs/agent/gemini_g3_final_real_smoke_execution_checklist.md`.
The harness dry-run verification evidence is documented in `docs/agent/gemini_g3_harness_dry_run_verification_evidence.md`.

If later approved, the first real Gemini smoke must be:

- local-only first
- minimal
- manually triggered
- never part of default tests
- not run in CI by default
- with a sanitized prompt only
- with clear pass / fail evidence
- with immediate rollback by `AGENT_ENABLE_LLM=false`

## Docker / Compose Gate

- LLM-disabled Compose smoke remains required before and after real provider work.
- The default Compose path must remain LLM disabled.
- The real key must not be placed in the Compose file.
- Gemini-enabled Compose smoke is a separate future manual approval.

## EC2 Gate

- No EC2 activation until local real-smoke is proven.
- The key must be injected safely.
- Logs must be sanitized.
- Rollback must be possible by environment variable.
- No public real LLM endpoint before safety review.

## Rollback / Kill Switch

- `AGENT_ENABLE_LLM=false`
- `AGENT_DEFAULT_PROVIDER=mock`
- fallback to mock
- remove the environment key
- revert the commit if needed
- no frontend rollback is needed if frontend behavior is unchanged

## Files Allowed in the First Real Implementation Slice

Likely allowed files:

- `src/inspection_ai/agent/gemini_provider.py`
- `src/inspection_ai/agent/provider_router.py` only if needed
- `tests/agent/test_gemini_g3_readiness.py`
- `tests/agent/test_provider_router.py`
- `tests/api/test_agent_endpoint.py` only if health or schema tests need adjustment
- `requirements-api.txt` only if the approved slice is the dependency installation slice
- `docs/agent/*` relevant planning and audit docs

## Files Forbidden Unless Separately Approved

- frontend files
- Dockerfiles
- `docker-compose.yml`
- component registry files
- evidence loader files
- model artifacts
- runtime assets
- training and evaluation scripts

## Final PASS / FAIL Decision

Current status: `READY_FOR_REAL_PROVIDER_PLANNING`

Decision:

- PASS for planning controls and readiness evidence.
- FAIL for activation readiness.

The repository is not ready for uncontrolled real provider activation.

Real provider implementation may start only after explicit user approval for the next non-code Gemini planning step.

Real API call remains blocked until the next approved slice.
The remaining-gap audit found no useful additional no-network Gemini code slice, so the next step is non-code approval and local real-smoke planning.

## Next Recommended Slice

Recommended next slice: **B. Non-code approval and local real-smoke planning**

Why this is the safest next move:

- it keeps the system mock-first while the no-network Gemini code gap is already exhausted
- it preserves the disabled-by-default boundary and keeps real smoke blocked pending explicit approval
- it avoids jumping directly to a real API call

## Explicit Statement

This audit is documentation only.
Real Gemini provider implementation has not started.
The backend/API `google-genai` requirements slice has been applied, but no package has been installed.
No SDK import is added.
No Gemini API call is implemented.
No real LLM execution is active.
Gemini runtime activation is not active.
The system remains mock-first.
The remaining-gap audit found no useful additional no-network Gemini code slice, so the next step is non-code approval and local real-smoke planning.
