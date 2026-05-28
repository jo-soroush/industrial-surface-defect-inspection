# Gemini G3 Local Real-Smoke Plan Only

## Executive Summary

This document is planning only.

No real Gemini API call is run by this document.
No `GEMINI_API_KEY` is used by this document.
No runtime activation is performed.
No provider routing activation is performed.
Normal `/agent/explain` remains mock-first.
This document does not approve real smoke execution.
A future real smoke requires explicit user approval.
Real Gemini, Grok, and OpenAI runtime remain not active.
Production readiness, HTTPS/domain readiness, and real LLM readiness are not claimed.
The local manual Gemini smoke milestone is paused / closed for now at the harness level.
The latest approved Gemini 2.5 Flash smoke reached `SUCCESS_LIMITED`, used `provider_used=gemini`, used no fallback, and kept `grounding_status=grounded` with `safety_status=limited`.
The validated smoke model is now `gemini-2.5-flash`, and the provider default is aligned to that same model.

This plan defines the narrowest local-only evidence path for a future manual smoke while keeping the current mock-first runtime untouched.

A disabled-by-default harness skeleton exists in `scripts/agent/run_gemini_local_smoke.py`; it is not executed by default and it does not call Gemini.

The final execution checklist is documented in `docs/agent/gemini_g3_final_real_smoke_execution_checklist.md`.
Dry-run verification evidence is documented in `docs/agent/gemini_g3_harness_dry_run_verification_evidence.md`.
Approved-but-not-executed discovery is documented in `docs/agent/gemini_g3_approved_real_smoke_attempt_discovery.md`.
The fourth approved local-only manual attempt result is documented in `docs/agent/gemini_g3_local_real_smoke_attempt_4_evidence.md`.
The ninth approved local-only manual attempt result is documented in `docs/agent/gemini_g3_local_real_smoke_attempt_9_evidence.md`.
The latest limited-success evidence is documented in `docs/agent/gemini_g3_local_real_smoke_success_limited_evidence.md`.

What this plan controls:

- the preconditions for a future local manual smoke
- the safe environment handling for a future smoke
- the minimal smoke input boundary
- the pass / fail criteria for a future smoke
- the rollback / kill switch path after a future smoke
- the evidence that must be recorded if a future smoke is ever approved and run
- the approval gate wording that must be satisfied before any execution is allowed

What remains blocked:

- any real Gemini API call
- any runtime activation
- any provider routing activation
- any Docker / Compose / EC2 based smoke
- any CI or default-test smoke
- any public or production-facing endpoint exposure
- any smoke execution without explicit user approval
- any immediate retry while external Gemini availability remains unresolved

## Harness Skeleton Status

The local smoke harness skeleton is documented and reviewed.

What this means:

- the script exists
- the default behavior is dry-run only
- the script does not read `GEMINI_API_KEY` on import
- the script does not call Gemini
- the script is disabled by default

What this does not mean:

- real smoke execution is complete
- Gemini is connected
- Gemini is active
- temporary local-shell `GEMINI_API_KEY` usage means Gemini is active by default
- limited smoke success means the runtime is active by default

## Fourth Attempt Evidence Status

The fourth approved local-only manual real-smoke attempt is documented and reviewed.

What this means:

- the smoke executed once locally and failed safely
- `result_status=provider_error`
- `error_category=provider_error`
- `sdk_missing` did not occur
- `grounding_status=insufficient_evidence`
- cleanup was completed
- the normal route remained mock-first
- provider routing activation remained disabled
- the Google AI Studio Usage page showed 2 API requests and 2 API errors
- the error type shown by the user was `429 TooManyRequests`

What this does not mean:

- real smoke execution is complete
- Gemini is connected
- Gemini is active
- temporary local-shell `GEMINI_API_KEY` usage means Gemini is active by default
- the next smoke should be run immediately
- the external Gemini availability situation is understood or enough time has passed
- the smoke context is grounded; the remaining blocker is external Gemini availability

Do not run another smoke until both conditions above are true.

## Ninth Attempt Evidence Status

The ninth approved local-only manual real-smoke attempt is documented and reviewed.

What this means:

- the smoke executed once locally and failed safely
- `result_status=provider_error`
- `error_category=provider_error`
- `provider_error_stage=client_invocation`
- `provider_error_reason=service_unavailable`
- `grounding_status=grounded`
- `safety_status=pass`
- `provider_used=mock`
- `fallback_used=true`
- cleanup was completed
- the normal route remained mock-first
- provider routing activation remained disabled

What this does not mean:

- real smoke execution is complete
- Gemini is connected
- Gemini is active
- `GEMINI_API_KEY` has been used
- the next smoke should be run immediately
- the external Gemini availability situation is understood or enough time has passed
- the smoke context is already grounded, but the next attempt still requires explicit approval and a fresh external-availability check

Do not run another smoke until the external Gemini availability situation is understood or enough time has passed, and any future attempt remains explicitly approved.

## Approval Gate Wording

The only approval that can unlock execution is:

> Approve a single local-only manual Gemini real-smoke attempt, using a temporary local shell environment only, with no Docker, no EC2, no CI, and no default-test execution.

If this wording is not explicitly approved, real smoke stays blocked.

This document does not approve execution.
This document does not activate Gemini.
This document does not prove real LLM readiness.
This document does not claim production/deployment/HTTPS readiness.

## Prerequisites Before First Real Smoke

All of the following are required before any future real smoke may be attempted:

- explicit user approval using the approval wording above
- clean git status
- latest validation suite passing
- `google-genai` available in the local API / backend environment
- `GEMINI_API_KEY` available only as a temporary local environment variable
- no key committed
- no key printed
- normal default provider remains `mock`
- `AGENT_ENABLE_LLM=false` for normal runtime
- first smoke must be manual and local-only
- fallback / kill switch documented
- no Docker / Compose / EC2 activation

## Secret Handling Rules

The key handling rules for a future smoke are:

- `GEMINI_API_KEY` must be exported only in a local shell or session
- the key may be set temporarily in a local terminal before the smoke and unset immediately after
- the key must never be written to files
- the key must never be stored in docs, logs, shells history, test fixtures, or artifacts
- command output must be reviewed for secret leakage
- no `.env` commit
- no Compose file key

Safe placeholder examples only:

```bash
export GEMINI_API_KEY="<set-locally-only>"
unset GEMINI_API_KEY
```

## Pre-Smoke Validation Commands

Run these checks immediately before any future real smoke, after explicit approval:

```bash
git status --short --untracked-files=all
python -m compileall frontend tests api src/inspection_ai scripts
pytest tests/agent/test_provider_router.py -q
pytest tests/agent/test_gemini_provider_mocked_client.py -q
pytest tests/agent/ -q
pytest tests/api/test_agent_endpoint.py -q
```

If the working tree is not clean or the tests fail, do not run the smoke.

## Final Execution Checklist Status

The final real-smoke execution checklist is documented and reviewed.

What this means:

- the mandatory pre-execution checks are defined
- the one-call rule is defined
- the safe input and output boundaries are defined
- the stop / fail conditions are defined
- the required post-smoke checks and evidence document are defined

What this does not mean:

- real smoke execution is approved
- Gemini is connected
- Gemini is active
- `GEMINI_API_KEY` has been used

## Dry-Run Verification Evidence Status

The harness dry-run verification evidence is documented and reviewed.

What this means:

- the default harness invocation was verified in dry-run mode
- the explicit `--dry-run` path was verified
- `--execute` without confirmation was blocked
- the explicit `--execute` + confirmation path is now implemented in code, but a real smoke has not been run
- no real Gemini API call was made
- no key was read
- the mock-first normal route remained unchanged
- provider routing activation remained disabled

What this does not mean:

- real smoke execution is complete
- Gemini is connected
- Gemini is active
- `GEMINI_API_KEY` has been used

## Preconditions Before First Real Smoke

All of the following are required before any future real smoke may be attempted:

- explicit user approval using the approval wording above
- clean git status
- latest validation suite passing
- `google-genai` available in the local API / backend environment
- `GEMINI_API_KEY` available only as a temporary local environment variable
- no key committed
- no key printed
- normal default provider remains `mock`
- `AGENT_ENABLE_LLM=false` for normal runtime
- first smoke must be manual and local-only
- fallback / kill switch documented
- no Docker / Compose / EC2 activation

## Environment Handling

The key handling rules for a future smoke are:

- `GEMINI_API_KEY` must be exported only in a local shell or session
- the key may be set temporarily in a local terminal before the smoke and unset immediately after
- the key must never be written to files
- the key must never appear in docs, logs, shell history, or test output
- command output must be reviewed for secret leakage
- the key must be unset after the smoke
- no `.env` commit
- no Compose file key

Safe placeholder examples only:

```bash
export GEMINI_API_KEY="<set-locally-only>"
unset GEMINI_API_KEY
```

## First Real-Smoke Scope

The first future smoke, if explicitly approved, must be:

- local-only
- manually triggered
- one minimal sanitized request
- not part of CI
- not part of the default test suite
- not run through Docker
- not run through EC2
- not exposed on a public endpoint
- not a frontend workflow
- not a production workflow
- not a CI workflow
- not a Docker / Compose workflow
- not an EC2 workflow
- not a retry loop

## What the First Real Smoke May Test

A future first smoke may test only:

- lazy SDK import works
- key presence is detected without exposing the key
- the real provider boundary can produce one safe response
- post-generation safety guard runs
- fallback behavior works after the smoke
- the normal mock route remains unchanged after the smoke
- provider_used is accurate for the explicit smoke path
- the response remains reviewable and does not claim production readiness

## What the First Real Smoke Must Not Test

A future smoke must not:

- route normal `/agent/explain` traffic to Gemini
- expose a public API
- run through Docker or Compose
- run on EC2
- send raw artifacts
- send full evidence JSON
- send file paths
- send secrets
- run bulk prompts
- run repeated calls
- measure production performance
- claim production readiness
- claim HTTPS or domain readiness
- claim real LLM readiness

## Smoke Input Boundary

The only allowed input for a future smoke is:

- one short sanitized user question
- compact `page_id`, `section_id`, and `component_id`
- compact evidence sources only
- manual review language required
- no raw images, files, or artifacts
- no local paths
- no secrets
- no hidden logs
- no raw provider output capture outside the approved evidence block

The future smoke must remain local-only and manual. If the input cannot be reduced to this boundary, the smoke is not allowed.

## Proposed Future Command Shape

This is a planning sketch only. Do not run it yet.

```bash
# Future manual local smoke only; not to be run yet.
# Requires explicit user approval, local-only secret export, and a reviewed safe prompt.
python -m inspection_ai.agent.run_gemini_local_smoke \
  --question "<sanitized question>" \
  --component-id "<component_id>" \
  --page-id "<page_id>" \
  --section-id "<section_id>"
```

The actual smoke command may differ. It must remain local-only, manual, and reviewed before use.

## PASS Criteria

A future first real smoke passes only if:

- exactly one intended real Gemini call is made
- no secret appears in output
- the response is safe
- the response includes manual review language
- `provider_used` is accurate for the explicit smoke path
- fallback still works after a failure test
- normal `/agent/explain` remains mock-first
- no docs, runtime, frontend, Docker, or EC2 behavior changes happen
- the key is unset immediately after the smoke
- the approved evidence is captured separately and contains no secret material

## FAIL Criteria

A future smoke fails if:

- the key leaks
- raw path or evidence leaks
- normal runtime routes to Gemini
- Gemini is used without explicit command
- the output claims production-ready, deployment-ready, or manual-review-not-needed
- unsafe output reaches display
- multiple unintended calls occur
- Docker, EC2, or a public endpoint is involved
- the key leaks to logs, shell history, docs, or artifacts
- the smoke is run without explicit user approval
- the smoke claims production, deployment, HTTPS, or real LLM readiness

## Rollback / Kill Switch

If a future smoke is ever run, rollback is:

- `unset GEMINI_API_KEY`
- `AGENT_ENABLE_LLM=false`
- `AGENT_DEFAULT_PROVIDER=mock`
- stop any running local process
- revert any smoke-specific commit if one exists
- no frontend rollback is needed if frontend is unchanged
- return `AGENT_ENABLE_LLM=false` and `AGENT_DEFAULT_PROVIDER=mock`
- verify `/agent/explain` is mock-first again after cleanup
- confirm no smoke-only artifacts remain in tracked files

## Evidence to Record After a Future Smoke

When a future smoke is eventually run, record:

- date / time
- command shape without the secret
- provider status
- sanitized request summary
- sanitized response summary
- pass / fail decision
- proof that the key was not logged
- proof that the normal route remains mock
- proof that the test suite still passes
- no raw provider response if unsafe
- whether the smoke path stayed local-only and manual
- whether `provider_used` reflects the approved explicit smoke path
- whether Gemini was disabled again after the smoke unless separately approved

## Documentation Update Requirement After a Future Smoke

If a real smoke is later run, create a separate evidence document.
Do not modify this plan to pretend execution happened.
Do not mark smoke complete until it actually happens.
The evidence document must be separate from this plan and must not replace it.

## Final Decision

Current status: `READY_FOR_LOCAL_REAL_SMOKE_APPROVAL`

Decision:

- PASS for local real-smoke planning readiness
- FAIL for real-smoke execution readiness until explicit user approval

Real Gemini API call remains blocked until the next approved slice.
This document does not approve execution.
This document does not activate Gemini.
This document does not prove real LLM readiness.
This document does not claim production/deployment/HTTPS readiness.

## Recommended Next Slice

Recommended next slice: **pause further attempts until the external Gemini availability situation is understood or enough time has passed, then keep the grounded smoke context and only retry with explicit approval**

Why this is the safest next move:

- it keeps the current mock-first runtime intact
- it does not require any real API call
- it avoids another attempt while the current external availability signal is unresolved
- it keeps the smoke context grounded for future retries
- it preserves the requirement for a separate evidence document after any approved smoke

If the user does not approve the exact approval gate wording above, the next step is to remain blocked.
