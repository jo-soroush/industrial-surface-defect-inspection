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

This plan defines the narrowest local-only evidence path for a future manual smoke while keeping the current mock-first runtime untouched.

A disabled-by-default harness skeleton exists in `scripts/agent/run_gemini_local_smoke.py`; it is not executed by default and it does not call Gemini.

What this plan controls:

- the preconditions for a future local manual smoke
- the safe environment handling for a future smoke
- the minimal smoke input boundary
- the pass / fail criteria for a future smoke
- the rollback / kill switch path after a future smoke
- the evidence that must be recorded if a future smoke is ever approved and run

What remains blocked:

- any real Gemini API call
- any runtime activation
- any provider routing activation
- any Docker / Compose / EC2 based smoke
- any CI or default-test smoke
- any public or production-facing endpoint exposure

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
- `GEMINI_API_KEY` has been used

## Preconditions Before First Real Smoke

All of the following are required before any future real smoke may be attempted:

- explicit user approval
- clean git status
- latest validation suite passing
- `google-genai` available in the local API / backend environment
- `GEMINI_API_KEY` available only as a local environment variable
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
- the key must never be written to files
- the key must never appear in docs, logs, or test output
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

## What the First Real Smoke May Test

A future first smoke may test only:

- lazy SDK import works
- key presence is detected without exposing the key
- the real provider boundary can produce one safe response
- post-generation safety guard runs
- fallback behavior works after the smoke
- the normal mock route remains unchanged after the smoke

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

## Rollback / Kill Switch

If a future smoke is ever run, rollback is:

- `unset GEMINI_API_KEY`
- `AGENT_ENABLE_LLM=false`
- `AGENT_DEFAULT_PROVIDER=mock`
- stop any running local process
- revert any smoke-specific commit if one exists
- no frontend rollback is needed if frontend is unchanged

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

## Documentation Update Requirement After a Future Smoke

If a real smoke is later run, create a separate evidence document.
Do not modify this plan to pretend execution happened.
Do not mark smoke complete until it actually happens.

## Final Decision

Current status: `READY_FOR_LOCAL_REAL_SMOKE_APPROVAL`

Decision:

- PASS for local real-smoke planning readiness
- FAIL for real-smoke execution readiness until explicit user approval

Real Gemini API call remains blocked until the next approved slice.

## Recommended Next Slice

Recommended next slice: **A. local real-smoke harness / script skeleton, disabled by default, not executed**

Why this is the safest next move:

- it prepares the manual smoke path without running Gemini
- it keeps the mock-first runtime intact
- it does not require any real API call
- it avoids accidental execution before approval
