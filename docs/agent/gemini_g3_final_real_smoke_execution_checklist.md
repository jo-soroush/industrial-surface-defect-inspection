# Gemini G3 Final Real-Smoke Execution Checklist

## Executive Summary

This is a checklist only.

No real Gemini API call is run by this document.
No `GEMINI_API_KEY` is used, read, printed, or validated by this document.
No runtime activation is performed.
No provider routing activation is performed.
Normal `/agent/explain` remains mock-first.
This checklist does not approve real smoke execution.
A future real smoke still requires explicit user approval in a separate step.

This checklist exists to define the mandatory review items that must be satisfied before the first real Gemini local smoke can ever be approved.

What this checklist controls:

- the pre-execution gate checks
- the command boundary for a future smoke
- the key handling rules
- the one-call rule
- the safe input and safe output boundaries
- the stop / fail conditions
- the required post-smoke checks and evidence document

What remains blocked:

- any real Gemini API call
- any runtime activation
- any provider routing activation
- any Docker / Compose / EC2-based execution
- any CI or default-test execution
- any public or production-facing endpoint exposure

## Mandatory Pre-Execution Checks

All of the following must be true before the first smoke:

- user gives explicit approval
- git status is clean
- latest validation suite passes
- local harness exists
- local plan exists
- real provider boundary exists
- provider routing remains disabled by default
- normal `/agent/explain` remains mock-first
- `requirements-api.txt` contains `google-genai`
- user has confirmed they understand this may make one real Gemini API call
- no Docker / Compose / EC2 involved
- no frontend involved
- no CI involved
- no public endpoint involved

## Allowed Command Boundary

Only the local harness may be used for the first smoke.

The command must be:

- local shell only
- manual only
- explicitly confirmed before any future non-dry-run path
- supplied with one sanitized question only
- not run through pytest
- not run through Docker or Compose
- not run on EC2
- not run through the frontend

## Forbidden Commands / Forbidden Execution Paths

The following are explicitly forbidden:

- running Docker or Compose
- running AWS / EC2
- exposing a public API
- changing the provider default to Gemini
- changing the frontend
- changing registry, evidence, or model files
- bulk prompts
- repeated calls
- raw artifact sending
- full evidence JSON sending
- sending local paths
- sending secrets
- committing a key
- writing a `.env` file with a key
- printing a key

## `GEMINI_API_KEY` Handling Checklist

- key must be exported only in a local shell
- key must never be written to files
- key must never be committed
- key must never appear in terminal output
- key must be unset immediately after smoke
- output must be reviewed for leakage
- placeholder examples only

Safe placeholders only:

```bash
export GEMINI_API_KEY="<set-locally-only>"
unset GEMINI_API_KEY
```

## One-Call Rule

The first smoke allows exactly one intended Gemini call.

- no retry loop
- no bulk execution
- no automated repeat
- if the call fails, stop and record the failure
- do not immediately retry without review

## Safe Input Checklist

Allowed input:

- one short sanitized question
- compact `page_id`
- compact `section_id`
- compact `component_id`
- compact evidence-source labels only
- manual review language required

Forbidden input:

- raw image
- raw file
- artifact path
- local path
- secret
- full evidence JSON
- hidden logs
- production claims

## Expected Safe Output Checklist

Output must:

- not expose the key
- not expose paths
- not expose raw evidence
- not claim production-ready
- not claim deployment-safe
- not say manual review is unnecessary
- include limitation / manual-review language
- be safe to display only after the safety guard
- record `provider_used` accurately for the explicit smoke path

## Immediate Stop / FAIL Conditions

Stop immediately if:

- a secret appears in output
- a raw path appears
- raw evidence appears
- more than one call is attempted
- the normal route changes from mock
- provider routing activates globally
- the output claims production readiness
- the output claims deployment readiness
- the output says manual review is not needed
- Docker / Compose / EC2 becomes involved
- frontend behavior changes

## Required Post-Smoke Checks

If a future smoke is eventually run, immediately run:

- `unset GEMINI_API_KEY`
- `git status --short --untracked-files=all`
- `python -m compileall frontend tests api src/inspection_ai scripts`
- `pytest tests/agent/ -q`
- `pytest tests/api/test_agent_endpoint.py -q`
- `pytest tests/frontend/ -q`
- `pytest tests/docker/ -q`
- prove normal `/agent/explain` remains mock-first
- inspect output for secret leakage

## Evidence Document Requirement

After a future real smoke, create a separate evidence document:

`docs/agent/gemini_g3_local_real_smoke_evidence.md`

It must include:

- date / time
- command shape without secret
- sanitized request summary
- sanitized response summary
- PASS / FAIL
- proof no key leaked
- proof one-call only
- proof normal route stayed mock
- proof test suite passed
- no raw unsafe provider response
- no real key

## Rollback / Kill Switch

- `unset GEMINI_API_KEY`
- `AGENT_ENABLE_LLM=false`
- `AGENT_DEFAULT_PROVIDER=mock`
- stop the local process
- revert smoke-specific changes if any
- do not rollback frontend if unchanged

## Final Decision

Current status: `READY_FOR_USER_APPROVAL_BEFORE_FIRST_REAL_SMOKE`

Decision:

- PASS for final real-smoke checklist readiness
- FAIL for real-smoke execution readiness until explicit user approval

Real Gemini API call remains blocked until the next approved slice.

## Recommended Next Slice

Recommended next slice: **one more dry-run verification of the harness before first real smoke**

Why this is the safest next move:

- it verifies the harness without using a key
- it keeps the mock-first runtime intact
- it avoids accidental execution before approval
