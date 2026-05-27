# Gemini G3 Real Provider Execution Gate Design

## Executive Summary

This document is documentation only.

Real Gemini provider execution has not started.
No real Gemini API call exists.
No executable SDK import is added.
Runtime activation is not active.
`/agent/explain` remains mock-first.
Provider routing to Gemini is not implemented.
This document does not approve real provider activation.

The purpose of this gate design is to define the disabled-by-default activation and routing strategy that must exist before any future Gemini execution can be considered.

What this controls:

- when Gemini can be considered available
- when Gemini can be routed from `/agent/explain`
- when fallback to mock must remain the only path
- which health and readiness metadata may be reported
- which safety checks must pass before and after any future provider call

What remains blocked:

- any real Gemini API call
- any automatic provider routing to Gemini
- any activation by dependency installation alone
- any runtime change that makes Gemini active by default

## Completed Foundation

The following foundation is already in place:

- readiness scaffolding
- lazy SDK loader boundary
- health / readiness metadata
- pre-real-call audit
- dependency decision
- `requirements-api.txt` `google-genai` dependency entry
- provider skeleton with injected / mock client
- deterministic safety guard
- provider contracts
- fallback policy
- current green tests

## Required Runtime Gates Before Gemini Can Answer Any `/agent/explain` Request

All of the following must be true before Gemini can answer a request:

- explicit user approval
- `AGENT_ENABLE_LLM=true`
- `AGENT_DEFAULT_PROVIDER` or provider order explicitly allows Gemini
- `GEMINI_API_KEY` present through environment or secret manager only
- `google-genai` installed in the API / backend environment
- lazy SDK import succeeds
- `real_provider_implemented=True`
- `activation_allowed=True`
- `provider_allowed=True`
- pre-generation safety guard passes
- sanitized provider request only
- allowed context boundary enforced
- post-generation safety guard passes
- fallback to mock remains enabled
- health reports safe status only
- no raw secret / path / evidence exposure

## Disabled-by-Default Behavior

The default behavior must remain disabled:

- `AGENT_ENABLE_LLM=false` by default
- default provider remains `mock`
- Gemini must not be in `available_providers` unless it is genuinely available
- `/agent/explain` must not route to Gemini by default
- local Docker / Compose defaults must remain LLM disabled
- missing key or missing SDK must never break the mock path

## Provider Routing Design

Future routing behavior must follow these rules:

- default route remains `mock`
- Gemini route is allowed only when all gates pass
- if requested provider is Gemini but gates fail, fallback to mock
- if default provider is Gemini but gates fail, fallback to mock
- if Gemini errors, fallback to mock
- if Gemini output is unsafe, fallback to mock
- if fallback is disabled, return a safe non-secret error
- `provider_used` must reflect the real provider or mock fallback accurately

## `real_provider_implemented` Gate

- currently `False`
- must remain `False` until real provider execution code exists and is tested
- cannot become `True` from environment variables alone
- must be controlled by code
- must be covered by tests
- must be visible in health / readiness metadata

## `activation_allowed` Gate

`activation_allowed` must require every condition below:

- `llm_enabled`
- `key_present`
- `sdk_available`
- `provider_allowed`
- `real_provider_implemented`
- `safety_ready`
- fallback policy is defined

This gate must remain `False` until the provider execution slice is approved.
Dependency installation alone must never make it `True`.

## Safety and Context Boundary

Pre-generation safety guard requirements:

- check the request before any provider call
- send only sanitized user question and compact context / evidence
- forbid raw artifacts, logs, paths, secrets, env vars, and unrestricted JSON

Post-generation safety guard requirements:

- inspect Gemini output before it is returned
- block unsafe output instead of displaying it directly
- block unsupported metric invention
- keep manual review language visible

## Health / Readiness Design

Health may expose:

- Gemini status
- available boolean
- configured / `key_present` boolean
- `llm_enabled` boolean
- `sdk_checked` boolean
- `sdk_status`
- `activation_allowed` boolean
- `real_provider_implemented` boolean
- `fallback_available` boolean

Health must not expose:

- key values
- raw env values
- local paths
- full provider prompts
- raw provider responses
- hidden evidence

## Test Requirements Before Any Router Activation Code

Required tests:

- default mock route
- `AGENT_ENABLE_LLM=false` with key present still mock
- `AGENT_ENABLE_LLM=true` without key fallback mock
- SDK missing fallback mock
- SDK load error fallback mock
- `real_provider_implemented=False` prevents activation
- `activation_allowed=False` prevents routing
- requested Gemini with failed gates fallback mock
- Gemini safe mocked response only in test seam
- unsafe Gemini output blocked
- invented metrics blocked
- raw path / secret output blocked
- timeout / rate-limit / provider error fallback mock
- health does not expose secrets
- API schema remains compatible
- frontend tests unchanged
- Docker tests unchanged

## Future Implementation Slices

Recommended safe order:

A. router activation tests only, mocked provider, disabled by default
B. real provider execution code with lazy import, but no default activation
C. local manual real smoke only, explicitly triggered
D. Docker / Compose LLM-disabled regression
E. later Gemini-enabled local smoke, manual approval only
F. EC2 activation only after local smoke and safety review

## Rollback / Kill Switch

- `AGENT_ENABLE_LLM=false`
- `AGENT_DEFAULT_PROVIDER=mock`
- remove `GEMINI_API_KEY` from the environment
- fallback to mock
- revert the commit
- no frontend rollback is needed if the frontend is unchanged

## Final Decision

Current status: `READY_FOR_DISABLED_BY_DEFAULT_ROUTER_PLANNING`

Decision:

- PASS for gate design readiness
- FAIL for runtime activation readiness

The repository is not ready for uncontrolled real provider activation.

Real Gemini provider execution may begin only after explicit user approval and the next approved slice.

Real API call remains blocked until the next approved slice.

## Next Recommended Slice

Recommended next slice: **A. router activation tests only, mocked provider, disabled by default**

Why this is the safest next move:

- it proves the disabled-by-default routing gate without enabling real Gemini execution
- it keeps the mock-first runtime intact
- it does not require a real Gemini API call
- it gives the next runtime routing boundary explicit test coverage before any provider execution code exists
