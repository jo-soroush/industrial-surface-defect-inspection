# Gemini G3 Local Real-Smoke Success Limited Evidence

## Executive Summary

This document records the latest approved local-only manual Gemini 2.5 Flash smoke result that succeeded in limited mode.

Observed result:

- `gemini_local_smoke_status=SUCCESS_LIMITED`
- `smoke_model=gemini-2.5-flash`
- `smoke_success_level=limited`
- `result_status=limited`
- `provider_used=gemini`
- `fallback_used=false`
- `grounding_status=grounded`
- `safety_status=limited`
- `smoke_exit_code=0`
- `cleanup_done=true`
- `GEMINI_API_KEY_unset=true`
- `AGENT_ENABLE_LLM=false`
- `AGENT_DEFAULT_PROVIDER=mock`

No secret appeared in the output.
No Docker, EC2, CI, frontend workflow, or public endpoint was involved.
This evidence does not claim a successful unrestricted Gemini response.
This document does not claim production readiness, deployment readiness, HTTPS/domain readiness, public API readiness, or real LLM readiness.

## Attempt Outcome

The latest approved Gemini 2.5 Flash smoke reached the provider path successfully and returned a limited, safe response.

The harness recorded:

- provider used: `gemini`
- fallback used: `false`
- grounding status: `grounded`
- safety status: `limited`
- success level: `limited`

This proves the local smoke-level Gemini response path worked for `gemini-2.5-flash`.
The gated frontend-to-agent path is the same explicit opt-in route used by the active Image Inspection explanation flow when the runtime gates are enabled.

## What This Means

- The smoke harness can reach the Gemini provider boundary for `gemini-2.5-flash`.
- The response path can return a safe limited result without falling back to mock.
- The validated smoke model is now `gemini-2.5-flash`.
- The smoke harness default and the Gemini real-provider default are aligned on `gemini-2.5-flash`.
- The explicit local gated runtime uses `AGENT_ENABLE_LLM=true`, `AGENT_ENABLE_REAL_PROVIDER_RUNTIME=true`, `AGENT_DEFAULT_PROVIDER=gemini`, `LLM_PROVIDER_ORDER=gemini,mock`, `LLM_ENABLE_FALLBACK=true`, `LLM_MAX_RETRIES=1`, with `GEMINI_API_KEY` supplied only at runtime and never committed.
- Runtime remains mock-first by default.
- Gated runtime activation remains disabled by default.

## What This Does Not Mean

- This does not mean real smoke execution is unrestricted.
- This does not mean Gemini is available only through the explicit gated `/agent/explain` runtime path and is not active by default.
- This does not mean Gemini is active by default.
- This does not mean Gemini is production-ready.
- This does not mean Gemini is deployment-ready.
- This does not mean Gemini is public-ready.
- This does not mean Gemini is HTTPS/domain-ready.
- This does not mean Gemini is real-LLM runtime-ready.

## Cleanup / Restoration Confirmed

After the attempt:

- `GEMINI_API_KEY` was unset during cleanup
- `AGENT_ENABLE_LLM=false` was restored
- `AGENT_DEFAULT_PROVIDER=mock` was restored
- cleanup was completed

## Current Interpretation

- `gemini-2.5-flash` is the aligned smoke and provider default model.
- `gemini-3-flash-preview` remains outside the current validated project path after a `service_unavailable` failure.
- The current next technical step is gated runtime validation, not Docker, EC2, UI redo, or API key storage.

## What This Does Not Change

- Normal `/agent/explain` remains mock-first by default.
- Provider routing remains disabled by default.
- Safety guard behavior is unchanged.
- No real LLM runtime is active by default.
