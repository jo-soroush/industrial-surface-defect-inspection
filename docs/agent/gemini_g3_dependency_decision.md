# Gemini G3 Dependency Installation Decision

## Executive Summary

This document is a documentation / decision slice only.

Real Gemini provider implementation has not started.
No package is installed.
No SDK import is added.
No Gemini API call is implemented.
The runtime remains mock-first.

This slice exists to decide how the future Gemini dependency should be handled without silently changing runtime behavior.

Outcome:

- `requirements-api.txt` now carries `google-genai` as the backend/API dependency entry.
- No runtime import was added.
- No package was installed in this step.
- No Gemini API call was added.

## Repository Dependency Structure Found

The repository contains the following dependency-related files:

- `requirements.txt`
- `requirements-api.txt`
- `requirements-frontend.txt`
- `requirements-dev.txt`
- `pyproject.toml`
- `Dockerfile`
- `Dockerfile.api`
- `Dockerfile.frontend`
- `docker-compose.yml`
- `README.md`

Observed structure:

- `requirements-api.txt` is the backend / API dependency set.
- `requirements-frontend.txt` is the frontend dependency set.
- `requirements-dev.txt` includes both runtime dependency files plus tooling.
- `Dockerfile.api` installs `requirements-api.txt`.
- `Dockerfile.frontend` installs `requirements-frontend.txt`.
- The top-level `Dockerfile` installs `requirements.txt`.
- `pyproject.toml` exists, but it only defines build metadata and does not currently define the project dependency policy.

Uncertainty:

- The repository has multiple dependency entry points.
- The backend/API target is clear for the API container, but the top-level `requirements.txt` still exists as a separate install path.
- Because there is more than one possible runtime path, a dependency change should not be made silently in this slice.

## Candidate Dependency

- Selected future package candidate: `google-genai`
- Exact version pinning: still pending
- Dependency placement: backend/API only
- Frontend requirements did not receive `google-genai`
- Dev/test requirements did not receive `google-genai`

## Decision

Conservative decision: **dependency change implemented in requirements-api.txt only**

Reason:

- The repository structure shows more than one runtime dependency entry point.
- The backend/API path is clear for `Dockerfile.api`.
- The dependency was added only to `requirements-api.txt`, leaving frontend and dev dependency files unchanged.
- The top-level `requirements.txt` remains unchanged.

Recommended next action:

- Review the added backend/API dependency entry.
- Keep the package version decision pending until a separate implementation slice chooses a pin.

## Version Pinning Strategy

- Exact version pinning is still pending.
- If the project convention is pinned runtime dependencies, follow that convention.
- If the project convention is unpinned runtime dependencies, document the trade-off explicitly before changing the file.
- Do not silently introduce an unpinned production dependency without review.

## Runtime Safety Requirements After Dependency Addition

The installed dependency must not activate Gemini.

- `AGENT_ENABLE_LLM=false` remains the default.
- `AGENT_DEFAULT_PROVIDER=mock` remains the default.
- `real_provider_implemented` must remain `False` until the real provider implementation is approved.
- `available` must remain `False` until actual provider implementation is complete and explicitly enabled.
- `/agent/explain` must remain mock by default.

## Required Tests After Dependency Change

If a future dependency slice is approved, the following checks must remain green:

- `python -m compileall frontend tests api src/inspection_ai scripts`
- `pytest tests/agent/ -q`
- `pytest tests/api/test_agent_endpoint.py -q`
- `pytest tests/frontend/ -q`
- `pytest tests/docker/ -q`
- grep for no executable Gemini import in the runtime module
- optional import smoke only if the dependency is already installed locally and explicitly approved

## Docker / Compose Implication

- Dockerfiles and Compose must not change in this slice.
- LLM-disabled Docker / Compose smoke remains required after any dependency change.
- Compose must not contain a real `GEMINI_API_KEY`.
- The default Compose path remains LLM disabled.

## EC2 Implication

- No EC2 changes are part of this slice.
- No EC2 Gemini activation is allowed.
- Future EC2 key injection must be a separate later step.
- Rollback remains environment-based.

## Allowed Next Slice

Safest next slice: **C. real provider skeleton with mocked SDK object only**

Reason:

- the dependency entry now exists in the backend/API requirements file
- the next highest-risk change is real provider code, not another dependency move
- the runtime should remain mock-first until a separate approved provider slice begins

## Explicit Non-Implementation Statement

This document does not add a real Gemini implementation.
No Gemini API call is implemented.
No real LLM execution is active.
Gemini runtime activation is not active.
The system remains mock-first.
