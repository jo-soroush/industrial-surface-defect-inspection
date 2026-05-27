# LLM-Disabled Docker / Compose Readiness Audit

## Executive Summary

This document is a read-only readiness audit for the Phase 12 local validation step before Gemini.

It records what is already known from repository inspection about the Docker / Compose layout, the LLM-disabled defaults, and the validation surface that should be checked later. It does not run Docker, does not run Docker Compose, and does not claim that a live runtime validation has already been completed in this task.

This audit exists to prepare the next explicit validation step:

- confirm the compose and container wiring with LLM disabled
- confirm the API and frontend containers start cleanly
- confirm the agent layer stays mock-first
- confirm no secret or real LLM activation is required

What remains not run in this task:

- Docker
- Docker Compose
- AWS / EC2 commands
- Gemini / Grok / OpenAI API calls
- any external network call
- any secret inspection of `GEMINI_API_KEY`

## Roadmap Alignment

This audit belongs to Phase 12 - Local Validation Before Gemini.

It is required before any Gemini implementation or real LLM activation work can move forward.

It is a prerequisite planning artifact, not a runtime approval.

## Current Docker / Compose Structure

The repository currently defines a split service layout in `docker-compose.yml`:

- `api`
  - build file: `Dockerfile.api`
  - exposed port: `8000:8000`
  - depends on: none
- `frontend`
  - build file: `Dockerfile.frontend`
  - exposed port: `8501:8501`
  - depends on: `api`

Additional structure observed:

- `frontend` points to the API service by DNS alias using `STREAMLIT_API_BASE_URL=http://api:8000`
- no compose volumes are mounted
- no extra observability / runtime services are defined in compose
- the API service is configured with LLM-disabled defaults
- the frontend service does not request agent/LLM defaults

The repository also contains a top-level `Dockerfile`, but the Compose path uses the split `Dockerfile.api` and `Dockerfile.frontend` images.

## Current LLM-Disabled Defaults

The current compose and Dockerfile defaults are mock-first:

- `AGENT_ENABLE_LLM=false`
- `AGENT_DEFAULT_PROVIDER=mock`
- `LLM_PROVIDER_ORDER=mock,gemini,grok`
- `LLM_ENABLE_FALLBACK=true`

Observed implications from the files:

- no `GEMINI_API_KEY` is required for the current default path
- Gemini, Grok, and OpenAI remain inactive by default
- `/agent/explain` is expected to stay mock-first unless a future explicit activation path is approved

## Files Inspected / Evidence Sources

The following files were inspected for this audit:

- `docker-compose.yml`
- `Dockerfile`
- `Dockerfile.api`
- `Dockerfile.frontend`
- `requirements.txt`
- `requirements-api.txt`
- `requirements-frontend.txt`
- `requirements-dev.txt`
- `api/app/main.py`
- `api/app/routes/system.py`
- `api/app/routes/agent.py`
- `api/app/routes/inspection.py`
- `frontend/streamlit_app.py`
- `tests/docker/*`
- `tests/api/test_agent_endpoint.py`
- docs / agent readiness files
- `docs/deployment/ec2_readiness_plan.md`

## Future Validation Command Plan

This command plan is **NOT RUN** in this task.

Planned sequence:

```bash
git status --short --untracked-files=all
python scripts/runtime_assets/stage_runtime_assets.py --manifest configs/runtime_assets/manifest.yaml --check
docker compose config
AGENT_ENABLE_LLM=false LLM_ENABLE_FALLBACK=true docker compose build
AGENT_ENABLE_LLM=false LLM_ENABLE_FALLBACK=true docker compose up --build -d
curl http://localhost:8000/health
curl http://localhost:8000/agent/health
curl POST http://localhost:8000/agent/explain with a mock request
curl POST http://localhost:8000/inspect/image with a sample image if a safe sample exists
curl http://localhost:8501
docker compose logs --tail=100 api frontend
docker compose down
pytest tests/docker/ -q
pytest tests/api/test_agent_endpoint.py -q
pytest tests/frontend/ -q
pytest tests/agent/ -q
git status --short --untracked-files=all
```

## PASS Criteria

Future actual validation passes only if all of the following hold:

- compose config is valid
- services build and start
- API health passes
- agent health reports mock-first / LLM disabled
- agent explain returns a mock / fallback-safe response
- image inspection endpoint works if included
- frontend is reachable
- frontend can reach the API
- no real LLM call is needed
- no key is needed
- logs remain safe
- containers stop cleanly
- post-validation tests pass
- working tree stays clean

## FAIL Criteria

Future actual validation fails if any of the following occurs:

- compose config is invalid
- any service fails to build or start
- API health fails
- frontend cannot reach the backend
- `/agent/health` exposes secrets
- `/agent/explain` tries Gemini, OpenAI, or Grok
- missing `GEMINI_API_KEY` breaks startup
- Docker requires a real LLM key
- logs contain secret-like values
- the working tree changes unexpectedly

## Evidence to Collect Later

If actual Docker / Compose validation is later approved and run, collect:

- command outputs
- health response summaries
- agent health summary
- agent explain response summary
- inspect image response summary, if included
- frontend reachability confirmation
- container logs summary
- no-secret confirmation
- PASS / FAIL decision
- `git status`
- post-validation test results

## Risks / Unknowns

- Docker / Compose runtime has not been executed in this task
- runtime asset staging completeness must still be checked
- the top-level `Dockerfile` is an alternate image path and could confuse validation if used accidentally
- `google-genai` exists in `requirements-api.txt`, but that does not activate Gemini by itself
- the frontend depends on `STREAMLIT_API_BASE_URL` and the compose service alias `api`
- if `/inspect/image` is included in the smoke path, a safe sample image path must be chosen

## Final Decision

Status: `READY_FOR_USER_APPROVAL_TO_RUN_LLM_DISABLED_DOCKER_COMPOSE_VALIDATION`

Decision:

- PASS for read-only readiness planning
- FAIL / PENDING for actual Docker / Compose runtime validation

This audit does not claim actual Docker / Compose validation is complete.

## Validation Evidence Update

The actual Phase 12 LLM-disabled Docker / Compose validation evidence is now documented in `docs/agent/llm_disabled_docker_compose_validation_evidence.md`.

That evidence records the local Docker / Compose run, the mock-first defaults, the safe `/agent/health` and `/agent/explain` results, the frontend reachability check, the logs summary, the clean shutdown, and the post-validation test suite.

## Recommended Next Slice

The next safe step is to update and confirm the roadmap status for the Phase 12 PASS, then continue with EC2 readiness planning while keeping LLM disabled.

Do not run Docker, Docker Compose, AWS / EC2, or any external provider call in this task.
