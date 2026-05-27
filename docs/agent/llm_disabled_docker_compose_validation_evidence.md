# LLM-Disabled Docker / Compose Validation Evidence

## Executive Summary

This document records the actual LLM-disabled Docker / Compose validation evidence for Phase 12.

Purpose:

- confirm the local split-image Docker / Compose path works with LLM disabled
- confirm the API, agent, and frontend surfaces start cleanly
- confirm the runtime remains mock-first
- confirm no real LLM provider was required

Commit validated:

- `a44370a`

Scope:

- local only
- mock-first only
- LLM disabled only
- no Gemini / Grok / OpenAI runtime activation

Final decision:

- PASS for Phase 12 LLM-disabled Docker / Compose local validation
- FAIL / NOT CLAIMED for production deployment readiness
- FAIL / NOT CLAIMED for real LLM readiness

## Commands Run and Evidence Summaries

### Preflight and Compose Config

Observed evidence:

- `git status --short --untracked-files=all` produced no output
- runtime asset check passed with:
  - `runtime_asset_stage_status=PASS`
  - `assets_checked=20`
  - `assets_staged=0`
  - `missing_required_assets=0`
  - `skipped_optional_assets=0`
  - `check_only=true`
- `docker compose config` completed successfully
- compose showed:
  - `api` built from `Dockerfile.api`
  - `api` exposed on `8000:8000`
  - `frontend` built from `Dockerfile.frontend`
  - `frontend` exposed on `8501:8501`
  - `frontend` depends on `api`
  - API env:
    - `AGENT_ENABLE_LLM=false`
    - `AGENT_DEFAULT_PROVIDER=mock`
    - `LLM_PROVIDER_ORDER=mock,gemini,grok`
    - `LLM_ENABLE_FALLBACK=true`
  - frontend env:
    - `STREAMLIT_API_BASE_URL=http://api:8000`

### Docker Compose Build

Observed command:

```bash
AGENT_ENABLE_LLM=false LLM_ENABLE_FALLBACK=true docker compose build
```

Observed result:

- build finished successfully
- image `industrial-surface-defect-inspection-frontend` built
- image `industrial-surface-defect-inspection-api` built

### Docker Compose Up and ps

Observed command:

```bash
AGENT_ENABLE_LLM=false LLM_ENABLE_FALLBACK=true docker compose up --build -d
docker compose ps
```

Observed result:

- `api` image built
- `frontend` image built
- network created
- `api` container started
- `frontend` container started
- `api` mapped to `0.0.0.0:8000->8000/tcp`
- `frontend` mapped to `0.0.0.0:8501->8501/tcp`
- API health transitioned from starting to healthy

### API Health

Observed command:

```bash
docker compose ps
curl -sS http://localhost:8000/health
curl -sS http://localhost:8000/agent/health
```

Observed result:

- `api` container: up and healthy
- `frontend` container: up
- `/health` returned:
  - `{"status":"ok","service":"industrial-surface-defect-api","api_ready":true,"prediction_ready":false}`
- `/agent/health` returned:
  - `status=ok`
  - `service=industrial-surface-defect-agent`
  - `agent_ready=true`
  - `llm_enabled=false`
  - `default_provider=mock`
  - `provider_order=["mock","gemini","grok"]`
  - `available_providers=["mock"]`
  - `fallback_available=true`
  - `grounding_ready=true`
  - Gemini readiness `status=disabled`
  - `activation_allowed=false`
  - `real_provider_implemented=false`

### Invalid `/agent/explain` Safe Rejection

Observed command:

```bash
POST /agent/explain
page_id=image_inspection
section_id=ai_panel
component_id=image_inspection_ai_panel
```

Observed result:

- API returned `422`
- reason: unsupported `section_id ai_panel` for `image_inspection`
- allowed values were listed

Interpretation:

- safe validation rejection worked
- this is not a runtime failure

### Non-Component `/agent/explain` Mock Response

Observed command:

```bash
POST /agent/explain
question=Explain the current image inspection final decision in a safe way.
page_id=image_inspection
section_id=final_decision
```

Observed result:

- `provider_used=mock`
- `fallback_used=true`
- `grounding_status=insufficient_evidence`
- `page_id=image_inspection`
- `section_id=final_decision`
- `component_id=null`
- answer preserved manual review language
- no production-ready claim
- no deployment-safe claim
- no real LLM provider call

### Component-Aware `/agent/explain` Mock Response

Observed command:

```bash
POST /agent/explain
question=Explain the current image inspection result in a safe way.
page_id=image_inspection
section_id=final_decision
component_id=image_inspection_ai_explanation_panel
```

Observed result:

- `provider_used=mock`
- `fallback_used=true`
- `grounding_status=insufficient_evidence`
- `component_id=image_inspection_ai_explanation_panel`
- `evidence_used` included:
  - `request.page_id`
  - `request.section_id`
  - `request.component_id`
  - `component.user_facing_label`
  - `component.component_type`
- `limitations` included:
  - Mock-only assistant path
  - No external LLM call
  - Manual review still applies
  - No real LLM provider call is made in the MVP mock path
- no invented metric

### Frontend Reachability and Logs

Observed command:

```bash
curl -I -sS http://localhost:8501
docker compose logs --tail=100 api frontend
```

Observed result:

- frontend returned `HTTP/1.1 200 OK`
- frontend started and Streamlit reported local URL `http://localhost:8501`
- API startup completed successfully
- `/health` returned multiple `200 OK` responses
- `/agent/health` returned `200 OK`
- `/agent/explain` returned one expected `422` for invalid `section_id`
- `/agent/explain` returned `200 OK` for valid requests
- no Gemini / Grok / OpenAI call appeared in logs
- no `GEMINI_API_KEY` appeared in logs
- Streamlit printed an external URL, but this validation remains local-only and does not claim public deployment

### Docker Compose Down

Observed command:

```bash
docker compose down
git status --short --untracked-files=all
```

Observed result:

- frontend container removed
- api container removed
- network removed
- git status produced no output

### Post-Validation Test Suite

Observed command:

```bash
pytest tests/docker/ -q
pytest tests/api/test_agent_endpoint.py -q
pytest tests/frontend/ -q
pytest tests/agent/ -q
```

Observed result:

- `tests/docker`: 15 passed
- `tests/api/test_agent_endpoint.py`: 10 passed
- `tests/frontend`: 34 passed
- `tests/agent`: 141 passed

## PASS Criteria Result

PASS:

- runtime assets check passed
- compose config valid
- images built
- containers started
- API healthy
- frontend reachable
- agent health mock-first
- `/agent/explain` mock-first
- component-aware request works
- invalid section rejected safely
- no Gemini / Grok / OpenAI call
- no key required
- logs safe
- containers stopped
- tests passed
- working tree clean

## FAIL / Not Claimed

This evidence does not validate:

- production deployment readiness
- EC2
- HTTPS
- public DNS
- real Gemini / Grok / OpenAI
- production readiness
- runtime LLM activation

## Risk Notes

- `prediction_ready=false` in `/health`, so prediction readiness was not proven by that endpoint
- agent evidence was insufficient for image inspection request because no live `inspection_response` was included in the explain request
- one expected `422` occurred for invalid `section_id` and was interpreted as safe validation behavior
- Streamlit printed an external URL, but this evidence remains local-only and not a public deployment claim

## Final Decision

Status: `PHASE_12_LLM_DISABLED_DOCKER_COMPOSE_VALIDATION_PASS`

Decision:

- PASS for Phase 12 LLM-disabled Docker / Compose local validation
- FAIL / NOT CLAIMED for production deployment readiness
- FAIL / NOT CLAIMED for real LLM readiness

## Recommended Next Slice

The safest next step is to update the roadmap / validation status documents to mark Phase 12 local validation PASS, then continue with EC2 readiness planning while keeping LLM disabled.
