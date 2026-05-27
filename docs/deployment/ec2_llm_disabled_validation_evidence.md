# EC2 LLM-Disabled Validation Evidence

## Executive Summary

This document records actual EC2 LLM-disabled validation evidence.

The validation was performed on a live EC2 instance, but the deployment remained mock-first and LLM-disabled throughout. No Gemini, Grok, or OpenAI API call was made. `GEMINI_API_KEY` was not used, read, printed, validated, or inspected. Real LLM runtime remained inactive.

This evidence is local to the EC2 validation run and does not claim production readiness, HTTPS/domain readiness, or real LLM readiness.

Final decision:

- PASS for EC2 LLM-disabled validation
- FAIL / NOT CLAIMED for production readiness
- FAIL / NOT CLAIMED for HTTPS/domain readiness
- FAIL / NOT CLAIMED for real LLM readiness

Status: `EC2_LLM_DISABLED_VALIDATION_PASS`

## EC2 Environment

Observed EC2 instance details during validation:

- Instance name: `industrial-surface-defect-demo`
- Instance ID: `i-07d344312ea58b4b2`
- Instance type: `t3.small`
- Availability Zone: `eu-north-1a`
- Public IPv4 used during validation: `13.62.224.174`
- Public DNS used during validation: `ec2-13-62-224-174.eu-north-1.compute.amazonaws.com`
- Security group: `industrial-surface-defect-sg`
- Key pair name: `industrial-surface-defect-key`
- OS: Ubuntu 24.04.4 LTS
- Architecture: `x86_64`
- Disk: about 29G root volume, about 27G initially available
- RAM: about 1.9GiB

The instance was stopped, not terminated, after validation completed.

## Tooling Setup

Preflight observations on EC2:

- SSH login worked with the `ubuntu` user.
- `whoami` returned `ubuntu`.
- `hostname` returned `ip-172-31-21-138`.
- `uname` reported a Linux x86_64 AWS kernel.
- `lsb_release` reported Ubuntu 24.04.4 LTS.
- `git` was installed: `git version 2.43.0`.
- `python3` was installed: `Python 3.12.3`.
- `docker` was not installed initially.

Manual setup performed by the user:

- `docker.io` was installed.
- `docker-compose-v2` was installed.
- Docker service was enabled.
- The `ubuntu` user was added to the `docker` group.
- After re-login, `docker --version` returned `Docker version 29.1.3`.
- `docker compose version` returned `Docker Compose version 2.40.3`.
- `docker ps` worked without `sudo`.

## Project Transfer and Extraction

The repository was not transferred from GitHub during this run. A local tarball package was created and copied to EC2 instead.

Transfer details:

- Large local folders were excluded from the tarball:
  - `.venv`
  - `.git`
  - `data`
  - `artifacts`
  - `notebooks`
  - caches
- `runtime_assets` was kept.
- Final package size: 47M
- EC2 transfer path: `/home/ubuntu/industrial-surface-defect-inspection-ec2.tar.gz`
- Extracted repository path: `/home/ubuntu/industrial-surface-defect-inspection`
- Required files were present after extraction, including:
  - `Dockerfile.api`
  - `Dockerfile.frontend`
  - `docker-compose.yml`
  - `requirements-api.txt`
  - `requirements-frontend.txt`
  - `api/app/main.py`
  - `runtime_assets`
- `runtime_assets` size was about 53M.
- macOS metadata files such as `.DS_Store` and `._*` were removed.

## Runtime Assets and Compose Config

Runtime asset checks on EC2:

- `python3 scripts/runtime_assets/stage_runtime_assets.py --manifest configs/runtime_assets/manifest.yaml --check` returned FAIL because full source artifact/data directories were intentionally excluded from the lightweight EC2 package.
- Manual runtime asset checks passed:
  - `track_a=OK`
  - `track_b=OK`
  - `frontend_detection=OK`
  - `track_a_checkpoint=OK`
  - `track_b_checkpoint=OK`
  - `yolo_best=OK`
  - `detection_predictions=OK`
  - `dataset_yaml=OK`

Relevant Dockerfile behavior:

- `Dockerfile.api` copies `runtime_assets/artifacts/` to `./artifacts/`
- `Dockerfile.api` copies `runtime_assets/configs/` to `./configs/`
- `Dockerfile.api` copies `runtime_assets/data/` to `./data/`
- `Dockerfile.frontend` copies `runtime_assets/artifacts/frontend/` to `./artifacts/frontend/`

Compose configuration passed and showed the expected LLM-disabled environment:

- `AGENT_ENABLE_LLM=false`
- `AGENT_DEFAULT_PROVIDER=mock`
- `LLM_ENABLE_FALLBACK=true`
- `LLM_PROVIDER_ORDER=mock,gemini,grok`
- `STREAMLIT_API_BASE_URL=http://api:8000`

## EC2 Docker Build and Run

The EC2 build and startup sequence succeeded with LLM disabled.

Commands run by the user:

- `AGENT_ENABLE_LLM=false LLM_ENABLE_FALLBACK=true docker compose build`
- `AGENT_ENABLE_LLM=false LLM_ENABLE_FALLBACK=true docker compose up -d`
- `docker compose ps`

Observed result:

- API image built successfully on EC2.
- frontend image built successfully on EC2.
- network created.
- API container started.
- frontend container started.
- API mapped to `0.0.0.0:8000->8000/tcp`.
- frontend mapped to `0.0.0.0:8501->8501/tcp`.
- API health progressed from starting to healthy.

## Health and Agent Validation

Local EC2 health checks:

- `curl -sS http://localhost:8000/health`
- `curl -sS http://localhost:8000/agent/health`

Observed `/health` response:

```json
{"status":"ok","service":"industrial-surface-defect-api","api_ready":true,"prediction_ready":false}
```

Observed `/agent/health` state:

- status: ok
- agent_ready: true
- llm_enabled: false
- default_provider: mock
- provider_order: `["mock","gemini","grok"]`
- available_providers: `["mock"]`
- fallback_available: true
- grounding_ready: true
- Gemini unavailable / disabled
- api_key_present: false
- activation_allowed: false
- real_provider_implemented: false

Local EC2 `/agent/explain` validation:

- question: `Explain the current image inspection result in a safe way.`
- page_id: `image_inspection`
- section_id: `final_decision`
- component_id: `image_inspection_ai_explanation_panel`

Observed result:

- `provider_used=mock`
- `fallback_used=true`
- `grounding_status=insufficient_evidence`
- limitations included:
  - Mock-only assistant path
  - No external LLM call
  - Manual review still applies
  - No real LLM provider call is made in the MVP mock path
- no production-ready claim
- no deployment-safe claim
- no invented metrics

Public validation from the user's machine:

- `/health` at `http://13.62.224.174:8000/health` returned status ok.
- `/agent/health` at `http://13.62.224.174:8000/agent/health` returned:
  - `llm_enabled=false`
  - `default_provider=mock`
  - `available_providers=["mock"]`
  - `api_key_present=false`
  - `activation_allowed=false`
  - `real_provider_implemented=false`
- `/agent/explain` returned mock-first output with `provider_used=mock` and `fallback_used=true`.

## Frontend Validation

Local frontend validation:

- `curl -I -sS http://localhost:8501`
- browser open to the public frontend URL

Observed result:

- frontend returned HTTP/1.1 200 OK
- Streamlit dashboard loaded successfully
- the dashboard displayed the Mock Agent active / external providers not connected
- validation remained HTTP-only and local to the EC2 instance

## Logs and Resource Checks

The user captured logs and resource snapshots after validation.

Observed results:

- API logs showed startup complete.
- API logs showed multiple `GET /health 200 OK`.
- API logs showed `GET /agent/health 200 OK`.
- API logs showed `POST /agent/explain 200 OK`.
- frontend logs showed the Streamlit app was available.
- frontend logs showed an external URL for the EC2 host.
- no Gemini, Grok, or OpenAI call appeared in logs.
- no `GEMINI_API_KEY` appeared in logs.
- two `Invalid HTTP request received` lines appeared in frontend logs, but the frontend still returned HTTP 200 and loaded successfully in the browser.

Resource snapshot:

- `/dev/root`: 29G total, 18G used, 11G available
- memory: 1.9Gi total, about 972Mi available, no swap
- image sizes:
  - API image about 9.08GB disk usage
  - frontend image about 1.04GB disk usage

## Restart / Persistence Setup

The Docker service was enabled and both containers were updated to restart unless stopped:

- `docker update --restart unless-stopped industrial-surface-defect-inspection-api-1 industrial-surface-defect-inspection-frontend-1`

At that point both containers reported `restart=unless-stopped`.

The user then chose to stop the demo and keep the host state ready for later work.

## Cleanup / Current State

The user stopped the stack with:

- `docker compose down`

Observed result:

- frontend container removed
- API container removed
- network removed

The EC2 instance was then stopped in the AWS console, not terminated.

Current post-validation state:

- EC2 state: Stopped
- files and setup remain on disk
- public IP may change on next start because no Elastic IP was configured

## PASS / FAIL Result

PASS:

- EC2 SSH access worked
- Ubuntu / x86_64 was verified
- Docker and Docker Compose were installed and validated
- the project tarball was transferred successfully
- runtime assets were present for Docker use
- compose config was valid
- build succeeded on EC2
- containers started on EC2
- API health passed
- frontend was reachable
- public API was reachable
- `/agent/health` remained mock-first and LLM disabled
- `/agent/explain` remained mock-first
- component-aware mock explain worked
- invalid section ID failed safely
- no Gemini / Grok / OpenAI call occurred
- no key was required
- logs were safe
- containers stopped cleanly
- the working tree remained clean at the recorded checkpoints

FAIL / NOT CLAIMED:

- production readiness
- HTTPS readiness
- domain readiness
- real LLM readiness
- Gemini activation
- provider routing activation
- EC2 deployment completion beyond this local validation

Status: `EC2_LLM_DISABLED_VALIDATION_PASS`

## Not Claimed

This evidence does not claim:

- production readiness
- deployment-safe status
- HTTPS readiness
- domain readiness
- real Gemini, Grok, or OpenAI readiness
- real LLM runtime activation
- public endpoint hardening
- Elastic IP usage
- Nginx / reverse proxy readiness
- automatic restart after `docker compose down`

## Recommended Next Slice

Keep the EC2 instance stopped until the next work session.

Continue Gemini / RAG / LLM completion locally before any public presentation.

When the next deployment cycle begins, refresh the EC2 package and images only after the local Gemini / RAG / LLM layer is complete.

Do not present this as production-ready until those later layers are implemented, validated, and explicitly reviewed.
