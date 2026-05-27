# EC2 LLM-Disabled Readiness Audit

## Executive Summary

This document is EC2 readiness planning only.

It records the next deployment planning step after the Phase 12 local LLM-disabled Docker / Compose validation PASS. It does not run AWS / EC2 commands, does not create or modify AWS resources, and does not claim that EC2 deployment readiness is complete.

Current local validation state:

- Phase 12 local LLM-disabled Docker / Compose validation is PASS.
- API was locally reachable on port `8000`.
- frontend was locally reachable on port `8501`.
- `/agent/health` remained mock-first and LLM disabled.
- `/agent/explain` remained mock-first.
- Gemini, Grok, and OpenAI remained inactive.
- `GEMINI_API_KEY` was not used.

What remains blocked:

- actual EC2 execution
- SSH access validation on EC2
- AWS resource creation or modification
- real LLM activation
- production readiness claims

## Roadmap Alignment

This audit aligns with the next roadmap step after the local Phase 12 PASS:

- Phase 12 local validation: PASS
- Next roadmap step: EC2 readiness planning with LLM disabled
- Real Gemini / Grok / OpenAI activation remains out of scope

This document is planning only and does not claim EC2 deployment readiness is complete.

## EC2 Target Assumptions

Conservative target assumptions for the later EC2 work:

- Ubuntu EC2 instance
- Docker installed on the instance
- Docker Compose plugin installed on the instance
- security group allows only intended ports
- port `8000` may be used for API testing
- port `8501` may be used for frontend testing
- later production-like setup may require Nginx / HTTPS, but not in this task
- no API keys are required for the LLM-disabled mode

## Mac-to-EC2 Platform Risk

The local development machine may be Mac / Apple Silicon or another architecture, while EC2 is likely Linux AMD64.

The early EC2 build strategy must avoid architecture mismatch:

- do not assume a local ARM image will run on EC2
- safest early route may be build directly on EC2
- alternatively build explicitly for `linux/amd64` if that is the target
- confirm the target architecture before any later execution step

## Required EC2 Preconditions Before Actual Deployment

Before any later EC2 deployment step, confirm:

- clean git status
- latest tests passing
- EC2 instance running
- SSH access working
- Docker installed
- Docker Compose plugin installed
- repository copied or cloned
- runtime assets present
- `.env` or equivalent env handling reviewed without secrets committed
- LLM-disabled env confirmed
- no `GEMINI_API_KEY` required
- security group reviewed
- ports `8000` and `8501` planned
- rollback plan documented

## Proposed Future EC2 Validation Command Plan

This command plan is **NOT RUN YET**.

```bash
git status --short --untracked-files=all
docker --version
docker compose version
python scripts/runtime_assets/stage_runtime_assets.py --manifest configs/runtime_assets/manifest.yaml --check
docker compose config
AGENT_ENABLE_LLM=false LLM_ENABLE_FALLBACK=true docker compose build
AGENT_ENABLE_LLM=false LLM_ENABLE_FALLBACK=true docker compose up -d
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/agent/health
curl -sS -X POST http://localhost:8000/agent/explain ...
curl http://localhost:8501
docker compose logs --tail=100 api frontend
docker compose down
pytest tests/docker/ -q
pytest tests/api/test_agent_endpoint.py -q
pytest tests/frontend/ -q
pytest tests/agent/ -q
git status --short --untracked-files=all
```

## PASS Criteria for Future Actual EC2 Validation

Future actual EC2 validation passes only if:

- SSH access works
- Docker and Compose are available
- compose config is valid
- build succeeds on EC2
- containers start
- API health passes
- agent health reports LLM disabled and mock-first
- `/agent/explain` returns mock / fallback-safe output
- frontend is reachable
- no real LLM call occurs
- no key is required
- logs are safe
- containers stop cleanly
- working tree is clean

## FAIL Criteria

Future EC2 validation fails if:

- SSH is unavailable
- Docker / Compose are missing
- architecture mismatch occurs
- build fails
- containers fail to start
- API health fails
- frontend is unreachable
- missing `GEMINI_API_KEY` breaks startup
- Gemini / Grok / OpenAI are attempted
- logs expose secrets
- security group exposes unintended ports
- working tree changes unexpectedly

## Evidence to Collect Later

When actual EC2 validation is later approved, collect:

- EC2 instance / environment summary without secrets
- Docker / Compose versions
- command outputs
- health summaries
- agent health summary
- `/agent/explain` summary
- frontend reachability
- logs summary
- no-secret confirmation
- PASS / FAIL decision
- cleanup / shutdown proof

## Security and Safety Boundaries

- do not commit secrets
- do not put real API keys in compose
- do not expose unnecessary ports
- do not claim production readiness
- do not claim HTTPS unless configured and tested
- do not activate Gemini / Grok / OpenAI
- keep mock-first fallback

## Rollback / Cleanup Plan

- `docker compose down`
- stop the EC2 instance if needed
- remove test containers / images if needed
- keep git clean
- revert only deployment-specific files if any are later changed

## Final Decision

Status: `READY_FOR_USER_APPROVAL_TO_PLAN_OR_RUN_EC2_LLM_DISABLED_VALIDATION`

Decision:

- PASS for EC2 readiness planning
- FAIL / PENDING for actual EC2 validation

This document does not claim production readiness, EC2 readiness completion, or real LLM readiness.

## Recommended Next Slice

Review this EC2 readiness plan, then run the EC2 preflight only after explicit user approval.

Do not run EC2 commands in this task.
