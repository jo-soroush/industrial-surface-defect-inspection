# Industrial Surface Defect Inspection Platform

## Project Overview

Industrial Surface Defect Inspection Platform is a YOLO-focused industrial surface defect inspection demo/platform for industrial image analysis. Its main inspection story is defect detection and localization with YOLO, supported by governed classification and anomaly evidence, a deterministic decision layer, and Gemini-gated explanations where explicitly enabled.

The system is designed for quality-control workflows in which inspection outputs must be usable, reviewable, and traceable. It is positioned as a decision-support system, not an autonomous inspector, and the current demo deployment has been validated on EC2 with Docker Compose and browser checks.

## Mission Statement

To provide a governed industrial surface defect inspection platform centered on YOLO-based defect detection and localization, with supporting classification and anomaly evidence, review-only explanations, and manual-review routing for industrial inspection workflows.

## System Identity

Industrial Surface Defect Inspection Platform is an AI inspection platform and an industrial surface inspection system. It is a decision-support system for industrial quality workflows, with YOLO as the primary defect detection/localization focus and classification/anomaly tracks providing supporting evidence. Gemini-gated explanations are evidence-grounded and review-only when explicitly enabled. The system is not fully autonomous.

Validated demo status: EC2 Gemini-gated demo validation passed on the public Docker Compose deployment, with API and frontend containers running successfully, `/agent/health` reporting Gemini readiness, and `/agent/explain` returning grounded Gemini-backed explanations in the validated runtime.

## Problem Definition Summary

Industrial surface inspection is operationally important because product-surface quality affects downstream manufacturing reliability, release decisions, scrap rates, rework cost, customer acceptance, and escalation burden. Inspection outputs must therefore support consistent decisions under throughput pressure while preserving traceability for audit, review, and process control.

Classification alone is insufficient because real industrial inspection streams do not contain only previously seen, cleanly labeled defect categories. Operating conditions change, image characteristics shift, novel failure patterns emerge, and ambiguous cases occur. A closed-set classifier can still return a label when the input does not belong to any known class, which creates false certainty and weakens operational control.

Anomaly detection is therefore a required system function, not an optional enhancement. It is needed to identify suspicious or distribution-shifted cases that cannot be handled safely through classification alone. Human review is also a required part of the workflow because inspection decisions can carry operational and quality consequences that require contextual judgment, exception handling, and accountable oversight beyond model output alone.

## Intended Use

This system is intended for industrial inspection workflows in which surface images are evaluated to support operator and quality-review decisions. Its intended use includes known defect classification for defined defect categories, anomaly scoring for suspicious cases, and generation of bounded decision outputs such as `accept`, `manual_review`, `high_risk`, and `uncertain`.

The system is intended to support human operators by converting model outputs into inspection-oriented decision-support rather than raw prediction alone. It is intended for triage, review prioritization, and escalation handling within governed industrial quality processes where human oversight remains part of the operating model.

## Non-Intended Use

This system is not authorized to replace human decision-making in inspection, release, rejection, or escalation workflows. It is not defined as universally applicable across all industrial domains, materials, sensors, imaging conditions, surface types, or production processes without domain-specific validation, documented controls, and bounded operating assumptions.

The system does not guarantee detection of unknown defects. Anomaly-related outputs must not be interpreted as evidence that all novel or rare failure modes will be identified. Confidence scores are not a substitute for review judgment and must not be trusted blindly. Explainability outputs are supporting artifacts only and must not be trusted blindly as independent grounds for operational acceptance, rejection, or escalation. The system is not intended for safety-critical use or medical use.

## Value Proposition

The system provides decision-support rather than prediction in isolation. In operational inspection settings, useful outputs must support routing, uncertainty handling, review prioritization, and exception management. A prediction-only system is insufficient because it leaves action handling undefined and does not establish how ambiguous or suspicious cases should be governed.

YOLO-based defect detection and localization is the core value driver, with classification and anomaly detection adding supporting evidence for review and escalation. Human review integration adds value by ensuring that ambiguous, high-risk, or distribution-shifted cases are handled through defined oversight instead of silent automation. Gemini-gated explanations add value by turning governed evidence into reviewable summaries without changing model outputs or removing manual review.

The resulting value is operational: more controlled inspection decisions, clearer escalation behavior, and better alignment between model output and real inspection workflow needs.

## Repository Structure Overview

This repository is being organized as a production-oriented AI system codebase. At the current stage, the Phase 0 governance documents are present in `docs/`, and the repository is being prepared for Phase 1 implementation structure.

The intended root-level structure includes the following major areas:

- `docs/` for system definition, governance, risk, and policy documents
- `src/` for the core Python package and application logic
- `api/` for the serving layer and API entrypoints
- `configs/` for versioned configuration and environment-specific settings
- `data/` for controlled local data structure and dataset references
- `artifacts/` for governed model and evaluation outputs
- `notebooks/` for exploratory and analysis notebooks under policy controls
- `scripts/` for operational and development scripts
- `tests/` for test coverage across package, API, and workflow behavior
- `ops/` for deployment, container, and operational support assets
- `frontend/` for any inspection-facing interface layer if included in later phases

These directories are part of the intended repository structure. Their mention here does not imply that all of them already exist in the current repository state.

## Basic Setup

For local Python development:

1. Create a Python virtual environment in the repository root.
2. Install runtime dependencies from `requirements.txt`.
3. Install development dependencies from `requirements-dev.txt` when working on tests, linting, or local quality checks.
4. Copy `.env.example` into a local environment file only when local runtime configuration is needed.

For the validated local Docker Compose runtime, use the steps below.

## Local Docker Usage

The project has been validated locally on macOS with Docker Desktop and Docker Compose. The default local mode intentionally disables Gemini real execution and uses mock/fallback explanations. Manual review still applies.

From the parent directory of the repository:

```bash
cd industrial-surface-defect-inspection

AGENT_ENABLE_LLM=false \
AGENT_ENABLE_REAL_PROVIDER_RUNTIME=false \
LLM_ENABLE_FALLBACK=true \
docker compose build

AGENT_ENABLE_LLM=false \
AGENT_ENABLE_REAL_PROVIDER_RUNTIME=false \
LLM_ENABLE_FALLBACK=true \
docker compose up -d

docker compose ps

curl -sS http://127.0.0.1:8000/health | python3 -m json.tool
```

After startup:

- Local dashboard: `http://localhost:8501`
- Local API health: `http://localhost:8000/health`
- `/health` returns `status=ok` and `api_ready=true`
- `/agent/health` returns `llm_enabled=false`, `default_provider=mock`, `available_providers=["mock"]`, `fallback_available=true`, and `grounding_ready=true`

This local default behavior is expected. Gemini real execution is enabled only when explicitly configured. It requires a local `.env` file containing `GEMINI_API_KEY`. Never commit `.env` or API keys.

The validated Gemini-enabled public EC2 demo remains available at `http://13.60.218.168:8501`.

## Human-in-the-Loop Principle

Human-in-the-loop operation is a system design principle. The inspection workflow is intentionally structured so that model outputs inform decisions while defined classes of cases are routed to qualified human review under explicit operational criteria.

Human review serves an operational control role within the system. It is part of the intended architecture for handling ambiguity, elevated risk, distribution shift, and decision accountability. Human involvement is not a fallback patch for weak model performance; it is a planned control layer in the inspection process.

## Core System Capabilities

The system is defined to support the following core capabilities:

- YOLO-based defect detection and localization as the main inspection focus
- Known defect classification for defined industrial defect categories as supporting evidence
- Anomaly-aware inspection support for suspicious, novel, or out-of-distribution cases as supporting evidence
- Confidence-aware decision-support for separating routine outputs from review-required outputs
- Gemini-gated evidence-grounded explanations for selected components when explicitly enabled
- Action routing into bounded operational decisions such as `accept`, `manual_review`, `high_risk`, and `uncertain`
- EC2/Docker Compose demo deployment validation for the gated explanation path

## Scope Boundaries

The system scope is intentionally bounded. Its behavior and claims are limited by the documented operating context, data characteristics, model behavior, deployment assumptions, review policy, and evidence standards under which it is evaluated and used. System outputs must be interpreted within those constraints and not as general guarantees.

All capabilities are governed by documented risk and limitation controls. Use of the system must remain aligned with the defined intended use, explicit non-intended use boundaries, human-in-the-loop policy, evidence requirements, and documented limitations. Any use outside those controls is outside the defined scope of the system.

## Responsible-Use Statement

This repository defines a decision-support system for governed industrial inspection workflows. It must not be interpreted or deployed as a fully autonomous inspection authority. Model outputs, anomaly-related signals, confidence-related signals, and explainability artifacts are bounded decision-support signals and must not be treated as ground truth or as a substitute for human judgment.

Use of the system remains subject to documented intended use, explicit non-intended use boundaries, risk and limitation controls, human-in-the-loop policy, and evidence requirements. Any use that bypasses those controls is outside the defined scope of the system.

## Validated Demo Status

### Local Docker Compose Validation

- macOS Docker Desktop startup: PASS
- Local Docker Compose build for `api` and `frontend`: PASS
- Local Docker Compose startup for both containers: PASS
- API container is healthy
- Frontend container is running
- `/health` returns `status=ok` and `api_ready=true`
- `/agent/health` reports local default mock/fallback mode: `llm_enabled=false`, `default_provider=mock`, `available_providers=["mock"]`, `fallback_available=true`, and `grounding_ready=true`

### EC2 Gemini-Enabled Demo Validation

- EC2 Gemini-gated demo validation: PASS
- Public EC2 Streamlit URL validated during testing: `http://13.60.218.168:8501`
- API and frontend containers run through Docker Compose on EC2
- API container is healthy
- Frontend container is running
- Docker restart policy is set to `unless-stopped`
- Docker service restart recovery was validated
- `/health` returns `status=ok` and `api_ready=true`
- `/agent/health` returns Gemini-enabled runtime readiness with mock fallback available
- `/agent/explain` returns grounded Gemini-backed explanations in the validated runtime
- Browser validation confirmed Gemini-backed explanations for Image Inspection, Surface Defect Classification, Surface Anomaly Detection, and the Detection confidence chart

Local Docker and EC2 Docker are demo/runtime validation only, not production readiness. They must not be interpreted as deployment-safe or factory-ready. Manual review remains required. Safe mock fallback remains available.

## Repository Documentation Map

The following documents define the project foundation for this repository:

- `docs/system_overview.md`
- `docs/problem_definition.md`
- `docs/intended_use.md`
- `docs/non_intended_use.md`
- `docs/risk_and_limitations.md`
- `docs/hitl_policy.md`
- `docs/evidence_policy.md`
