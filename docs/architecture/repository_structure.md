# Repository Structure

## Repository Structure Overview

This repository is organized as a production-oriented AI system codebase for the Industrial Surface Defect Inspection Platform. The structure separates governance documentation, reusable Python packages, API serving concerns, configuration ownership, data boundaries, governed artifacts, exploratory workflows, operational assets, test coverage, and frontend concerns so that repository responsibilities remain explicit and reviewable.

The repository structure is part of system governance. Directory placement is not incidental. It is used to make ownership visible, reduce ambiguity, and prevent critical logic from being distributed across uncontrolled locations.

## Top-Level Repository Directories and Responsibilities

- `docs/` contains system-definition, governance, architecture, ML, API, deployment, operations, runbook, and product-facing documentation.
- `src/` contains the reusable Python package structure for core backend system code.
- `api/` contains the serving-layer package structure and API-facing application bootstrap.
- `configs/` contains controlled configuration grouped by domain.
- `data/` contains bounded local data structure for raw, interim, processed, external, and manifest-oriented data organization.
- `artifacts/` contains governed outputs such as datasets, models, evaluations, calibration outputs, explainability outputs, decisions, benchmarks, failures, and reports.
- `notebooks/` contains exploratory notebooks, experiment-support notebooks, and report-support notebooks.
- `scripts/` contains repeatable operational and development scripts by domain.
- `tests/` contains repository test structure for unit, integration, API, security, resilience, and fixture support.
- `ops/` contains deployment and operational support structure such as container, cloud, CI, monitoring, and release ownership areas.
- `frontend/` contains the repository’s frontend-oriented application, components, services, and frontend test structure.

## Source Package Structure Under `src/inspection_ai/`

The source package under `src/inspection_ai/` is the canonical backend package root. It is organized into explicit ownership areas:

- `common` for shared cross-cutting primitives that do not belong to a narrower domain package
- `config` for reusable configuration-handling code
- `data` for reusable data-access and data-processing support code
- `features` for reusable feature-related logic
- `models` for reusable model-related abstractions and support code
- `training` for training ownership
- `evaluation` for evaluation ownership
- `calibration` for calibration ownership
- `explainability` for explainability ownership
- `anomaly` for anomaly-related ownership
- `decision` for decision-support and routing-related ownership
- `inference` for inference ownership
- `security` for security-related support code
- `observability` for logging, metrics, tracing, and operational visibility support code
- `governance` for governance-aware support code and control-oriented utilities
- `services` for reusable service-layer coordination logic that does not belong in the API package
- `utils` for bounded helper utilities that do not warrant a narrower domain package

## Ownership Boundaries

Ownership boundaries are explicit at the repository level:

- Governance and definition documents belong in `docs/`.
- Reusable backend code belongs in `src/inspection_ai/`.
- Serving ownership belongs in `api/`.
- Configuration ownership belongs in `configs/`.
- Local data organization belongs in `data/`.
- Generated outputs and evidence-oriented artifacts belong in `artifacts/`.
- Exploratory and analysis workflows belong in `notebooks/`.
- Repeatable command-oriented workflows belong in `scripts/`.
- Verification assets belong in `tests/`.
- Operational support assets belong in `ops/`.
- Frontend ownership belongs in `frontend/`.

These boundaries are structural controls. They reduce ambiguity about where code, configuration, outputs, and documentation should reside.

## Prohibited Mixed Placements

The following mixed placements are prohibited at the repository-structure level:

- Business logic must not live in notebooks only.
- Generated artifacts must not be stored in source or API directories.
- Training logic must not live in `api/`.
- Frontend code must not be mixed into backend directories.
- Configs must not be scattered informally across unrelated directories.

These restrictions exist to preserve reviewability, governance, packaging clarity, and operational control.

## Notebook Restriction

Notebooks are allowed for exploration, experiments, and analysis support. They are not the canonical home of critical project logic.

Any important logic must be moved to `src/` or `scripts/`. Notebooks should consume reusable code instead of duplicating business logic. If logic becomes important to training, inference, evaluation, decision handling, or other controlled system behavior, that logic must not remain notebook-only.

## Training-Serving Separation

Training-serving separation is enforced structurally:

- Training ownership is in `src/inspection_ai/training/`.
- Inference ownership is in `src/inspection_ai/inference/`.
- Serving ownership is in `api/`.

This separation makes it explicit that model development, runtime inference behavior, and API serving concerns are not the same responsibility. Training logic must remain outside the API package. API structure must remain focused on serving concerns rather than model-development ownership.
