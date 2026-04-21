# Repository Structure

## Overview

This repository is organized as a production-oriented AI system codebase for the Industrial Surface Defect Inspection Platform. The structure is intended to separate governance documentation, reusable application code, configuration, operational assets, data handling boundaries, and analysis workflows so that the repository remains maintainable, reviewable, and suitable for controlled system development.

## Top-Level Directory Responsibilities

- `docs/` contains system-definition, governance, and architecture documentation.
- `src/` is the intended home for reusable Python package code and shared system logic.
- `api/` is reserved for service-facing API structure and related application entrypoints.
- `configs/` contains controlled configuration by domain.
- `data/` defines bounded local data structure for datasets, intermediate outputs, and manifests.
- `artifacts/` is reserved for governed outputs such as models, evaluations, reports, and related evidence artifacts.
- `notebooks/` is reserved for exploratory work, experiment support, and analysis-oriented reporting.
- `scripts/` is the intended home for repeatable operational or development scripts.
- `tests/` is reserved for repository test coverage.
- `ops/` is reserved for deployment and operational support assets.
- `frontend/` is reserved for any interface layer introduced later in the repository lifecycle.

## Notebook Policy

Notebooks are allowed for exploration. They are also allowed for experiments and analysis support where an interactive workflow is useful for inspection, investigation, or reporting.

Notebooks are not the canonical home of critical project logic. Any important logic must be moved to `src/` or `scripts/` so that it can be reviewed, versioned, reused, and governed through standard repository workflows.

Notebooks should consume reusable code instead of duplicating business logic. If logic becomes important to training, inference, evaluation, decision handling, or other controlled system behavior, that logic must not remain notebook-only.
