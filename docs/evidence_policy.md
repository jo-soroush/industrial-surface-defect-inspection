# Evidence Policy

## Evidence Policy Definition

This document defines the evidence requirements for Industrial Surface Defect Inspection Platform. Evidence is mandatory for all claims made about the system, including but not limited to claims regarding performance, reliability, robustness, suitability, deployment readiness, decision logic behavior, and operational use.

No claim is valid without supporting artifacts. A statement that cannot be linked to identifiable, reviewable, and retrievable evidence is not an accepted system claim and must not be presented as fact within project documentation, evaluation records, deployment decisions, or operational representations.

## Evidence Principles

The evidence policy is governed by the following principles:

- Reproducibility
- Traceability
- Verifiability
- Consistency

Evidence must be reviewable and repeatable. A reviewer must be able to determine what was evaluated, under which conditions it was evaluated, which artifacts were produced, and whether the result can be reproduced from the documented inputs, configuration, and execution context.

Evidence must also remain internally consistent. Reported results, associated datasets, associated model versions, associated thresholds, and associated decision logic must not conflict with one another.

## Evidence Categories

The following evidence categories are required:

- Model performance evidence
- Dataset evidence
- Evaluation evidence
- Decision logic evidence
- Deployment evidence
- Testing evidence

## Required Artifacts

Acceptable evidence must include:

- Evaluation reports
- Metrics outputs
- Confusion matrices
- Dataset manifests
- Model metadata
- Logs
- Test results

Artifacts must be:
- stored
- identifiable
- retrievable

## Claim-to-Evidence Mapping

Every claim must map directly to evidence.

Each claim must identify:
- dataset
- model version
- configuration
- evaluation conditions
- outputs

No implicit claims are allowed.

## Evidence Validation Rules

Evidence is valid only if:

- consistent with system state
- reproducible
- not outdated
- complete

Evidence must NOT be reused across different system states without revalidation.

## Evidence Traceability

All results must be traceable to:
- dataset
- model
- config
- run

Untraceable results are invalid.

## Non-Acceptable Evidence

Not allowed:

- screenshots without context
- unverifiable claims
- anecdotal results
- missing metadata
- partial results

## Alignment Notes

This policy is binding and governs all system claims.
