# Non-Intended Use

## Non-Intended Use Definition

This document defines uses that are outside the authorized scope of Industrial Surface Defect Inspection Platform. These exclusions are governance boundaries. They are binding limits on how the system is to be interpreted, deployed, and used.

Any use that conflicts with this document is outside the defined system scope. Such use must not be represented as consistent with the intended design, intended operating model, or authorized capability of the system.

## Prohibited Operational Uses

The following operational uses are prohibited:

- Full replacement of human decision-making in inspection, acceptance, rejection, release, escalation, or quality-control workflows
- Operation of the system as a fully autonomous inspection authority
- Blind acceptance or blind rejection based on model output alone
- Use of system outputs without review workflows, control workflows, or documented operating rules

The system must not be used as a sole decision source for product disposition or quality escalation. It is not authorized to convert model outputs directly into unreviewed operational action where governance controls are absent.

## Model Trust Limitations

Confidence scores are not guarantees of correctness. They are model-generated signals and must not be interpreted as certainty, proof, or authorization for unconditional operational action.

Anomaly detection is not complete. It does not guarantee identification of all unknown defects, all novel failure modes, or all distribution-shifted cases. Its outputs must not be treated as complete coverage of unfamiliar conditions.

Explainability outputs are not proof. Attention maps, saliency views, and related artifacts do not establish correctness, causality, or ground truth. They are support signals only.

Blind trust in confidence signals, anomaly signals, or explainability outputs is prohibited. No individual model-derived signal may be treated as sufficient on its own to justify acceptance, rejection, escalation, or assurance of correctness.

## Domain and Deployment Restrictions

The system has no claim of universal applicability. It must not be represented as valid across all industrial sectors, materials, imaging setups, surface types, defect regimes, production conditions, or inspection workflows.

Transfer of the system across domains is prohibited unless domain-specific validation has been completed and documented. This includes transfer across new materials, new sensors, new imaging conditions, new defect taxonomies, new process environments, or new operational decision policies.

Uncontrolled deployment is prohibited. The system must not be deployed without documented operating assumptions, documented thresholds or routing logic, documented review controls, and documented governance of model and decision artifacts. Domain-specific validation is required for any deployment claim.

## Safety and Critical Use Exclusions

The system must not be used in safety-critical systems. It must not be used in medical applications. It must not be used in any context where an incorrect decision can directly cause harm without human oversight.

The system is not authorized for autonomous use in environments where inspection error could directly produce unacceptable safety, health, or harm outcomes. Any such use is outside the defined limits of the system.

## Misinterpretation Risks

Use of the system carries a risk of false certainty. A model may produce a confident-looking output even when the input is ambiguous, shifted, unfamiliar, or poorly represented by the training distribution.

Use of the system carries a risk of distribution shift. Changes in data source, imaging conditions, materials, process state, or defect patterns can degrade reliability without being immediately obvious from output form alone.

Use of the system carries a risk of misclassification. Known-class prediction can be wrong even under apparently routine operating conditions, and incorrect labels can be produced for unfamiliar inputs.

Use of the system also carries a risk that outputs will be misused as ground truth. This is prohibited. System outputs are governed decision-support signals, not authoritative truth statements about product condition.

## Alignment Notes

This document is aligned with `README.md`, `docs/system_overview.md`, and `docs/intended_use.md`. It is intended to remain consistent with the repository-wide system definition, operating model, and bounded-use requirements established in those documents.

This document is binding in defining system limits. It must be read together with intended use, risk and limitations, human-in-the-loop policy, and evidence policy. Where interpretation is uncertain, the stricter boundary governs.
