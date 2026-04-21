# Human-in-the-Loop Policy

## HITL Policy Definition

Human-in-the-loop is a designed system behavior of Industrial Surface Defect Inspection Platform. It is a required part of the system operating model and a binding condition of system use.

Human review is not optional and not a fallback. It is not an auxiliary step added only when the system appears unreliable. It is a planned control layer embedded in the inspection workflow to support governed decision-making, risk handling, and accountable operational use.

## Role of Human Review

Human review is part of the normal workflow of the system. The system is designed on the assumption that model outputs will be used within an inspection process that includes human assessment under defined conditions.

The role of human review is broader than error correction. Human review exists to handle ambiguity, elevated risk, unfamiliar patterns, exception cases, policy-governed escalation, and inspection contexts where model-generated signals are not sufficient for accountable action on their own.

Human review therefore serves as an operational control mechanism. It supports disciplined interpretation of system outputs within real inspection workflows rather than post hoc correction of isolated model mistakes.

## Decision Authority Model

Model outputs are not always final. The system produces decision-support signals that inform workflow handling, but final operational authority is retained by human operators or reviewers in defined cases.

Human operators retain decision authority wherever the workflow requires manual review, risk-based escalation, or policy-governed confirmation. This includes cases routed by anomaly-related signals, confidence-related interpretation, conflicting signals, or other documented review triggers.

The system must not be interpreted as having full inspection authority. It is not authorized to act as a fully autonomous inspection decision-maker.

## Review Trigger Conditions

Human review is expected under defined trigger conditions that include the following:

- High anomaly signals
- Low confidence or ambiguous confidence-related interpretation
- Conflicting signals across classification, anomaly, confidence, or policy logic
- Policy-defined risk conditions

Additional review triggers may be defined by documented operating policy, but all such triggers must remain explicit, reviewable, and governed. Routing to review under these conditions is part of normal system behavior and must not be treated as exceptional workflow failure.

## Workflow Integration

Model outputs and human review are integrated by design. Classification results, anomaly-related interpretation, confidence-related interpretation, and bounded decision outputs interact to determine how an inspection case proceeds through the workflow.

Routing to review is part of system design. Outputs such as `manual_review`, `high_risk`, or `uncertain` exist to guide workflow behavior toward additional oversight when defined conditions are met. These outputs guide workflow actions; they do not replace the workflow itself.

The operating model therefore requires explicit interaction between model-generated signals and human review activity. The system is designed to support controlled progression through inspection stages rather than direct autonomous disposition.

## Accountability and Oversight

Human review provides accountability within the inspection process. It ensures that certain decisions remain subject to qualified human judgment, documented oversight, and policy-based handling rather than unsupported automation.

System outputs must be reviewable and explainable at the workflow level. This does not mean that model internals become self-justifying; it means the basis for routing, review, and action handling must be inspectable through documented signals, documented logic, and documented workflow controls.

Traceability is required. Review-triggering conditions, relevant decision-support outputs, and applicable workflow actions must be capable of being traced to documented system behavior and documented operational policy.

## Prohibited Interpretations

The following interpretations are prohibited:

- Interpreting model outputs as final in all cases
- Removing human review from the workflow where this policy requires it
- Treating the system as autonomous

It is also prohibited to redefine human review as a purely optional convenience step, to treat review routing as evidence of system failure, or to claim that the presence of model confidence or explainability artifacts eliminates the need for human oversight.

## Alignment Notes

This policy is aligned with `README.md`, `docs/system_overview.md`, `docs/intended_use.md`, and `docs/non_intended_use.md`. It is intended to remain consistent with the repository-wide system definition, intended operating model, and bounded-use controls established in those documents.

This HITL policy is binding for system operation. Any deployment, interpretation, or workflow design that conflicts with this policy is outside the defined scope of the system.
