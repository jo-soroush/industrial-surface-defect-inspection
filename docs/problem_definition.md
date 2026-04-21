# Problem Definition

## Problem Statement

The problem addressed by this system is industrial surface inspection in operational environments where image-based quality decisions must be made with consistency, scalability, and traceability. Industrial inspection workflows require more than isolated model predictions. They require controlled handling of known defects, structured treatment of suspicious or unfamiliar cases, and explicit decision paths that can be reviewed and justified.

The operational challenge is to support inspection decisions without collapsing the workflow into unsupported automation. Surface inspection outputs influence release decisions, rejection decisions, rework handling, escalation activity, and quality accountability. A production-oriented inspection system must therefore provide bounded, reviewable, and governable outputs rather than raw classification alone.

## Operational Context

Industrial surface inspection operates under throughput pressure. Inspection decisions must be produced at a pace compatible with manufacturing or quality-control flow without reducing the integrity of review. Delayed decisions increase queue buildup, operational friction, and downstream process disruption.

Industrial surface inspection also operates under strict quality-control requirements. Decisions affect acceptance, rejection, re-inspection, and escalation pathways. Wrong decisions impose direct operational cost through scrap, rework, defective product release, unnecessary manual review, and unstable quality performance.

The cost of wrong decisions is not limited to immediate production loss. Incorrect inspection outcomes can distort defect statistics, weaken root-cause analysis, and undermine confidence in the inspection process. For that reason, auditability is required. Inspection-relevant outputs must be traceable to documented logic, documented signals, and documented workflow rules so that decisions can be reviewed, challenged, and governed.

## Limitations of Classification-Only Systems

Classification-only systems operate under a closed-set assumption. They assign inputs to a predefined set of known classes and therefore depend on the premise that the input space is adequately represented by those classes. Industrial inspection environments do not satisfy that premise reliably.

Unknown defects, novel failure patterns, process drift, imaging changes, and ambiguous inputs are all expected realities in industrial operation. A classification-only system cannot represent true uncertainty about the absence of a suitable known class with sufficient operational rigor. It can still return a label when the input does not belong to any valid known category.

This creates a risk of false certainty. A predicted class may appear operationally decisive even when the underlying input is unfamiliar, shifted, or outside the intended model scope. For that reason, classification alone is insufficient for governed industrial inspection. It does not provide enough control over unfamiliar cases, does not define escalation behavior, and does not protect the workflow from unsupported certainty.

## Need for Anomaly Detection

Anomaly detection is required because the inspection workflow must identify unfamiliar or distribution-shifted inputs that cannot be handled safely through known-class prediction alone. Its role is to surface suspicious cases that warrant stronger scrutiny, stricter routing, or mandatory review.

Anomaly-related signals support identification of inputs that differ materially from expected patterns, including cases associated with novel defects, unusual texture behavior, unexpected image conditions, or operational drift. This function is necessary to reduce dependence on closed-set assumptions and to strengthen handling of inspection cases that fall outside routine behavior.

Anomaly detection is not guaranteed to detect all unknown defects. It does not provide complete coverage of all rare, novel, or shifted conditions. Its role is to contribute a bounded risk signal within the broader decision-support workflow, not to serve as proof that all unknown failure modes will be identified.

## Need for Decision Support

Prediction alone is not enough for operational inspection. Industrial workflows require governed actions, not isolated scores or labels. A system that returns only class predictions leaves routing, review thresholds, escalation handling, and ambiguity treatment undefined.

Decision support is required to translate model signals into structured outputs such as `accept`, `manual_review`, `high_risk`, and `uncertain`. These outputs are necessary because inspection operations depend on knowing how the workflow should proceed, not merely what the model predicted.

Operational routing logic is therefore a required part of the problem definition. Classification signals, anomaly signals, and confidence-related interpretation must be combined under explicit rules so that the system produces bounded actions suitable for review, escalation, and process control.

## Need for Human-in-the-Loop

Human review is required for accountability and risk handling. Inspection decisions can affect product disposition, process quality, escalation burden, and audit outcomes. Those consequences require accountable oversight that cannot be delegated entirely to model output.

Model outputs cannot be final in all cases. Certain cases must be routed to manual review because they involve ambiguity, elevated anomaly indicators, conflicting signals, policy-defined risk conditions, or insufficient decision confidence. A governed inspection system must therefore preserve a formal role for human judgment within the operating workflow.

Human-in-the-loop is a designed requirement, not a fallback. It is part of the intended control architecture of the system and a necessary mechanism for handling cases where automated interpretation alone is not sufficient for accountable operational use.

## Problem Scope Boundaries

This problem definition applies to bounded industrial contexts in which surface-image inspection is performed under documented operating assumptions, documented workflow rules, and documented governance controls. It does not claim universal applicability across all industrial domains, materials, sensors, production conditions, or inspection regimes.

The system does not guarantee perfect detection, perfect classification, perfect anomaly identification, or complete coverage of unknown defects. It does not define model outputs as absolute truth. Outputs produced by the system are governed signals that support operational decisions under controlled interpretation.

This problem definition aligns with the repository-wide system definition in `README.md` and `docs/system_overview.md`. It is intended to remain consistent with intended use, non-intended use, risk and limitations, human-in-the-loop policy, and evidence policy. No statement in this document should be interpreted as an overclaim of capability.
