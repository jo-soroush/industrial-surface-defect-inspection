# System Overview

Industrial Surface Defect Inspection Platform is a production-oriented AI inspection platform for industrial image analysis. It is defined as an industrial surface inspection system that supports governed inspection workflows through known defect classification, anomaly detection, confidence-aware interpretation, explainability support signals, and controlled human-review routing. The system is designed to produce bounded inspection actions that are operationally usable, reviewable, and traceable.

This system is defined as a decision-support system. It does not claim full automation and is not intended to operate as an ungoverned prediction service. Its outputs are meant to support industrial inspection workflows in which model signals are interpreted through explicit decision logic and human oversight.

## Mission Statement

To provide a production-oriented industrial surface inspection system that combines classification, anomaly detection, decision-support, and human-review routing within an AWS-deployable AI inspection platform for governed industrial inspection workflows.

## System Identity

Industrial Surface Defect Inspection Platform is:

- An AI inspection platform
- An industrial surface inspection system
- Production-oriented
- A decision-support system
- AWS-deployable

Industrial Surface Defect Inspection Platform is not:

- Notebook-only
- Research-only
- Fully autonomous

The system is defined for disciplined industrial inspection use, with explicit operating boundaries, documented governance expectations, and review-aware decision handling.

## Scope Note

Current governed data tracks are MVTec for classification, MVTec for anomaly detection, and GC10-DET for object detection. The raw GC10-DET payload is present in the repository. Detection remains part of the active governed scope, but the Phase 2 governance layer is not yet fully aligned across manifests, configs, and reports. NEU and NEU-DET are outside the current executable scope.

## Operational Problem Context

Industrial surface inspection matters operationally because surface quality affects product release decisions, downstream manufacturing reliability, scrap rates, rework burden, customer acceptance, and quality escalation handling. Inspection outputs are not isolated predictions; they influence operational actions that can carry cost, quality, and accountability consequences.

Classification alone is insufficient for this setting. Real inspection inputs do not remain limited to cleanly labeled, previously observed defect categories. Imaging conditions shift, process conditions change, unfamiliar defect patterns emerge, and ambiguous cases appear. A classifier can still produce a label under these conditions even when the label is not operationally reliable.

Anomaly detection is needed because industrial inspection workflows must handle suspicious, unfamiliar, or distribution-shifted inputs that cannot be governed safely through closed-set classification alone. Human review must remain part of the workflow because certain inspection outcomes require contextual judgment, exception handling, and accountable oversight that should not be delegated solely to model outputs.

## System Operating Model

The system operating model combines classification, anomaly detection, confidence-aware interpretation, and human-review routing within a bounded inspection workflow. Model signals are not treated as self-executing decisions. They are interpreted together to support governed action selection.

The system is therefore designed as an operational decision-support workflow, not as autonomous automation. Classification contributes information about known classes. Anomaly detection contributes information about unfamiliarity or distribution shift. Confidence contributes a model-side signal about output strength. Explainability contributes inspection support artifacts for analysis and review. Human-review routing determines when manual oversight is required as part of the designed operating path.

## Value Proposition

The system provides decision-support rather than raw prediction in isolation. In industrial inspection workflows, outputs must support action, routing, and governance rather than simply returning a predicted label. A prediction-only system is insufficient because it does not define how uncertainty, ambiguity, or unfamiliar cases should be handled within an operational process.

Anomaly detection adds operational value by identifying inputs that may not align with known defect categories. This reduces reliance on closed-set classification and supports stronger handling of unfamiliar or distribution-shifted cases that require additional scrutiny.

Human-in-the-loop integration adds value by ensuring that ambiguous, high-risk, or policy-sensitive cases are routed to qualified human review rather than being handled through unsupported automation. This preserves accountability and aligns system behavior with real inspection governance requirements.

The resulting value of the system is operational. It enables structured inspection handling, clearer escalation behavior, controlled decision routing, and improved alignment between model outputs and real-world quality-control workflows.

## Classification Role

Classification is responsible for handling known classes that are explicitly defined within the intended inspection context. Its role is to estimate which known defect category, if any, best matches a given industrial surface image under the constraints of the trained label space.

Classification is not the final decision. A predicted class label is one input into downstream decision logic, where it is interpreted together with anomaly signals, confidence signals, and review-routing rules. Classification output therefore feeds decision logic; it does not replace it.

## Anomaly Detection Role

Anomaly detection is responsible for identifying suspicious, unfamiliar, or distribution-shifted patterns that may not fit the known classification space. Its role is to support decision-making by highlighting cases that require stronger scrutiny than classification output alone would justify.

Anomaly detection is a support mechanism for inspection governance, not a guarantee of unknown-defect discovery. It is not guaranteed to detect all unknown defects, all novel patterns, or all rare failure modes. Its outputs must be interpreted as bounded risk signals within the broader decision-support workflow.

## Confidence Role

Confidence is a model signal that contributes to decision interpretation. It may be derived from classification behavior, scoring behavior, or related model-output characteristics, but it remains a model-generated signal rather than direct evidence of correctness.

Confidence is not certainty. High-confidence errors are possible, especially under distribution shift, class overlap, data imbalance, or weak calibration. Confidence values therefore require later calibration and policy-based interpretation before they can support governed operational use. Confidence must not be used as a standalone basis for unconditional acceptance or rejection.

## Explainability Role

Explainability supports debugging, analysis, model inspection, and human review by providing artifacts that help users inspect which regions or features may have influenced model behavior. These artifacts can assist failure analysis, review conversations, and investigation of unexpected outputs.

Explainability is not proof. Attention maps, saliency views, and similar visual signals do not establish causal correctness and do not constitute ground truth. Explainability outputs are support signals only and must not be treated as independent evidence that a decision is correct.

## Human Review Role

Human review is part of the designed workflow. It is not an exception path added only when the system fails. The operating model assumes that certain cases will require manual assessment under defined routing criteria.

Model outputs are not always final. Manual review is expected in cases involving elevated anomaly indicators, uncertainty, conflicting signals, ambiguous visual evidence, or other policy-defined escalation triggers. Human review serves an operational control function by providing accountable oversight where automated interpretation alone is not sufficient.

## Decision Output Model

The system produces bounded decision outputs such as `accept`, `manual_review`, `high_risk`, and `uncertain`. These outputs represent governed operational actions rather than raw model predictions.

Decision outputs are derived from combined system signals, including classification results, anomaly-related indicators, confidence-related interpretation, and review-routing logic. They are intended to express how the inspection workflow should proceed, not merely what the model predicted. This distinction is central to the system definition: the platform is designed to support action governance, not prediction display alone.

## Execution Control Rules

The following execution control rules apply to the system definition and implementation baseline:

- Critical inspection logic must not exist only in notebooks. Any logic required for training, inference, routing, evaluation, or operational decision handling must be implemented in governed code paths that are versioned, reviewable, and reproducible.
- Configuration must be separated from application code. Thresholds, routing rules, environment settings, model references, and deployment parameters must be externally managed through explicit configuration mechanisms rather than embedded as untracked operational constants.
- Artifacts must be governed. Models, datasets, evaluation outputs, threshold records, and decision-policy artifacts must be identifiable, versioned where applicable, and traceable to their source conditions and evidence basis.
- Training and serving concerns must remain separated. Experimental training workflows, model development processes, and production-serving behavior must not be coupled in a way that obscures reproducibility, deployment control, or auditability.
- Claims require evidence. No performance, reliability, robustness, explainability, or operational suitability claim may be treated as valid without supporting evidence that is documented, reviewable, and aligned with the project evidence policy.

## Scope Alignment Notes

This document aligns with `README.md` and is intended to remain consistent with the project definition established there. It also aligns with the repository documents governing intended use, non-intended use, risk and limitations, human-in-the-loop policy, and evidence policy.

No part of this system definition should be interpreted to overclaim capability. Statements in this document are bounded by documented system scope, documented operating assumptions, and documented governance controls. Where ambiguity exists, the narrower interpretation governs.
