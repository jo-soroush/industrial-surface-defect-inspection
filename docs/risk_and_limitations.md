# Risk and Limitations

## Risk and Limitations Overview

Industrial Surface Defect Inspection Platform is a bounded system and is not perfect. Its outputs are decision-support signals produced within defined operating assumptions, not authoritative truth statements about product condition, defect presence, or inspection correctness.

This document defines the principal limitations and risk factors that constrain how system outputs may be interpreted and used. These limitations are part of the system boundary. Any interpretation that ignores them is outside the defined scope of the system.

## Dataset Limitations

System performance depends on dataset characteristics. If the data used for development, evaluation, or operation does not adequately represent the operating domain, the resulting system behavior may be unreliable even when outputs appear plausible.

Dataset bias is a material risk. Bias can arise from selective sampling, uneven imaging conditions, narrow process coverage, restricted material coverage, or label practices that do not reflect real operating variation. Such bias can distort both classification behavior and anomaly-related behavior.

Coverage of defect types may be limited. Known defect categories may not fully represent the defect landscape encountered in operation, and unknown or weakly represented defect patterns may not be handled reliably. Class imbalance is also a material limitation because underrepresented classes may receive weaker model treatment and less reliable output behavior.

Dataset quality is a direct dependency. Label noise, inconsistent labeling rules, image corruption, low-quality captures, and incomplete metadata can degrade performance and weaken the reliability of both development conclusions and operational outputs.

## Model and Classification Limitations

Misclassification is a core system risk. The model may assign an incorrect known defect label to an input, including cases where the input appears superficially similar to a known class or where class boundaries are weak.

The classification function is subject to a closed-set limitation. It operates over defined known classes and does not inherently guarantee recognition that an input belongs outside the label space. This means classification outputs can appear valid even when the input is unfamiliar or outside the intended operating distribution.

The model is sensitive to distribution shift. Changes in imaging conditions, material characteristics, process state, defect presentation, camera setup, preprocessing behavior, or other domain factors can degrade output reliability. Correct-looking outputs can still be wrong, and plausible class predictions must not be treated as proof of correctness.

## Anomaly Detection Limitations

Anomaly detection provides incomplete coverage of unknown defects. It does not guarantee identification of every unfamiliar pattern, every rare failure mode, or every distribution-shifted condition.

Anomaly-related behavior is sensitive to threshold selection. Threshold settings can materially change routing outcomes and can produce either excessive review burden or insufficient escalation. Poor threshold selection can distort the intended inspection workflow.

False positives and false negatives are expected risks. Some routine cases may be flagged unnecessarily, and some suspicious or unfamiliar cases may not be surfaced at the required level. Anomaly detection is not guaranteed and must not be treated as complete protection against unknown conditions.

## Confidence Limitations

Confidence does not equal certainty. A confidence-related output is a model-side signal and does not constitute proof that the underlying classification or routing outcome is correct.

Confidence requires calibration before it can support governed interpretation in operational settings. Without calibration and policy-based use rules, confidence values can be misleading or unstable across classes, conditions, or data sources.

High-confidence errors are possible. A model may produce a strong confidence signal for an incorrect prediction, particularly under class overlap, dataset imbalance, ambiguous inputs, or distribution shift. Confidence therefore cannot justify blind trust in the output.

## Explainability Limitations

Explainability outputs such as attention maps, saliency views, or related artifacts are not proof of correctness. They do not establish causal validity and do not convert a model output into verified ground truth.

Explanations are not ground truth. They are interpretive support artifacts that may assist analysis, debugging, and review, but they do not independently validate the underlying decision-support output.

There is a material risk of misinterpretation. Visual explanation artifacts can be overread, treated as stronger evidence than they are, or mistaken for direct evidence of correct reasoning. Such use is outside the intended interpretation of the system.

## Operational Risks

Use of the system in uncontrolled workflows is a material operational risk. If outputs are consumed without defined review logic, defined escalation paths, or documented operating policy, the system can be misused as an unsupported decision authority.

Lack of proper review processes creates direct governance failure. Where manual review criteria, oversight responsibilities, or escalation procedures are missing or weakened, the system may be used in ways that exceed its intended scope.

Incorrect configuration of thresholds, routing logic, or domain-specific settings can materially alter system behavior. Misconfigured policies can create false acceptance, unnecessary escalation, unstable review burden, or misleading interpretations of model output.

Deployment without validation is prohibited and presents a direct risk. Use in a new domain, new workflow, new sensor setup, new defect regime, or new process environment without documented validation can produce unreliable behavior while creating a false impression of readiness.

## Over-Reliance Warning

System outputs must not be trusted blindly. Classification results, anomaly-related outputs, confidence-related outputs, and explainability artifacts are all bounded signals subject to model error, dataset limitation, configuration error, and operational misuse.

Human review and governance controls are required to manage these risks. The system must be used within documented review processes, documented routing logic, documented operating assumptions, and documented evidence standards. Over-reliance on model outputs is outside the defined operating boundary of the system.

## Alignment Notes

This document aligns with `README.md`, `docs/system_overview.md`, `docs/problem_definition.md`, `docs/intended_use.md`, `docs/non_intended_use.md`, and `docs/hitl_policy.md`. It is intended to remain consistent with the repository-wide definition of system purpose, use boundaries, and human-review requirements.

This document is part of the system boundary definition. It constrains how capability claims, deployment claims, and operational interpretations may be made. Where any interpretation conflicts with the limitations described here, the stricter limitation governs.
