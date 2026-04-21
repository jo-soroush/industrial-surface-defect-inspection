# Industrial Surface Defect Inspection Platform

## Project Overview

Industrial Surface Defect Inspection Platform is a production-oriented AI inspection platform for industrial image analysis. It is defined as an industrial surface inspection system that combines known defect classification, anomaly-aware evaluation, decision-support, and controlled human-review routing within a governed inspection workflow.

The system is designed for operational quality-control environments in which inspection outputs must be usable, reviewable, and traceable. It is AWS-deployable, intended for disciplined deployment and governance practices, and explicitly positioned as a decision-support system. It does not claim full automation.

## Mission Statement

To provide a production-oriented industrial surface inspection system that combines classification, anomaly detection, decision-support, and human-review routing within an AWS-deployable AI inspection platform for governed industrial inspection workflows.

## System Identity

Industrial Surface Defect Inspection Platform is an AI inspection platform and an industrial surface inspection system. It is production-oriented, AWS-deployable, and defined as a decision-support system for industrial quality workflows. The system is not notebook-only, not research-only, and not fully autonomous.

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

Anomaly detection adds value by identifying cases that may not align with known defect categories and therefore require stronger scrutiny than standard classification output can provide. Human review integration adds value by ensuring that ambiguous, high-risk, or distribution-shifted cases are handled through defined oversight instead of silent automation. The resulting value is operational: more controlled inspection decisions, clearer escalation behavior, and better alignment between model output and real inspection workflow needs.

## Human-in-the-Loop Principle

Human-in-the-loop operation is a system design principle. The inspection workflow is intentionally structured so that model outputs inform decisions while defined classes of cases are routed to qualified human review under explicit operational criteria.

Human review serves an operational control role within the system. It is part of the intended architecture for handling ambiguity, elevated risk, distribution shift, and decision accountability. Human involvement is not a fallback patch for weak model performance; it is a planned control layer in the inspection process.

## Core System Capabilities

The system is defined to support the following core capabilities:

- Known defect classification for defined industrial defect categories
- Anomaly-aware inspection support for suspicious, novel, or out-of-distribution cases
- Confidence-aware decision-support for separating routine outputs from review-required outputs
- Action routing into bounded operational decisions such as `accept`, `manual_review`, `high_risk`, and `uncertain`
- AWS-deployable, production-oriented system design suitable for governed deployment and operational control

## Scope Boundaries

The system scope is intentionally bounded. Its behavior and claims are limited by the documented operating context, data characteristics, model behavior, deployment assumptions, review policy, and evidence standards under which it is evaluated and used. System outputs must be interpreted within those constraints and not as general guarantees.

All capabilities are governed by documented risk and limitation controls. Use of the system must remain aligned with the defined intended use, explicit non-intended use boundaries, human-in-the-loop policy, evidence requirements, and documented limitations. Any use outside those controls is outside the defined scope of the system.

## Repository Documentation Map

The following documents define the project foundation for this repository:

- `docs/system_overview.md`
- `docs/problem_definition.md`
- `docs/intended_use.md`
- `docs/non_intended_use.md`
- `docs/risk_and_limitations.md`
- `docs/hitl_policy.md`
- `docs/evidence_policy.md`
