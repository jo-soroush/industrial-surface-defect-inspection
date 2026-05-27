# Pre-Gemini Remaining Gap Audit

## Executive Summary

The repository now has a working mock Agent foundation with component-aware explanations for the highest-priority dashboard surfaces. The foundation is real, test-backed, and evidence-grounded, but it is still not ready for Gemini/Grok/OpenAI integration.

What is confirmed:

- The mock Agent API exists and is healthy.
- Component registry, evidence loader, context builder, and component-aware frontend wiring are implemented for selected components.
- Mock answers are evidence-aware and remain offline.
- Frontend wording has been aligned so it no longer claims that no backend Agent exists.

What is still missing before any Gemini work:

- A Gemini implementation phase approval from the user.

## Current Confirmed State

- `GET /agent/health` reports `agent_ready=True`, `llm_enabled=False`, `default_provider=mock`, `available_providers=["mock"]`, `fallback_available=True`, and `grounding_ready=True`.
- `POST /agent/explain` returns mock responses with `provider_used=mock`, `fallback_used=True`, and grounded evidence for supported component requests.
- The component registry exists and validates allowed component definitions.
- The evidence loader exists and loads compact governed evidence for registry-backed components.
- The context builder supports optional `component_id`.
- The Agent API schema supports optional `component_id`.
- The Agent safety guard module exists and is exercised by the mock provider path.
- The provider contract/readiness layer exists and is used by the provider router.
- The active explainability scope has been formally accepted for the pre-Gemini phase and is intentionally limited to four priority components.
- The formal requirement-to-test matrix now exists and maps the main pre-Gemini requirements to repository tests and validations.
- Phase G1 stub/config implementation is complete.
- Real Gemini provider integration has not started.
- No Gemini API call is implemented.
- No real LLM execution is active.
- Gemini runtime activation is not active.
- Frontend component-aware explanation panels are active for:
  - Image Inspection AI panel
  - Detection confidence chart
  - Classification threshold chart
  - Anomaly threshold behavior chart
- Mock answers remain offline and evidence-aware.
- The safety and AI assistant wording now reflects the current mock Agent state and future LLM boundaries.
- Recent validations passed:
  - `python -m compileall frontend tests api src/inspection_ai scripts`
  - `pytest tests/frontend/ -q`
  - `pytest tests/api/test_agent_endpoint.py -q`
  - `pytest tests/agent/ -q`

## Phase-by-Phase Status

| Phase | Status | Evidence | Remaining gap | Required next action | Blocks Gemini |
|---|---|---|---|---|---|
| Phase 0 - Clean Baseline Confirmation | PASS | Current validation set passes; `git status` was clean at the validated baseline; Agent health and explain endpoints are working in mock mode. | No baseline gap in the current evidence set. | Keep baseline validations green while the remaining gaps are closed. | No |
| Phase 1 - Full Dashboard Component Evidence Audit | PARTIAL | Priority components were audited and wired; the active explainability scope is intentionally limited to four accepted components, while the remaining registry-ready components stay inactive. | A formal, complete coverage audit across every visible dashboard component remains a tracked artifact, but the limited active scope is now accepted. | Keep the component registry coverage audit current if the dashboard scope changes. | No |
| Phase 2 - Universal Component ID Contract | PASS | `configs/agent/component_registry.yaml` defines stable component IDs; the registry validator enforces uniqueness, safety, and allowed values. | Additional components may still be added later, but the contract itself is in place. | Preserve the contract and extend coverage only through the registry. | No |
| Phase 3 - Component Evidence Registry | PARTIAL | The registry exists and is validated by `src/inspection_ai/agent/component_registry.py` with tests. | Registry coverage for the whole dashboard is documented, and the remaining registry-ready components are intentionally inactive by scope decision. | Keep the registry coverage audit current if new components are added. | No |
| Phase 4 - Evidence Loader Layer | PASS | `src/inspection_ai/agent/evidence_loader.py` loads compact, traceable evidence for governed files and runtime inspection data. | Raw evidence remains intentionally blocked by registry defaults. | Keep the loader offline-only and allowlisted. | No |
| Phase 5 - Context Builder Upgrade | PASS | `build_grounding_context` accepts `component_id` and loads component evidence into grounding context. | Full dashboard use still depends on registry coverage completeness. | Keep component-aware context behavior stable. | No |
| Phase 6 - Agent API Schema Upgrade | PASS | `AgentExplainRequest` and `AgentExplainResponse` support optional `component_id`; invalid component IDs return validation errors. | No schema gap in the current implementation. | Preserve backward compatibility for non-component requests. | No |
| Phase 7 - Frontend Component Wiring | PARTIAL | Frontend wiring exists for the active priority components, and that limited scope is accepted for the pre-Gemini phase. | Only the priority components are wired; the rest are intentionally inactive and not considered missing. | Add more component wiring only through a separate approved expansion. | No |
| Phase 8 - Mock Answer Upgrade Based on Evidence | PASS | Mock answers are evidence-aware and component-specific for the active surfaces. | Future richer language output is still limited to mock behavior. | Keep mock answers offline and bounded by evidence. | No |
| Phase 9 - Safety Guard Layer Before LLM | PASS | `src/inspection_ai/agent/safety_guard.py` provides deterministic pre- and post-generation checks, and the provider router exercises the guard on the mock Agent path. | Future provider-specific policy wiring will still need to reuse the same guard contract, but the formal safety layer now exists. | Preserve the guard contract and keep expanding its tests before any real provider work. | No |
| Phase 10 - Provider Interface, Still No Gemini | PASS | `src/inspection_ai/agent/provider_contracts.py` defines typed provider request/response/status/readiness contracts, and `src/inspection_ai/agent/provider_router.py` uses the readiness snapshot for health and fallback semantics. | Future Gemini/Grok/OpenAI provider code will still need to reuse this contract, but the readiness layer itself now exists and is exercised. | Preserve the contract and keep future provider work behind the same mock-first guardrails. | No |
| Phase 11 - Full Test Matrix Before Gemini | PASS | `docs/agent/pre_gemini_test_matrix.md` now provides a formal requirement-to-test matrix that maps the main pre-Gemini requirements to specific tests and validation commands. | Remaining updates may be needed if new scope is added later, but the matrix now exists. | Keep the matrix current when scope changes. | No |
| Phase 12 - Local Validation Before Gemini | PASS | Python compilation, pytest validations, and the LLM-disabled Docker/Compose smoke validation all pass. | The final smoke path is validated; future changes should re-run it if the deployment surface changes. | Keep the smoke validation current if Docker, Compose, or runtime assets change. | No |

## Detailed Phase Notes

### Phase 0 - Clean Baseline Confirmation

Status: PASS

The repository is currently in a validated state. The mock Agent endpoints work, frontend tests pass, and the current branch has been exercised through compile and pytest checks. This is a clean baseline for remaining gap work.

### Phase 1 - Full Dashboard Component Evidence Audit

Status: PARTIAL

The repository has a formal dashboard component coverage audit, and the active explainability scope is now explicitly accepted for the pre-Gemini phase. The remaining registry-ready components are intentionally inactive rather than missing.

Required next action: keep the coverage audit current if the dashboard scope changes.

### Phase 2 - Universal Component ID Contract

Status: PASS

The component ID contract exists, is enforced, and is stable enough for current component-aware wiring. IDs are validated for shape, uniqueness, and safety, which is the right prerequisite for all later Agent work.

### Phase 3 - Component Evidence Registry

Status: PARTIAL

The registry is present and validated, and the coverage audit documents the accepted active scope. The remaining registry-ready components are intentionally inactive by decision, not absent by accident.

### Phase 4 - Evidence Loader Layer

Status: PASS

The evidence loader is implemented, offline, allowlisted, and traceable. It supports governed files and runtime inspection data without dumping raw JSON by default.

### Phase 5 - Context Builder Upgrade

Status: PASS

The context builder now supports `component_id` and can enrich grounding context with component-specific evidence. This is the key bridge between registry data and the mock Agent response path.

### Phase 6 - Agent API Schema Upgrade

Status: PASS

The Agent request and response contracts now support component-aware requests while preserving backward compatibility for existing non-component calls.

### Phase 7 - Frontend Component Wiring

Status: PARTIAL

The active priority components are wired:

- Image Inspection AI panel
- Detection confidence chart
- Classification threshold chart
- Anomaly threshold behavior chart

The remaining registry-ready components are intentionally not wired yet. That is a deliberate pre-Gemini scope decision, not a missing implementation.

### Phase 8 - Mock Answer Upgrade Based on Evidence

Status: PASS

The mock provider now produces evidence-aware answers that reference selected component labels, evidence values, and safety boundaries. The answers remain concise, offline, and non-LLM.

### Phase 9 - Safety Guard Layer Before LLM

Status: PASS

The repository now has a formal, deterministic safety guard module and the mock Agent path exercises it before and after answer generation. The guard enforces claim-blocking and redaction rules for future provider work without changing the current safe mock behavior.

### Phase 10 - Provider Interface, Still No Gemini

Status: PASS

The repository now has a formal provider contract/readiness layer and the provider router uses it for health and fallback semantics. The mock-first runtime remains intact, and no real provider execution is enabled.

### Phase 11 - Full Test Matrix Before Gemini

Status: PASS

The formal requirement-to-test matrix now exists in `docs/agent/pre_gemini_test_matrix.md`. It maps the main pre-Gemini requirements to concrete repository tests and validation commands, including the remaining PENDING and MANUAL items.

Required next action: keep the matrix current whenever scope changes.

### Phase 12 - Local Validation Before Gemini

Status: PASS

Python compilation, pytest validations, and the LLM-disabled Docker / Compose smoke validation have passed. The local deployment path is now validated in mock-first mode.

Required next action: re-run the smoke if the deployment surface, Dockerfiles, Compose file, or runtime assets change.

## Blocking Items Before Gemini

- User approval to start the Gemini implementation phase.

## Recommended Next Sequence

1. Full Component Registry Coverage Audit
2. Formal Safety Guard Layer
3. Provider Interface Readiness
4. Formal Pre-Gemini Test Matrix
5. User approval to start the Gemini implementation phase
6. Gemini implementation only after the user approves it and the remaining operational gates remain green

## Gemini Must Not Start Until...

- The full component registry coverage audit is complete for the current accepted active scope.
- The safety guard layer has explicit tests.
- Provider readiness and contract rules are documented and validated.
- The pre-Gemini requirement-to-test matrix exists and is current.
- The LLM-disabled Docker / Compose smoke validation has passed.
- The repository still passes the focused compile and pytest checks.
- The project has not introduced any production/deployment claims.

## Maintenance Notes

- Keep the requirement-to-test matrix current if scope changes.
- Re-audit the component coverage if visible dashboard components change.
