# Pre-Gemini Remaining Gap Audit

## Executive Summary

The repository now has a working mock Agent foundation with component-aware explanations for the highest-priority dashboard surfaces. The foundation is real, test-backed, and evidence-grounded, but it is still not ready for Gemini/Grok/OpenAI integration.

What is confirmed:

- The mock Agent API exists and is healthy.
- Component registry, evidence loader, context builder, and component-aware frontend wiring are implemented for selected components.
- Mock answers are evidence-aware and remain offline.
- Frontend wording has been aligned so it no longer claims that no backend Agent exists.

What is still missing before any Gemini work:

- Active explainability coverage still needs explicit scope acceptance before Gemini.
- A requirements-to-test matrix that explicitly maps every Gemini blocker to a test.
- A final LLM-disabled Docker/Compose smoke validation after the remaining gaps are closed.

## Current Confirmed State

- `GET /agent/health` reports `agent_ready=True`, `llm_enabled=False`, `default_provider=mock`, `available_providers=["mock"]`, `fallback_available=True`, and `grounding_ready=True`.
- `POST /agent/explain` returns mock responses with `provider_used=mock`, `fallback_used=True`, and grounded evidence for supported component requests.
- The component registry exists and validates allowed component definitions.
- The evidence loader exists and loads compact governed evidence for registry-backed components.
- The context builder supports optional `component_id`.
- The Agent API schema supports optional `component_id`.
- The Agent safety guard module exists and is exercised by the mock provider path.
- The provider contract/readiness layer exists and is used by the provider router.
- The formal requirement-to-test matrix now exists and maps the main pre-Gemini requirements to repository tests and validations.
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
| Phase 1 - Full Dashboard Component Evidence Audit | PARTIAL | Priority components were audited and wired; component-aware explanations exist for the active surfaces. | A formal, complete coverage audit across every visible dashboard component is still needed as a tracked artifact. | Complete the full component registry coverage audit. | Yes |
| Phase 2 - Universal Component ID Contract | PASS | `configs/agent/component_registry.yaml` defines stable component IDs; the registry validator enforces uniqueness, safety, and allowed values. | Additional components may still be added later, but the contract itself is in place. | Preserve the contract and extend coverage only through the registry. | No |
| Phase 3 - Component Evidence Registry | PARTIAL | The registry exists and is validated by `src/inspection_ai/agent/component_registry.py` with tests. | Registry coverage for the whole dashboard still needs explicit audit confirmation. | Finish the coverage audit and record any remaining uncatalogued components. | Yes |
| Phase 4 - Evidence Loader Layer | PASS | `src/inspection_ai/agent/evidence_loader.py` loads compact, traceable evidence for governed files and runtime inspection data. | Raw evidence remains intentionally blocked by registry defaults. | Keep the loader offline-only and allowlisted. | No |
| Phase 5 - Context Builder Upgrade | PASS | `build_grounding_context` accepts `component_id` and loads component evidence into grounding context. | Full dashboard use still depends on registry coverage completeness. | Keep component-aware context behavior stable. | No |
| Phase 6 - Agent API Schema Upgrade | PASS | `AgentExplainRequest` and `AgentExplainResponse` support optional `component_id`; invalid component IDs return validation errors. | No schema gap in the current implementation. | Preserve backward compatibility for non-component requests. | No |
| Phase 7 - Frontend Component Wiring | PARTIAL | Frontend wiring exists for the active priority components. | Only the priority components are wired; the full dashboard is not component-wired. | Wire additional components only after coverage audit and safety guard completion. | Yes |
| Phase 8 - Mock Answer Upgrade Based on Evidence | PASS | Mock answers are evidence-aware and component-specific for the active surfaces. | Future richer language output is still limited to mock behavior. | Keep mock answers offline and bounded by evidence. | No |
| Phase 9 - Safety Guard Layer Before LLM | PASS | `src/inspection_ai/agent/safety_guard.py` provides deterministic pre- and post-generation checks, and the provider router exercises the guard on the mock Agent path. | Future provider-specific policy wiring will still need to reuse the same guard contract, but the formal safety layer now exists. | Preserve the guard contract and keep expanding its tests before any real provider work. | No |
| Phase 10 - Provider Interface, Still No Gemini | PASS | `src/inspection_ai/agent/provider_contracts.py` defines typed provider request/response/status/readiness contracts, and `src/inspection_ai/agent/provider_router.py` uses the readiness snapshot for health and fallback semantics. | Future Gemini/Grok/OpenAI provider code will still need to reuse this contract, but the readiness layer itself now exists and is exercised. | Preserve the contract and keep future provider work behind the same mock-first guardrails. | No |
| Phase 11 - Full Test Matrix Before Gemini | PASS | `docs/agent/pre_gemini_test_matrix.md` now provides a formal requirement-to-test matrix that maps the main pre-Gemini requirements to specific tests and validation commands. | Remaining updates may be needed if new scope is added later, but the matrix now exists. | Keep the matrix current when scope changes. | No |
| Phase 12 - Local Validation Before Gemini | PARTIAL | Python compilation and pytest validations pass. | LLM-disabled Docker/Compose smoke validation is still pending after the remaining pre-Gemini work is completed. | Run the final Docker/Compose validation with LLM disabled before any Gemini plan begins. | Yes |

## Detailed Phase Notes

### Phase 0 - Clean Baseline Confirmation

Status: PASS

The repository is currently in a validated state. The mock Agent endpoints work, frontend tests pass, and the current branch has been exercised through compile and pytest checks. This is a clean baseline for remaining gap work.

### Phase 1 - Full Dashboard Component Evidence Audit

Status: PARTIAL

The priority dashboard surfaces now have component-aware explanations and a registry-backed evidence model. However, the repository still needs a formal audit artifact that proves every visible dashboard component has been reviewed and mapped.

Required next action: complete a dashboard-wide component coverage audit and record the remaining gaps explicitly.

### Phase 2 - Universal Component ID Contract

Status: PASS

The component ID contract exists, is enforced, and is stable enough for current component-aware wiring. IDs are validated for shape, uniqueness, and safety, which is the right prerequisite for all later Agent work.

### Phase 3 - Component Evidence Registry

Status: PARTIAL

The registry is present and validated, but the repository still needs an audit proving that registry coverage matches the full dashboard surface. The contract exists; the coverage proof does not yet.

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

But the full dashboard is not yet component-wired. That is a deliberate partial implementation, not a failure.

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

Status: PARTIAL

Python compilation and pytest validations have passed, which is a strong local signal. The remaining item is the planned Docker/Compose smoke run with LLM disabled after the outstanding pre-Gemini work is complete.

Required next action: run the final Docker/Compose validation with LLM disabled, then record the result.

## Blocking Items Before Gemini

- Active explainability scope is not yet formally accepted for Gemini use.
- Final LLM-disabled Docker/Compose validation is not yet run after the remaining work.
- Gemini provider integration readiness plan is not yet written/reviewed.

## Recommended Next Sequence

1. Full Component Registry Coverage Audit
2. Formal Safety Guard Layer
3. Provider Interface Readiness
4. Formal Pre-Gemini Test Matrix
5. Local Docker/Compose validation with LLM disabled
6. Gemini Provider Integration Readiness Plan
7. Gemini implementation only after all previous items PASS

## Gemini Must Not Start Until...

- The full component registry coverage audit is complete.
- The safety guard layer has explicit tests.
- Provider readiness and contract rules are documented and validated.
- The pre-Gemini requirement-to-test matrix exists and is current.
- Local Docker/Compose validation with LLM disabled has passed after the remaining work.
- The repository still passes the focused compile and pytest checks.
- The project has not introduced any production/deployment claims.
