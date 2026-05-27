# Pre-Gemini Requirement-to-Test Matrix

## Executive Summary

This document maps the current pre-Gemini Agent/RAG foundation requirements to the concrete tests and validation commands that support them.

Current conclusion:

- The core Agent foundation is implemented and well covered by targeted tests.
- The formal test matrix now exists and ties the main pre-Gemini requirements to repository evidence.
- Phase G1 stub/config implementation is complete.
- Phase G2 mocked-client test layer is complete.
- The G3 preparation audit exists and records the package, environment, API-key, activation, rollback, and test strategy.
- The G3 package verification artifact exists and records the verified future `google-genai` dependency decision.
- The G3 entry checklist and first-slice plan exist and define the first implementation boundary.
- The first G3 readiness-scaffolding slice is implemented and tested, while real Gemini provider execution remains inactive.
- The second G3 lazy SDK loader boundary is implemented and tested, while real Gemini provider execution remains inactive.
- The third G3 health/readiness integration slice is implemented and tested, while real Gemini provider execution remains inactive.
- The G3 provider skeleton with mocked / injected SDK object is implemented and tested, while real Gemini provider execution remains inactive.
- The G3 pre-real-call audit exists and defines the final activation gates before any real Gemini implementation.
- The G3 execution gate design exists and defines the disabled-by-default router strategy before any runtime routing change.
- Router activation tests exist and prove Gemini remains mock-first unless all gates pass.
- The G3 local real-smoke plan exists and defines the manual local-only smoke boundary, but it has not been executed.
- The G3 final real-smoke execution checklist exists and defines the mandatory pre-execution review items, but it has not approved execution.
- The G3 dependency decision artifact exists and records the conservative dependency-change decision.
- The backend/API `google-genai` requirements slice exists and is applied without changing runtime behavior.
- Real Gemini provider integration has not started.
- The remaining gap is user approval to start the Gemini implementation phase.
- This matrix does not justify Gemini integration by itself. It is a control document, not an implementation approval.

## Current Validated State

Validated in the repository state referenced by this audit:

- `python -m compileall frontend tests api src/inspection_ai scripts`
- `pytest tests/agent/ -q`
- `pytest tests/api/test_agent_endpoint.py -q`
- `pytest tests/frontend/ -q`
- `git status --short --untracked-files=all` was clean at the validated baseline referenced in the audit history.

## Matrix Status Legend

| Status | Meaning |
|---|---|
| PASS | Requirement is implemented and covered by explicit tests or validation commands. |
| PARTIAL | Requirement is implemented in part, but the coverage or scope proof is not complete. |
| PENDING | Requirement is expected but not yet validated in the current evidence set. |
| MANUAL | Requirement depends on a manual review or a non-automated control. |

## Requirement-to-Test Matrix

### 1. Baseline and Repository Cleanliness

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Python source compiles cleanly | `python -m compileall frontend tests api src/inspection_ai scripts` | PASS | Current validation set passes. | None in the current evidence set. | No |
| Focused pytest suites pass | `pytest tests/agent/ -q`, `pytest tests/api/test_agent_endpoint.py -q`, `pytest tests/frontend/ -q` | PASS | Agent, API, and frontend targeted suites pass. | None in the current evidence set. | No |
| Repository cleanliness is maintained at the validated baseline | `git status --short --untracked-files=all` | PASS | The validated baseline was clean; later work may add uncommitted changes, but the baseline itself was clean. | None for the documented baseline. | No |

### 2. Component Registry Contract

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Component IDs are unique and snake_case | `tests/agent/test_component_registry.py::test_all_component_ids_are_unique`, `::test_all_component_ids_are_snake_case` | PASS | Registry tests enforce uniqueness and ID shape. | None in current tests. | No |
| Registry entries have expected fields and allowed values | `tests/agent/test_component_registry.py::test_registry_loads_successfully`, `::test_required_high_priority_components_exist`, `::test_invalid_registry_examples_fail_validation` | PASS | Loader validation and invalid-registry tests cover shape and required fields. | Additional future components may still be added. | No |
| Raw evidence defaults remain blocked | `tests/agent/test_component_registry.py::test_raw_evidence_is_disabled_for_all_initial_components` | PASS | Initial registry entries keep `raw_allowed=False`. | None in the current evidence set. | No |
| Documented non-explainable components are explicit | `docs/agent/component_registry_coverage_audit.md` plus `tests/frontend/test_ai_assistant_page_copy.py`, `tests/frontend/test_safety_page_copy.py` | PASS | AI Assistant and Safety/Limitations remain fixed boundary copy, not active explainability targets. | Future scope changes would need a new audit. | No |

### 3. Component Registry Coverage

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Coverage audit exists | `docs/agent/component_registry_coverage_audit.md` | PASS | The registry coverage audit was written and reviewed. | None in the audit artifact itself. | No |
| Missing registry entries are zero in the static review | `docs/agent/component_registry_coverage_audit.md` | PASS | The audit found no missing registry entries in the static review. | Future visible components may still require a new review if the UI changes. | No |
| Active explainability coverage is intentionally scope-limited | `docs/agent/component_registry_coverage_audit.md`, `docs/agent/active_explainability_scope_acceptance.md`, `tests/frontend/test_detection_page_polish.py`, `tests/frontend/test_classification_page_polish.py`, `tests/frontend/test_anomaly_page_wiring.py`, `tests/frontend/test_image_inspection_page.py` | PASS | Four priority components are actively wired; the remaining registry-ready components are intentionally inactive by accepted scope decision. | None for the accepted pre-Gemini scope. | No |
| Explicit scope acceptance is documented | `docs/agent/active_explainability_scope_acceptance.md`, `docs/agent/pre_gemini_gap_audit.md` | PASS | The accepted scope decision is now documented and traceable. | Future scope changes need a new approval cycle. | No |

### 4. Evidence Loader

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Governed evidence loads | `tests/agent/test_evidence_loader.py::test_classification_threshold_curve_chart_loads_compact_evidence`, `::test_detection_confidence_chart_loads_yolo_confidence_evidence` | PASS | Governed frontend evidence loads compactly and deterministically. | None in the current evidence set. | No |
| Runtime evidence loads | `tests/agent/test_evidence_loader.py::test_image_inspection_final_decision_card_loads_runtime_evidence` | PASS | Runtime inspection response evidence is supported. | None in the current evidence set. | No |
| Global context evidence loads | `tests/agent/test_evidence_loader.py::test_partial_evidence_uses_global_context_without_fabricating_dashboard_copy` | PASS | Global context is used without inventing dashboard copy. | None in the current evidence set. | No |
| Missing files do not crash | `tests/agent/test_evidence_loader.py::test_missing_governed_evidence_file_does_not_crash` | PASS | Missing governed files are reported as limitations, not exceptions. | None in the current evidence set. | No |
| Missing fields are reported | `tests/agent/test_evidence_loader.py::test_missing_allowed_field_does_not_crash` | PASS | Missing allowlisted fields are surfaced safely. | None in the current evidence set. | No |
| Invalid JSON is handled safely | `tests/agent/test_evidence_loader.py::test_invalid_json_evidence_file_produces_limitation_not_exception` | PASS | Invalid JSON becomes a limitation, not a crash. | None in the current evidence set. | No |
| Raw evidence is blocked by default | `tests/agent/test_evidence_loader.py::test_include_raw_evidence_does_not_include_raw_when_component_disallows_it` | PASS | Raw evidence is not exposed unless explicitly allowed. | None in the current evidence set. | No |
| Path safety is enforced | `tests/agent/test_evidence_loader.py::test_no_evidence_item_source_is_absolute` | PASS | Evidence source paths remain repo-relative or runtime-scoped. | None in the current evidence set. | No |

### 5. Context Builder

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Legacy page/section requests still work | `tests/agent/test_context_builder.py::test_build_grounding_context_extracts_image_inspection_evidence`, `tests/api/test_agent_endpoint.py::test_agent_explain_existing_image_inspection_request_without_component_id_still_works` | PASS | Non-component requests remain compatible. | None in the current evidence set. | No |
| `component_id` requests validate against the registry | `tests/agent/test_context_builder.py::test_component_context_rejects_invalid_component_id` | PASS | Invalid component IDs fail safely. | None in the current evidence set. | No |
| Compact evidence is attached for component requests | `tests/agent/test_context_builder.py::test_component_context_loads_classification_threshold_curve_evidence`, `::test_component_context_loads_detection_confidence_evidence`, `::test_component_context_loads_image_inspection_runtime_evidence` | PASS | Component-aware grounding contains compact evidence. | None in the current evidence set. | No |
| Limitations are preserved | `tests/agent/test_context_builder.py::test_component_context_preserves_anomaly_review_only_limitation`, `tests/agent/test_evidence_loader.py::test_anomaly_threshold_behavior_preserves_review_only_limitation` | PASS | Review-only boundary text survives the context build path. | None in the current evidence set. | No |
| Raw evidence remains blocked | `tests/agent/test_context_builder.py::test_component_context_blocks_raw_evidence_when_component_disallows_it` | PASS | Raw evidence does not flow through component context when disallowed. | None in the current evidence set. | No |

### 6. API Schema and Endpoint

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Optional `component_id` is accepted | `tests/api/test_agent_endpoint.py::test_agent_explain_accepts_classification_component_id`, `::test_agent_explain_accepts_detection_component_id`, `::test_agent_explain_accepts_image_inspection_ai_panel_component_id` | PASS | The endpoint accepts component-aware requests. | None in the current evidence set. | No |
| Old requests without `component_id` still work | `tests/api/test_agent_endpoint.py::test_agent_explain_existing_image_inspection_request_without_component_id_still_works` | PASS | Backward compatibility is preserved. | None in the current evidence set. | No |
| Invalid `component_id` gives a safe error | `tests/api/test_agent_endpoint.py::test_agent_explain_rejects_invalid_component_id_safely` | PASS | Invalid components fail safely and deterministically. | None in the current evidence set. | No |
| `provider_used` remains mock | `tests/api/test_agent_endpoint.py::test_agent_health_reports_mock_only_mvp_state`, `tests/api/test_agent_endpoint.py::test_agent_explain_returns_grounded_mock_answer_for_image_inspection` | PASS | The API remains mock-first. | None in the current evidence set. | No |
| `fallback_used` remains compatible | `tests/api/test_agent_endpoint.py::test_agent_explain_returns_grounded_mock_answer_for_image_inspection`, related mock provider tests | PASS | Fallback semantics remain stable. | None in the current evidence set. | No |
| `grounding_status` is preserved | `tests/api/test_agent_endpoint.py`, `tests/agent/test_provider_router.py` | PASS | Grounding status remains part of the response contract and is exercised in the tests. | No formal contract drift identified. | No |

### 7. Frontend Component Wiring

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Image Inspection component ID is sent | `tests/frontend/test_image_inspection_page.py::test_image_inspection_agent_request_includes_required_fields`, `::test_agent_explanation_status_caption_reports_mock_fallback` | PASS | The Image Inspection panel sends the component-aware request. | None in the current evidence set. | No |
| Detection confidence chart component ID is sent | `tests/frontend/test_detection_page_polish.py::test_detection_confidence_agent_request_is_component_aware` | PASS | Detection chart wiring is active. | None in the current evidence set. | No |
| Classification threshold component ID is sent | `tests/frontend/test_classification_page_polish.py::test_classification_threshold_agent_request_is_component_aware` | PASS | Classification chart wiring is active. | None in the current evidence set. | No |
| Anomaly threshold component ID is sent | `tests/frontend/test_anomaly_page_wiring.py::test_anomaly_threshold_agent_request_is_component_aware` | PASS | Anomaly chart wiring is active. | None in the current evidence set. | No |
| Full-width component Agent panels remain | `tests/frontend/test_detection_page_polish.py::test_shared_component_agent_panel_is_horizontal_and_full_width`, similar classification/anomaly helper tests | PASS | The shared panel structure is present and tested. | None in the current evidence set. | No |
| Stale “no backend agent implemented yet” wording does not return | `tests/frontend/test_frontend_wording.py`, `tests/frontend/test_ai_assistant_page_copy.py`, page-specific polish tests | PASS | Wording tests protect against stale placeholder copy. | Future copy changes should preserve these assertions. | No |
| External LLM not connected wording remains clear | `tests/frontend/test_frontend_wording.py`, `tests/frontend/test_ai_assistant_page_copy.py` | PASS | The page copy reflects the mock/pre-Gemini state. | None in the current evidence set. | No |
| Manual review wording remains clear | `tests/frontend/test_frontend_wording.py`, page-specific panel tests | PASS | Manual review boundaries remain visible. | None in the current evidence set. | No |

### 8. Mock Provider Behavior

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Safe evidence-grounded mock answers pass | `tests/agent/test_provider_router.py::test_mock_provider_grounded_answer_mentions_manual_review`, `::test_non_component_image_inspection_mock_answer_still_works` | PASS | The mock provider returns grounded answers without external LLM calls. | None in the current evidence set. | No |
| Component-aware mock answers use evidence | `tests/agent/test_provider_router.py::test_component_image_inspection_mock_answer_mentions_decision_and_manual_review`, `::test_component_detection_confidence_mock_answer_mentions_confidence_and_review`, `::test_component_anomaly_threshold_mock_answer_mentions_weak_review_only_boundary`, `::test_component_classification_threshold_mock_answer_mentions_validation_threshold` | PASS | Component-specific answers reference the governed evidence surface. | None in the current evidence set. | No |
| Non-component Image Inspection behavior still works | `tests/agent/test_provider_router.py::test_non_component_image_inspection_mock_answer_still_works` | PASS | Legacy non-component path remains compatible. | None in the current evidence set. | No |
| No production/deployment claims are made | `tests/agent/test_provider_router.py::test_component_mock_answers_do_not_claim_readiness_or_provider_integration`, `tests/agent/test_safety_guard.py` | PASS | Guarded outputs block readiness claims. | None in the current evidence set. | No |
| No provider integration claims are made | `tests/agent/test_provider_router.py::test_component_mock_answers_do_not_claim_readiness_or_provider_integration`, `tests/agent/test_safety_guard.py` | PASS | Outputs do not claim Gemini/Grok/OpenAI integration. | None in the current evidence set. | No |

### 9. Safety Guard

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Pre-generation safe compact context passes | `tests/agent/test_safety_guard.py::test_pre_generation_guard_passes_safe_compact_context` | PASS | Safe compact context is allowed. | None in the current evidence set. | No |
| Raw evidence is blocked | `tests/agent/test_safety_guard.py::test_pre_generation_guard_blocks_raw_evidence_when_present` | PASS | Raw evidence is blocked before generation. | None in the current evidence set. | No |
| Secret-like values are redacted | `tests/agent/test_safety_guard.py::test_pre_generation_guard_redacts_secret_like_and_absolute_path_values` | PASS | Secret-like values are redacted deterministically. | None in the current evidence set. | No |
| Local absolute paths are redacted | `tests/agent/test_safety_guard.py::test_pre_generation_guard_redacts_secret_like_and_absolute_path_values` | PASS | Local absolute paths are removed from the prompt context. | None in the current evidence set. | No |
| Manual review and traceability are preserved | `tests/agent/test_safety_guard.py::test_pre_generation_guard_preserves_manual_review_and_traceability` | PASS | Traceability survives sanitization. | None in the current evidence set. | No |
| Production/deployment readiness claims are blocked | `tests/agent/test_safety_guard.py::test_post_generation_guard_blocks_prohibited_claims`, `::test_post_generation_guard_blocks_mixed_readiness_claims` | PASS | Unsafe readiness claims are blocked, including mixed claims. | None in the current evidence set. | No |
| Manual-review replacement claims are blocked | `tests/agent/test_safety_guard.py::test_post_generation_guard_blocks_prohibited_claims` | PASS | Claims that replace manual review are blocked. | None in the current evidence set. | No |
| Autonomous decision claims are blocked | `tests/agent/test_safety_guard.py::test_post_generation_guard_blocks_prohibited_claims` | PASS | Autonomous-final-decision language is blocked. | None in the current evidence set. | No |
| Provider-connected claims are blocked | `tests/agent/test_safety_guard.py::test_post_generation_guard_blocks_mixed_provider_claims`, `::test_post_generation_guard_allows_provider_disabled_statements` | PASS | Disabled-provider statements pass, mixed connected claims are blocked. | None in the current evidence set. | No |
| Invented metric-like values are blocked | `tests/agent/test_safety_guard.py::test_post_generation_guard_blocks_prohibited_claims` | PASS | The guard catches obviously invented metric/threshold claims. | The check is intentionally conservative and deterministic. | No |
| Safe disclaimers such as not production-ready / not deployment-safe pass | `tests/agent/test_safety_guard.py::test_post_generation_guard_allows_not_ready_disclaimers` | PASS | Safe disclaimers remain allowed. | Mixed unsafe claims are still blocked, by design. | No |
| Mixed unsafe claims are blocked | `tests/agent/test_safety_guard.py::test_post_generation_guard_blocks_mixed_readiness_claims`, `::test_post_generation_guard_blocks_mixed_provider_claims` | PASS | Mixed safe/unsafe text is handled conservatively. | None in the current evidence set. | No |
| Safe provider-disabled statements pass | `tests/agent/test_safety_guard.py::test_post_generation_guard_allows_provider_disabled_statements` | PASS | “Not active / not connected” provider statements are allowed when they are cleanly phrased. | None in the current evidence set. | No |
| Safety guard is exercised by `provider_router` | `tests/agent/test_provider_router.py::test_mock_provider_routes_through_safety_guard`, `src/inspection_ai/agent/provider_router.py` | PASS | The mock provider path invokes the safety guard before and after generation. | Future provider-specific integration will still reuse this layer. | No |

### 10. Provider Contract / Readiness

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Provider request / response / status contracts construct correctly | `tests/agent/test_provider_contracts.py::test_provider_contract_objects_can_be_constructed`, `::test_provider_status_objects_can_be_constructed` | PASS | The contract layer is typed, constructible, and deterministic. | None in the current evidence set. | No |
| Raw provider response exposure defaults to false | `tests/agent/test_provider_contracts.py::test_provider_contract_objects_can_be_constructed` | PASS | Raw provider response exposure is off by default. | None in the current evidence set. | No |
| Fallback reason can be represented | `tests/agent/test_provider_contracts.py::test_provider_contract_objects_can_be_constructed`, `src/inspection_ai/agent/provider_contracts.py` | PASS | Fallback reason is explicit in the contract. | None in the current evidence set. | No |
| Mock provider is available | `tests/agent/test_provider_contracts.py::test_provider_readiness_keeps_mock_available_and_real_providers_unavailable_without_keys`, `tests/agent/test_provider_router.py::test_health_reports_mock_first_mvp_state` | PASS | Mock readiness is always available. | None in the current evidence set. | No |
| Gemini/Grok/OpenAI remain unavailable / disabled | `tests/agent/test_provider_contracts.py::test_provider_readiness_keeps_mock_available_and_real_providers_unavailable_without_keys`, `::test_provider_readiness_allows_ready_for_future_use_when_llm_enabled_and_keys_present` | PASS | Future providers are readiness-tracked but do not execute. | None in the current evidence set. | No |
| Readiness does not expose secret values | `tests/agent/test_provider_contracts.py::test_provider_readiness_does_not_expose_secret_values` | PASS | Readiness warnings do not leak keys. | None in the current evidence set. | No |
| No provider SDK/network call is required | `src/inspection_ai/agent/provider_contracts.py`, `src/inspection_ai/agent/provider_router.py`, the test suite | PASS | The contract/readiness layer is local-only and offline. | None in the current evidence set. | No |
| `provider_router` uses readiness for health/fallback semantics | `tests/agent/test_provider_router.py::test_health_reports_mock_first_mvp_state`, `::test_missing_provider_keys_do_not_break_mock_health`, `src/inspection_ai/agent/provider_router.py` | PASS | Health uses the readiness snapshot and still reports mock-only runtime behavior. | None in the current evidence set. | No |
| Response API shape remains compatible | `tests/api/test_agent_endpoint.py`, `tests/agent/test_provider_router.py` | PASS | Existing API tests continue to pass with the same response shape. | None in the current evidence set. | No |

### 10.1 Gemini Provider G1 Stub

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Gemini provider stub can be constructed without SDK or API key | `tests/agent/test_gemini_provider_stub.py::test_gemini_provider_stub_can_be_constructed_without_sdk_or_api_key` | PASS | The stub is constructible and remains offline-only. | None in the current evidence set. | No |
| Gemini stub refuses real generation in G1 | `tests/agent/test_gemini_provider_stub.py::test_gemini_provider_stub_refuses_real_generation` | PASS | The stub raises a safe disabled-state error instead of generating. | None in the current evidence set. | No |
| Gemini stub remains unavailable even with key and LLM enabled | `tests/agent/test_gemini_provider_stub.py::test_gemini_provider_stub_reports_disabled_state_even_with_key_and_llm_enabled`, `::test_gemini_provider_readiness_keeps_gemini_unavailable_for_runtime_settings` | PASS | G1 keeps Gemini unavailable and disabled for real execution. | None in the current evidence set. | No |
| Gemini stub does not import provider SDKs or network libraries | `tests/agent/test_gemini_provider_stub.py::test_gemini_provider_stub_does_not_import_provider_sdks_or_network_libraries`, `::test_gemini_stub_module_text_does_not_reference_provider_sdk_imports` | PASS | Source inspection confirms the stub is offline-only and import-safe. | None in the current evidence set. | No |

### 10.2 Gemini Provider G2 Mocked Client Tests

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Mocked Gemini client success path translates to a safe provider response | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_mocked_client_success_translates_to_gemini_response` | PASS | The offline mocked-client seam can produce a Gemini-shaped response without network access. | None in the current evidence set. | No |
| Mocked timeout falls back safely | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_mocked_client_timeout_falls_back_to_mock` | PASS | Timeout maps to a safe mock fallback result. | None in the current evidence set. | No |
| Mocked provider error falls back safely | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_mocked_client_provider_error_falls_back_to_mock` | PASS | Generic provider errors map to a safe mock fallback result. | None in the current evidence set. | No |
| Mocked rate limit falls back safely | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_mocked_client_rate_limit_falls_back_to_mock` | PASS | Rate-limit handling remains offline and fallback-safe. | None in the current evidence set. | No |
| Empty and malformed mocked responses are handled safely | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_mocked_client_empty_response_falls_back_to_mock`, `::test_gemini_mocked_client_malformed_response_falls_back_to_mock` | PASS | Empty or malformed payloads do not crash the harness. | None in the current evidence set. | No |
| Unsafe and invented metric-like mocked outputs are blocked by the safety guard | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_mocked_client_unsafe_output_is_blocked_by_safety_guard`, `::test_gemini_mocked_client_invented_metric_like_output_is_blocked_by_safety_guard` | PASS | The post-generation guard blocks readiness claims and invented metrics. | None in the current evidence set. | No |
| Secret-like questions are sanitized before mocked-client handling | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_mocked_client_sanitizes_secret_like_questions_before_handling` | PASS | The provider request is sanitized before the mocked client seam handles it. | None in the current evidence set. | No |
| The Gemini provider module stays offline-only | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_provider_module_does_not_import_provider_sdks_or_network_libraries` | PASS | Source inspection confirms the module avoids SDK and network imports. | None in the current evidence set. | No |
| Normal Agent runtime remains mock-first | `tests/agent/test_gemini_provider_mocked_client.py::test_existing_agent_provider_router_normal_mock_path_remains_unchanged` | PASS | The normal `/agent/explain` path still uses the mock provider. | None in the current evidence set. | No |

### 10.3 Gemini Provider G3 Preparation Audit

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| G3 preparation audit exists and documents the real-provider strategy | `docs/agent/gemini_phase_g3_preparation_audit.md` | PASS | The repository now contains a dedicated G3 preparation audit covering package, env, API-key, gating, fallback, Docker/Compose, EC2, observability, rollback, and entry checklist strategy. | Real Gemini provider code is still not started. | No |

### 10.4 Gemini Provider G3 Package Verification

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| G3 package verification exists and records the verified future dependency candidate | `docs/agent/gemini_g3_package_verification.md` | PASS | The repository now contains a dedicated package-verification artifact that selects `google-genai`, requires lazy import, and keeps the mock-first runtime SDK-free. | Exact version pinning remains a G3 implementation decision. | No |

### 10.5 Gemini Provider G3 Entry Checklist

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| G3 entry checklist and first-slice plan exist and define the initial implementation boundary | `docs/agent/gemini_phase_g3_entry_checklist.md` | PASS | The repository now contains a dedicated G3 entry checklist that constrains the first coding slice to readiness scaffolding, SDK-missing behavior, and tests only. | Real Gemini provider code is still not started. | No |

### 10.6 Gemini Provider G3 First Slice Readiness Scaffolding

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| First readiness-scaffolding slice models disabled, missing-key, missing-SDK, and not-implemented states | `tests/agent/test_gemini_g3_readiness.py` | PASS | The repository now models G3 readiness without importing the SDK, while keeping Gemini unavailable and mock fallback safe. | Real Gemini provider code is still not started. | No |

### 10.7 Gemini Provider G3 Lazy SDK Loader Boundary

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Lazy SDK loader boundary exists, is injectable, and remains offline-only | `tests/agent/test_gemini_g3_readiness.py` | PASS | The repository now exposes a test-injectable SDK loader/status boundary without adding a real SDK import or activating Gemini. | Real Gemini provider code is still not started. | No |

### 10.8 Gemini Provider G3 Health / Readiness Integration

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Gemini readiness metadata is threaded into the existing health surface without activating Gemini | `tests/agent/test_provider_router.py::test_health_reports_mock_first_mvp_state`, `::test_missing_provider_keys_do_not_break_mock_health`, `::test_gemini_health_metadata_does_not_expose_raw_key_values`, `tests/api/test_agent_endpoint.py::test_agent_health_reports_mock_only_mvp_state` | PASS | Health and readiness now expose safe Gemini metadata in warnings and router helpers while keeping mock-only runtime behavior and no secret exposure. | Real Gemini provider code is still not started. | No |

### 10.9 Gemini Provider G3 Pre-Real-Call Audit

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Final activation gates are documented before any real Gemini implementation | `docs/agent/gemini_g3_pre_real_call_audit.md` | PASS | The repository now contains a final activation-gate review that defines the package, environment, safety, rollback, testing, and rollout controls before any real Gemini call may exist. | Real Gemini provider code is still not started. | No |

### 10.10 Gemini Provider G3 Dependency Decision

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Conservative dependency installation decision exists and records the applied backend/API requirements slice | `docs/agent/gemini_g3_dependency_decision.md` | PASS | The repository now contains a dependency decision artifact that identifies `google-genai` as the future candidate and records that `requirements-api.txt` has been updated while frontend/dev/top-level requirement files remain unchanged. | Exact version pinning remains pending. | No |

### 10.11 Gemini Provider G3 Backend/API Requirements Slice

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| `google-genai` is added to the backend/API requirements file only | `requirements-api.txt`, `docs/agent/gemini_g3_dependency_decision.md` | PASS | The backend/API dependency file now includes `google-genai`; frontend, dev, and top-level requirement files remain unchanged. | Exact version pinning remains pending. | No |

### 10.12 Gemini Provider G3 Provider Skeleton with Mocked / Injected SDK Object

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Provider skeleton exists for injected test clients only and stays offline-only | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_provider_skeleton_without_injected_client_returns_not_implemented`, `::test_gemini_provider_skeleton_with_fake_safe_client_returns_safe_provider_result`, `::test_gemini_provider_skeleton_blocks_fake_unsafe_output_by_safety_guard`, `::test_gemini_provider_skeleton_blocks_fake_invented_metric_output`, `::test_gemini_provider_skeleton_blocks_fake_path_and_secret_like_output`, `::test_gemini_provider_skeleton_handles_timeout_provider_error_rate_limit_and_empty_or_malformed_outputs` | PASS | The provider skeleton can translate injected test-client behavior into safe provider-style results without real SDK import, network access, or runtime activation. | Real Gemini provider execution is still not started. | No |

### 10.13 Gemini Provider G3 Execution Gate Design

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Disabled-by-default router activation strategy exists and stays documentation-only | `docs/agent/gemini_g3_execution_gate_design.md` | PASS | The repository now contains a dedicated execution gate design that keeps `/agent/explain` mock-first by default and defines the conditions required before any future Gemini routing or runtime activation is even considered. | Real Gemini provider execution is still not started. | No |

### 10.14 Gemini Provider G3 Router Activation Tests

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Router activation tests prove Gemini remains mock-first unless all gates pass | `tests/agent/test_provider_router.py::test_gemini_route_decision_stays_mock_when_llm_disabled_even_with_key`, `::test_gemini_route_decision_stays_mock_when_key_is_missing`, `::test_gemini_route_decision_stays_mock_when_real_provider_is_not_implemented`, `::test_gemini_route_decision_stays_mock_when_sdk_is_available_but_activation_is_disabled` | PASS | The router helper reports mock fallback and disabled-by-default routing when any required gate fails, while the normal `/agent/explain` path remains mock-only. | Real Gemini provider execution is still not started. | No |

### 10.15 Gemini Provider G3 Real Provider Execution Boundary

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Real provider execution exists behind lazy import and injected SDK/client only, while remaining disabled by default | `tests/agent/test_gemini_provider_mocked_client.py::test_gemini_real_provider_without_sdk_import_gate_returns_not_implemented_without_lazy_import`, `::test_gemini_real_provider_missing_key_skips_sdk_loader_and_falls_back`, `::test_gemini_real_provider_sdk_missing_returns_fallback_without_lazy_import`, `::test_gemini_real_provider_with_fake_safe_sdk_client_returns_safe_provider_result`, `::test_gemini_real_provider_blocks_unsafe_or_invalid_fake_sdk_outputs`, `::test_gemini_real_provider_handles_fake_sdk_errors_safely` | PASS | The provider now has a real-execution boundary that only runs with explicit gates and injected fake SDK seams; the default runtime remains mock-first and no real SDK import is exercised by the tests. | Real Gemini provider execution is still not started in normal runtime. | No |

### 10.16 Gemini Provider G3 Local Real-Smoke Plan

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Local real-smoke plan exists and remains planning only | `docs/agent/gemini_g3_local_real_smoke_plan.md` | PASS | The repository now contains a manual, local-only real-smoke plan that defines the future evidence path without approving execution. | Real smoke execution is still blocked until explicit user approval. | No |

### 10.17 Gemini Provider G3 Local Real-Smoke Harness Skeleton

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Local real-smoke harness skeleton exists, stays disabled by default, and does not call Gemini | `scripts/agent/run_gemini_local_smoke.py`, `tests/agent/test_gemini_local_smoke_harness.py` | PASS | The repository now contains a disabled-by-default harness that only prints planned checks in dry-run mode and refuses to run a real smoke in this slice. | Real smoke execution is still blocked until explicit user approval. | No |

### 10.18 Gemini Provider G3 Final Real-Smoke Execution Checklist

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Final real-smoke execution checklist exists and remains documentation only | `docs/agent/gemini_g3_final_real_smoke_execution_checklist.md` | PASS | The repository now contains the final checklist that must be reviewed before any first real smoke can be approved, while still blocking execution. | Real smoke execution is still blocked until explicit user approval. | No |

### 11. Frontend / Wording Consistency

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| AI Assistant page wording reflects mock backend Agent | `tests/frontend/test_ai_assistant_page_copy.py::test_ai_assistant_page_copy_reflects_mock_agent_and_future_llm_boundary` | PASS | The page copy acknowledges the mock Agent and future LLM boundary. | None in the current evidence set. | No |
| Safety page wording reflects mock Agent and LLM disabled | `tests/frontend/test_safety_page_copy.py::test_safety_page_copy_is_current` | PASS | Safety copy reflects the active mock Agent state and external LLM boundaries. | None in the current evidence set. | No |
| No stale “no backend agent implemented yet” wording | `tests/frontend/test_frontend_wording.py::test_frontend_source_no_longer_shows_global_ai_placeholder_sentence`, plus page-specific wording tests | PASS | The stale placeholder sentence is no longer present. | Future wording changes should keep the assertion updated. | No |
| No misleading production/deployment readiness claims | `tests/frontend/test_frontend_wording.py`, `tests/agent/test_safety_guard.py` | PASS | Copy and guard logic stay aligned on non-readiness. | None in the current evidence set. | No |
| UI truncation fix is preserved | `tests/frontend/test_safety_page_copy.py`, `tests/frontend/test_visual_presentation_polish.py` | PASS | The Safety multi-model signal label is shortened and the broader presentation polish remains stable. | None in the current evidence set. | No |

### 12. Local Validation Before Gemini

| Requirement | Test file / validation | Status | Evidence summary | Remaining gap | Blocks Gemini |
|---|---|---|---|---|---|
| Python compile and pytest validations pass | `python -m compileall ...`, `pytest tests/agent/ -q`, `pytest tests/api/test_agent_endpoint.py -q`, `pytest tests/frontend/ -q` | PASS | The current local Python and pytest validation set passes. | None in the current evidence set. | No |
| Docker / Compose LLM-disabled smoke validation | `docs/agent/pre_gemini_docker_compose_smoke_validation.md` | PASS | The mock-first Docker / Compose stack started successfully with LLM disabled, the API and Agent health endpoints returned expected results, the component-aware mock explain smoke succeeded, and the frontend served HTTP responses. | Re-run if Docker, Compose, or runtime assets change. | No |
| Runtime asset validation / maintenance | `docs/agent/pre_gemini_docker_compose_smoke_validation.md` | MANUAL | The smoke validated the current mock-first API, Agent, and frontend path with LLM disabled; runtime assets should be rechecked if model/runtime assets, Dockerfiles, Compose, or serving surfaces change. | Re-run the smoke when the deployment surface changes. | No |
| Final manual visual checks are either PASS or pending based on actual evidence | `tests/frontend/*`, prior manual UI validation notes | MANUAL | Visual checks have been performed for the active panels, but a new future UI change would require re-review. | Manual review remains relevant for UI polish and copy drift. | No |

### 13. Explicit Remaining Blockers Before Gemini

| Blocker | Evidence / source | Status | Notes | Blocks Gemini |
|---|---|---|---|---|
| Docker / Compose LLM-disabled smoke validation | `docs/agent/pre_gemini_docker_compose_smoke_validation.md` | PASS | The final local deployment smoke passed in mock-first, LLM-disabled mode. | Re-run if Docker, Compose, or runtime assets change. | No |
| Gemini Provider Integration Readiness Plan | `docs/agent/gemini_provider_integration_readiness_plan.md` | PASS | The formal Gemini readiness plan now documents the non-negotiable integration rules, rollout phases, failure behavior, and acceptance criteria. | Future implementation still requires user approval. | No |
| Keep the matrix current if scope changes | `docs/agent/pre_gemini_test_matrix.md`, `docs/agent/active_explainability_scope_acceptance.md` | MANUAL | If the accepted scope changes, this matrix must be updated. | Depends | No |
| Any new requirement added after this matrix | Future repo state | MANUAL | A new requirement would need its own test mapping. | Depends |

## PASS Items

The following are currently covered well enough to count as PASS in the current evidence set:

- compileall and focused pytest suites
- component registry contract
- component evidence loader
- context builder with optional `component_id`
- Agent API schema and endpoint compatibility
- active component wiring for the four priority surfaces
- mock provider behavior
- deterministic safety guard
- provider contract/readiness layer
- Gemini provider G1 stub
- Gemini provider G2 mocked client tests
- Gemini provider G3 preparation audit
- Gemini provider G3 package verification
- Gemini provider G3 entry checklist
- Gemini provider G3 first slice readiness scaffolding
- Gemini provider G3 lazy SDK loader boundary
- Gemini provider G3 health/readiness integration
- Gemini provider G3 real provider execution boundary with lazy import and injected SDK/client only
- Gemini provider G3 pre-real-call audit
- Gemini provider G3 execution gate design
- Gemini provider G3 router activation tests
- Gemini provider G3 dependency decision
- Gemini provider G3 backend/API requirements slice
- frontend wording consistency
- the Safety truncation/UI polish fix

## PARTIAL / PENDING Items

The following remain incomplete or not yet fully proven:

- user approval to start the Gemini implementation phase

## Gemini Blockers Remaining

Current blockers supported by the audit docs:

1. User approval is required before any Gemini implementation phase begins.

## Required Next Sequence

1. Request and obtain user approval before any Gemini implementation phase.
2. Re-run the requirement-to-test matrix if the scope changes.

## Gemini Must Not Start Until...

- The active explainability scope is explicitly accepted.
- The final LLM-disabled Docker / Compose smoke validation has passed.
- The Gemini readiness plan exists and is reviewed.
- The user has approved the next implementation phase.
- The requirement-to-test matrix remains current after any scope change.
- The mock Agent foundation continues to pass compile and pytest validation.
