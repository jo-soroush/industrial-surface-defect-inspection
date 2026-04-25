# Training Run Config Contract

## Purpose

This document defines the execution-input contract for governed training runs in
this repository. Its purpose is to make the training entry architecture
explicit, reproducible, CI/CD-friendly, and MLOps-friendly before executable run
configs are introduced.

This contract applies to the training entrypoint:

- `scripts/training/train_model.py`

## Architectural Decision

The training entrypoint must accept exactly one fully resolved run config as its
executable input.

This repository therefore distinguishes between three configuration layers:

- `configs/models/`
  Source-layer model configuration and model-family defaults
- `configs/training/`
  Source-layer training policy and runtime defaults
- `configs/runs/`
  Execution-layer resolved run configs used as direct inputs to training runs

`configs/models/` and `configs/training/` are governance and source layers. They
are not direct execution inputs.

`configs/runs/` is the intended execution layer for governed training runs.

## Required Execution Model

`scripts/training/train_model.py` must receive one fully resolved run config
that already contains the fields required to execute a governed training run.

The entrypoint must not depend on hidden runtime merging between multiple config
files in order to determine effective run behavior.

The execution config must be explicit enough that a run can be:

- reviewed before execution
- reproduced later
- referenced in CI/CD workflows
- referenced in MLOps pipelines
- traced to a single governed execution input

## Why Direct Multi-Config Execution Is Not Preferred

Direct execution from multiple partial configs is not the preferred
architecture because it makes the effective runtime contract ambiguous.

If `train_model.py` must combine model config, training config, dataset binding,
and runtime overrides at execution time, the true run definition becomes
distributed across multiple files and possibly CLI state. That weakens
traceability and makes run reconstruction more error-prone.

Multi-config direct execution also increases the risk of:

- hidden precedence rules
- accidental runtime drift
- untracked override behavior
- CI/CD ambiguity about what was actually executed
- weaker handoff into model registry and artifact governance

## Why Hidden Merge Logic Is Not Allowed

Hidden merge logic inside `train_model.py` is not allowed because the training
entrypoint is an execution boundary, not a configuration-resolution engine.

The effective run definition must be externally governed and explicit before the
entrypoint is invoked. If merge or override behavior exists, it must be handled
through a governed config-generation step, not implicitly during execution.

This keeps the entrypoint behavior deterministic and reviewable.

## Why Resolved Run Configs Are Preferred

Resolved run configs are preferred because they provide a single source of truth
for actual execution.

This is better for:

- reproducibility
- CI/CD invocation
- scheduled training workflows
- artifact traceability
- model-lifecycle governance
- MLOps integration

A resolved run config makes it possible to answer, unambiguously, which model,
dataset, preprocessing assumptions, training parameters, and output controls
were used for a specific run.

## Required Sections for a Resolved Run Config

A resolved run config must contain, at minimum, the following sections.

### Identity

Required fields:

- `run_config_id`
- `description`
- `task_type`
- `track_id`

### Model Identity

Required fields:

- `model_name`
- `model_type`
- `model_version`
- `model_config_id`

### Dataset Binding

Required fields:

- `dataset_id`
- `dataset_version`
- `split_manifest_path`

### Preprocessing

Required fields:

- `preprocessing_policy_path`
- `preprocessing_version`
- `augmentation_policy_version`

### Training / Runtime

Required fields:

- `batch_size`
- `learning_rate`
- `epochs`
- `optimizer`
- `loss_function`
- `seed`
- `device`

### Checkpoint / Early Stopping

Required fields:

- `checkpoint.enabled`
- `checkpoint.save_best`
- `checkpoint.save_last`
- `early_stopping.enabled`
- `early_stopping.patience`

### Output Control

Required fields:

- `save_artifact`
- `save_metadata`
- `save_logs`
- `save_metrics`

## Hard Rules

- No ambiguous execution input is allowed.
- No training execution from a partial config only is allowed.
- No hidden runtime overrides are allowed without explicit governance.
- No hidden config merge behavior is allowed inside `train_model.py`.
- The resolved run config must be self-sufficient enough for governed training
  entry.
- Source-layer configs may inform a run config, but they must not be treated as
  the executable contract themselves.

## Entry-Point Boundary Rule

The responsibility of `scripts/training/train_model.py` is to:

- accept one governed run config
- validate that execution input
- route execution by governed task type

It is not responsible for:

- resolving partial configuration graphs implicitly
- discovering missing execution fields dynamically
- inventing runtime defaults silently
- applying hidden environment-specific overrides

## Governance Outcome

This contract establishes the expected separation between:

- source-layer configuration
- execution-layer configuration
- runtime entrypoint behavior

That separation is required for company-grade training governance and for later
Phase 3, Phase 4, and Phase 5 model-lifecycle controls.
