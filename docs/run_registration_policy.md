# Run Registration Policy

This policy defines when training runs are treated as experiments and when they
are eligible for governed run registry registration.

## Experiment runs

Run configs with `identity.is_experiment: true` are experiment runs. They may
produce local or analysis outputs, including structured `TrainingResult` JSON,
but they must not update `run_registry.yaml`.

Experiment runs are appropriate for scaffold execution, dry runs, development
checks, integration checks, and exploratory validation where the output should
not be treated as governed run evidence.

Current Phase 3 scaffold run configs are experiment runs because real training
is not implemented yet.

## Governed runs

Run configs with `identity.is_experiment: false` are eligible for run registry
registration. Eligibility does not mean the run registers model artifacts; model
artifact registration is a separate governance process.

A run may become non-experiment only when its config, data binding, output
persistence, and validation behavior are governed and intentional.

Minimum eligibility criteria for `identity.is_experiment: false`:

- A resolved run config exists.
- `identity.task_type` is valid.
- `model_identity.model_type` is valid.
- `identity.run_config_id` is present.
- The output path is governed.
- Result persistence works.
- The run registry writer works.
- The run purpose is intentional.
- Placeholder-only runs are not promoted unless explicitly marked as validation evidence.

## Registry eligibility

The run registry records completed governed runs. It should not include routine
dry runs, placeholder execution, local development checks, or scaffold runs that
do not represent intentional governed evidence.

This prevents registry noise, avoids mixing dry runs with governed runs, reduces
misleading CI/CD evidence, and preserves clear MLOps lineage.

## Dry-run behavior

Dry runs should use `identity.is_experiment: true`. They may validate config
loading, dispatch, model factory resolution, structured result creation, and
local result persistence. They must not create run registry entries unless the
dry run is explicitly promoted as validation evidence through a governed config
and review decision.

## CI/CD and MLOps implications

CI/CD jobs that execute experiment configs must not treat successful execution
as governed model-training evidence. They may use the output as scaffold,
integration, or smoke-test evidence only.

MLOps workflows should use `identity.is_experiment: false` only for intentional
governed runs where lineage, result persistence, and registry updates are part
of the expected operational record.
