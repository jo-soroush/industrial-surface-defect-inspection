# Phase 3 Known Issues

## Import path resolution for src package

During Phase 3 Step 17, `src/inspection_ai/models/factory.py` passed
`py_compile`, and import/instantiation worked when running with
`PYTHONPATH=src`. A direct import without `PYTHONPATH=src` failed with:

```text
ModuleNotFoundError: No module named 'inspection_ai'
```

This is a runtime package-resolution issue for the repository's `src/` layout.
It is not a model factory logic failure.

This matters because real training execution must be able to resolve the
project package consistently. The issue must be resolved before real CLI
training execution and before CI/CD training jobs are treated as reliable. It
affects local CLI execution, CI/CD jobs, Docker execution, and broader MLOps
reliability.

Follow-up options include:

- Install the package in editable mode.
- Define an official `PYTHONPATH=src` execution convention.
- Create or adjust a project-level runtime wrapper.
- Update CI/CD and Docker execution setup accordingly.
