# Phase 3 Verification

## Purpose

This verification checks that the Phase 3 training pipeline can run end to end through the smoke test entry point. It confirms that the training CLI starts, produces a structured training result, and exits successfully.

## Command

```bash
python scripts/testing/run_phase3_smoke_test.py
```

## Expected Output

A successful run prints:

```text
[PASS] Phase 3 smoke test
```

A failed run prints:

```text
[FAIL] Phase 3 smoke test
```

The smoke test passes only when the training command exits with status `0` and its stdout includes `Training result created`.

## Notes

- The configured run is an experiment, so no registry updates are expected.
- The temporary training result artifact created during the smoke test is deleted after the run.
- This is a smoke test for pipeline wiring and validation, not real training.
