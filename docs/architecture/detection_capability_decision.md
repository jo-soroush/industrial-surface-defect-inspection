# Detection Capability Decision Lock

The project includes an active object-detection track using YOLO.

The selected detection dataset is GC10-DET.

Current governed data tracks remain:

- MVTec for classification
- MVTec for anomaly detection
- GC10-DET for object detection

Detection is part of the active governed scope, but Phase 2 detection governance is not yet fully aligned across manifests, configs, and reports.
The raw GC10-DET payload is present in the repository.
NEU and NEU-DET are outside the current executable scope.

Detection governance alignment should follow the same structure-first, truthful zero-state pattern already used for the governed data tracks in this repository.

## Detection Phase 2 Pattern Alignment

Detection governance alignment should follow the same repository pattern already used for the governed data tracks in this repository.

This means detection governance may be established in a structure-first manner, and any incomplete governance state must remain truthful and auditable.

If detection governance artifacts are created later, they must remain auditable, truthful, and free of fake validation state or implicit assumptions about dataset contents.
