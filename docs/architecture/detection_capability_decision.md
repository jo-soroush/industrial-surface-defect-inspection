# Detection Capability Decision Lock

The project includes a planned future object-detection track using YOLO.

The selected future detection dataset is NEU-DET.

Current governed data tracks remain:

- NEU for classification
- MVTec for anomaly detection

Detection is approved in scope, but it is not yet operationally implemented in Phase 2.
No NEU-DET dataset payload is currently present in the repository.
No detection Phase 2 manifests, configs, or reports exist yet.

Any future detection Phase 2 extension should follow the same structure-first, truthful zero-state pattern already used for the governed NEU and MVTec data tracks.

## Detection Phase 2 Pattern Alignment

Any future detection Phase 2 extension should follow the same repository pattern already used for the governed NEU and MVTec data tracks.

This means detection governance may be established in a structure-first manner, and a truthful zero-state is allowed before dataset payload is present.

If detection governance artifacts are created later, they must remain auditable, truthful, and free of fake validation state or implicit assumptions about dataset contents.
