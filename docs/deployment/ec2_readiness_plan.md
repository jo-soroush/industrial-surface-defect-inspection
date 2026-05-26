# EC2 Readiness and Deployment Runbook

## Purpose

This document defines the safe EC2 readiness path for the Industrial Surface Defect Inspection Platform after successful local Docker and Docker Compose validation.

It is a demo/readiness planning document. It does not claim production readiness, deployment safety, or operational hardening. The goal is to preserve the evidence from local validation, avoid known architecture mistakes, and define the checks required before any EC2 deployment commands are run.

## Current Validated Local State

The current committed split-image state has been validated locally on the Mac development machine.

- Split API and frontend Docker images exist.
- Local Docker Compose validation passed.
- The API service passed `/health`.
- The API service passed `/agent/health`.
- The API service passed `/inspect/image`.
- The frontend opened successfully on port `8501`.
- Frontend Image Inspection worked.
- The AI explanation panel worked.
- The duplicate anomaly warning remained fixed.

The local validation confirms that the current API/frontend split works in the developer environment. It does not, by itself, confirm that the same local images are suitable for EC2.

## Architecture Warning: Mac ARM vs EC2 AMD64

The local Mac-built images were built as `linux/arm64`.

Common EC2 hosts are often `linux/amd64`. Do not transfer or reuse the Mac-built ARM images directly on an AMD64 EC2 host. Running an ARM image on an AMD64 host can fail with an executable format error, including this known failure pattern:

```text
exec /usr/local/bin/uvicorn: exec format error
```

Before deployment, the target EC2 architecture must be confirmed. The deployment path must match that architecture:

- Use AMD64 images on AMD64 EC2.
- Use ARM64 images on ARM64 EC2.
- Build natively on the EC2 instance when the target architecture is uncertain or when avoiding cross-platform image transfer risk.

## Recommended First EC2 Demo Strategy

For the first EC2 demo, use the source repository plus a verified `runtime_assets` bundle, then build the API and frontend images natively on the EC2 instance.

This strategy:

- avoids reusing Mac-built ARM images on AMD64 EC2;
- makes the EC2 build produce images for the host architecture;
- keeps `runtime_assets` out of Git;
- includes `runtime_assets` in a controlled deployment bundle;
- avoids requiring a registry for the first demo path.

This is a readiness/demo strategy, not a final production deployment architecture.

## Runtime Asset Requirements

The Docker build requires `runtime_assets/` to exist in the Docker build context before image build starts.

Important repository boundaries:

- `runtime_assets/` is ignored by Git.
- Raw `artifacts/` are ignored by Git.
- `data/processed/` is ignored by Git.
- `Dockerfile.api` and `Dockerfile.frontend` copy files from `runtime_assets/`.

Therefore, a fresh clone is not enough for an EC2 build unless `runtime_assets/` is generated or transferred first.

Recommended asset handling for the first EC2 demo:

- Keep raw artifacts and processed datasets out of Git.
- Generate `runtime_assets/` from available source artifacts when those artifacts are present on the EC2 host.
- Or transfer a verified `runtime_assets` bundle as part of a controlled deployment package.
- Validate `runtime_assets` before running any Docker build.

Validation command:

```bash
python scripts/runtime_assets/stage_runtime_assets.py --manifest configs/runtime_assets/manifest.yaml --check
```

The command must pass before attempting an EC2 Docker build.

## EC2 Storage Guidance

The current local image size evidence shows that the API image is large and the Docker build cache can consume significant disk space.

Qualitative guidance:

- An `8GB` root volume is risky.
- A `16GB` root volume is also risky.
- A `30GB` root volume may work, but can be tight during builds and repeated validation.
- A `50GB` root volume is safer for demo work.
- Docker build cache, intermediate layers, pulled base images, and failed builds can consume significant space.
- Docker pruning can be used later as planned maintenance, but should not be run during critical validation unless it is part of the deployment plan and the impact is understood.

This document does not make AWS pricing claims. Storage should be selected based on the observed image sizes and expected build-cache behavior.

## Security Boundaries

Do not bake secrets into Docker images.

The current MVP uses a mock-first AI provider path:

- `AGENT_ENABLE_LLM=false` by default.
- `AGENT_DEFAULT_PROVIDER=mock` by default.
- Gemini and Grok keys are not required for the current demo.

If real providers are enabled later, provide credentials through environment variables or a managed secret mechanism. Do not commit keys, include them in images, or place them in deployment bundles.

For a short demo, direct access to ports `8000` and `8501` may be acceptable only in a controlled environment. Do not expose ports `8000` and `8501` publicly long-term without a reverse proxy, HTTPS, and appropriate access controls.

Nginx, HTTPS, domain configuration, and stronger network boundaries are later hardening steps. They are not required for this immediate readiness plan.

## EC2 Deployment Checklist

- [ ] Choose the EC2 architecture.
- [ ] Choose the EC2 disk size.
- [ ] Install Docker and Docker Compose.
- [ ] Clone the repository.
- [ ] Transfer or generate `runtime_assets/`.
- [ ] Validate `runtime_assets/`.
- [ ] Build API and frontend images on EC2.
- [ ] Run `docker compose up`.
- [ ] Test `/health`.
- [ ] Test `/agent/health`.
- [ ] Test `/inspect/image`.
- [ ] Open the frontend.
- [ ] Test Image Inspection.
- [ ] Test the AI explanation panel.
- [ ] Document results and screenshots.

This checklist is intentionally ordered so architecture, storage, and runtime assets are confirmed before image build or service startup.

## Decision Table

| Strategy | Pros | Cons | Risk | Recommendation |
| --- | --- | --- | --- | --- |
| Build on EC2 after source plus `runtime_assets` transfer | Builds for the EC2 host architecture; avoids Mac ARM image reuse; does not require a registry; simple first-demo mental model | Requires Docker and enough disk on EC2; requires verified `runtime_assets`; build may take time | Medium: disk pressure and missing assets are the main risks | Recommended first EC2 demo path |
| Build `linux/amd64` locally and push to registry | Produces AMD64 images for common EC2 hosts; EC2 can pull instead of build; repeatable once registry is configured | Requires registry setup, authentication, push/pull workflow, and cross-platform build discipline | Medium: registry/auth complexity and possible cross-platform build issues | Good later path after demo readiness is documented |
| Build `linux/amd64` locally and `docker save`/`load` | Avoids registry; can transfer exact images; avoids ARM/AMD64 mismatch if built correctly | Large image archives; slow transfer; manual process; easy to mix old images | Medium to high: large artifacts and manual handling increase deadline risk | Not preferred for first demo unless registry access is unavailable and EC2 build is impossible |
| Use ARM64 EC2 | Matches current Mac ARM image architecture; can reduce architecture conversion work; may be fast for a demo if selected intentionally | EC2 instance type must be ARM64; dependency and image expectations must be checked; less representative of common AMD64 assumptions | Medium: safe only if the target is deliberately ARM64 | Viable if the EC2 target is confirmed ARM64; otherwise do not rely on this |

## Next Recommended Action

Before any EC2 deployment commands, confirm the target EC2 architecture and storage plan.

Do not run EC2 deployment commands, Docker build commands, Docker Compose commands, AWS commands, or SSH sessions until the architecture and storage decision is recorded.
