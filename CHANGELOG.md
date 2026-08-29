# Changelog

All notable changes to ActServe are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Optional bounded scheduler queues with explicit `scheduler_overloaded`
  outcomes; the standalone HTTP server defaults to a 1024-request pending
  limit and returns retryable HTTP 429 responses when it is full.
- Rolling p90 HTTP backend latency estimation by batch size, with configurable
  warm-start floor and safety factor for deadline-aware admission.

### Changed

- GitHub Actions now use pinned Node 24-compatible checkout and uv setup
  revisions.

## [0.8.0] - 2026-08-29

### Added

- Generic identity-preserving JSON/HTTP policy backend.
- `actserve serve` standalone control plane with localhost-by-default binding,
  optional front-end and backend Bearer tokens sourced from environment
  variables, health and readiness probes, and graceful backend shutdown.
- Installable-wheel smoke tests and Python 3.13 CI coverage.

### Fixed

- FastAPI metadata and `/healthz` now report the installed ActServe version
  instead of the historical hard-coded `0.4.0` value.

## [0.7.0] - 2026-08-16

### Added

- Shared-backbone adapter routing with opt-in mixed-adapter batching.
- Offline adapter residency planning under a declared GPU-memory budget.
- Synthetic shared-backbone routing benchmark and formal-workflow integration
  gate.

## [0.6.0] - 2026-08-16

### Added

- Privacy-safe training phase profiler for data wait, forward, backward,
  optimizer, and checkpoint timing.
- Evidence-based training trial selection with OOM, memory-headroom, and p95
  constraints.

## [0.5.0] - 2026-08-16

### Added

- Privacy-safe serving and command profilers.
- Asynchronous low-watermark action queue and public timing benchmark.

## [0.4.0] - 2026-08-15

### Added

- Embodied.cpp ZeroMQ/Protobuf backend and live public-checkpoint overload
  benchmark.
- Deadline-aware EDF scheduling, latest-frame coalescing, dynamic batching,
  explicit failure outcomes, metrics, replay, and optional FastAPI server.

[Unreleased]: https://github.com/Felix-robot/actserve/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/Felix-robot/actserve/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Felix-robot/actserve/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Felix-robot/actserve/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Felix-robot/actserve/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Felix-robot/actserve/releases/tag/v0.4.0
