# Contributing

ActServe welcomes backend adapters, scheduler policies, benchmarks, and
documentation improvements.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
uv build
```

## Pull-request requirements

- Add tests for scheduling or failure-semantics changes.
- Keep the core dependency-free unless a dependency is essential.
- Do not include model weights, private camera data, prompts, credentials,
  unpublished task configurations, or proprietary traces.
- State whether benchmark numbers are simulated, replayed, or measured on real
  hardware.
- Runtime performance claims must follow `docs/BENCHMARKING.md` and include
  behavior-preservation evidence when a robot or simulator is involved.

## Design principle

Late, stale, mismatched, and failed actions are different outcomes. Never merge
them into a generic success count.
