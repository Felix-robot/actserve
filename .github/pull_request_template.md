## What changed

Describe the user-visible behavior and interface.

## Why

State the concrete serving, training-efficiency, or integration pain.

## Validation

- [ ] `uv run ruff check .`
- [ ] `uv run pytest -q`
- [ ] `uv build`
- [ ] New failure semantics have tests.
- [ ] Package or CLI changes were tested from the built wheel.

## Evidence and privacy boundary

- [ ] Benchmark results are labeled synthetic, replayed, or real-hardware.
- [ ] No credentials, private observations/actions, model weights, unpublished
      task configurations, proprietary traces, or robot access details are included.
- [ ] Robot/simulator behavior claims include parity or task-quality checks; an
      engineering diagnostic is not presented as task success.
