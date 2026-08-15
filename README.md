# ActServe

Deadline-aware inference serving for embodied AI agents.

ActServe treats robot inference as a stream of perishable observations,
not a queue of ordinary request-response jobs. It schedules the observation with
the earliest control deadline, replaces stale pending frames from the same robot,
microbatches compatible requests, and refuses to return late actions as if they
were safe successes.

> v0.2 is an alpha scheduler and public CUDA benchmark harness. It does not send
> hardware commands or claim closed-loop task superiority.

## The pain it solves

A FIFO model server can spend GPU time on frame 42 even after frame 43 has
arrived. Under multiple robot sessions, this creates queue growth, stale actions,
and long-tail latency. ActServe adds the control-plane semantics that a
closed-loop serving system needs:

- persistent robot sessions;
- monotonic per-observation deadlines;
- earliest-deadline-first scheduling;
- pending-frame coalescing per session;
- duplicate/out-of-order observation rejection;
- compatible dynamic microbatching;
- latency-aware batch admission when the backend supplies an estimator;
- explicit expired, replaced, missed, and failed outcomes;
- privacy-safe scheduling traces;
- backend-neutral integration;
- fail-closed action identity validation across request, session, and sequence.

## Install

```bash
uv sync --extra dev
```

The scheduler core has no runtime dependencies. Python 3.10+ is supported.

## Try it

```bash
uv run python examples/basic.py
uv run actserve benchmark --sessions 8 --observations 20
uv run pytest
```

On a CUDA machine with PyTorch already installed, run the public synthetic
vision-policy benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 actserve benchmark-cuda
```

This calibrates batch latency on the current GPU, then compares FIFO batch-1
against ActServe using the same synthetic ViT-style policy and observation
stream. It does not download model weights or use private task data.

In a 32-session stress run at 60 Hz per session with a 30 ms deadline, FIFO
produced 291/960 on-time actions while ActServe produced 960/960. ActServe also
reduced p95 end-to-end latency from 31.43 ms to 16.42 ms. See
[`benchmarks/README.md`](benchmarks/README.md) for both public runs, raw JSON,
checksums, and the evidence boundary.

Optional development HTTP server:

```bash
uv sync --extra server
uv run uvicorn examples.server:app
```

It exposes `POST /v1/actions`, JSON metrics at `/v1/metrics`, and Prometheus
exposition at `/metrics`.

Minimal integration:

```python
import asyncio

from actserve.backend import CallableBackend
from actserve.scheduler import Scheduler
from actserve.types import ActionChunk, InferenceRequest


async def infer(requests):
    # Stack tensors, run your model once, then return one ActionChunk per request.
    return [
        ActionChunk(
            request_id=req.request_id,
            session_id=req.session_id,
            sequence_no=req.sequence_no,
            actions=my_model(req.observation),
            model=req.model,
        )
        for req in requests
    ]


async def control_step():
    backend = CallableBackend(infer, max_batch_size=8)
    async with Scheduler(backend) as scheduler:
        outcome = await scheduler.submit(
            InferenceRequest.with_timeout(
                session_id="robot-1",
                model="my-vla",
                observation={"images": [], "state": []},
                timeout_ms=100,
                sequence_no=42,
            )
        )
        if outcome.action is not None:
            consume_action(outcome.action.actions)


asyncio.run(control_step())
```

## Benchmark honesty

The included simulated benchmark demonstrates scheduler behavior only. A claim
against another runtime requires identical public model weights, hardware,
precision, observations, simulator seeds, and action interfaces, plus both
systems' latency and closed-loop task success. See
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Private adapters stay private

The backend API is intentionally narrow. Proprietary models, camera data,
prompts, action conventions, checkpoints, and experiment traces can live in a
separate package. The default JSONL trace excludes observations and actions.

## Roadmap

- Public real-model adapter and reproducible same-model comparison.
- Session-aware vision-feature cache with explicit invalidation.
- Multi-adapter routing and residency policy.
- gRPC transport and Prometheus exporter.
- Backend adapter for a portable C++ embodied runtime.

See [`docs/POSITIONING.md`](docs/POSITIONING.md) for the deliberately narrow
comparison boundary and [`SECURITY.md`](SECURITY.md) before any physical-robot
integration.

## License

Apache-2.0. See [LICENSE](LICENSE).
