# ActServe

[![CI](https://github.com/Felix-robot/actserve/actions/workflows/ci.yml/badge.svg)](https://github.com/Felix-robot/actserve/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](pyproject.toml)

Deadline-aware inference serving for embodied AI agents.

ActServe treats robot inference as a stream of perishable observations,
not a queue of ordinary request-response jobs. It schedules the observation with
the earliest control deadline, replaces stale pending frames from the same robot,
microbatches compatible requests, and refuses to return late actions as if they
were safe successes.

> v0.7 adds shared-backbone adapter routing and residency planning to the
> privacy-safe training/serving profiler, asynchronous action queue, and public
> benchmark harness. It does not send hardware commands or claim task
> superiority.

Formal experiments must follow the explicit staged gate in
[`docs/WORKFLOW_INTEGRATION.md`](docs/WORKFLOW_INTEGRATION.md); public benchmark
success alone never authorizes attachment to an existing job.

## The pain it solves

A FIFO model server can spend GPU time on frame 42 even after frame 43 has
arrived. Under multiple robot sessions, this creates queue growth, stale actions,
and long-tail latency. ActServe adds the control-plane semantics that a
closed-loop serving system needs:

- persistent robot sessions;
- asynchronous action queues with explicit low-watermark refill;
- explicit shared-backbone and multi-adapter routing;
- monotonic per-observation deadlines;
- earliest-deadline-first scheduling;
- pending-frame coalescing per session;
- duplicate/out-of-order observation rejection;
- compatible dynamic microbatching;
- latency-aware batch admission when the backend supplies an estimator;
- explicit expired, replaced, missed, and failed outcomes;
- privacy-safe scheduling traces;
- privacy-safe training phase and bottleneck profiling;
- backend-neutral integration;
- fail-closed action identity validation across request, session, and sequence.

## Install

```bash
uv sync --extra dev
```

The scheduler core has no runtime dependencies. Python 3.10–3.13 is tested.
Install from the repository with `pip install "actserve[server] @
git+https://github.com/Felix-robot/actserve.git"` until a PyPI release is
published.

## Try it

```bash
uv run python examples/basic.py
uv run actserve benchmark --sessions 8 --observations 20
uv run actserve benchmark-adapters
uv run actserve benchmark-async
uv run actserve benchmark-serial
uv run actserve plan-adapters examples/adapter_demand.json --budget-mb 512
uv run actserve profile --gpu 0 -- python your_workload.py
uv run actserve tune-training examples/training_trials.json
uv run pytest
```

`actserve profile` records process wall time plus optional GPU utilization,
memory, and power samples. Command arguments are redacted unless
`--include-command` is explicitly supplied, and model inputs/outputs are never
captured.

`actserve benchmark-async` uses a public synthetic timing loop to compare
blocking chunk inference with low-watermark asynchronous refill. It demonstrates
control-loop overlap only; it does not claim simulator or robot task success.

On a CUDA machine with PyTorch already installed, run the public synthetic
vision-policy benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 actserve benchmark-cuda
CUDA_VISIBLE_DEVICES=0 actserve benchmark-cuda --batch-sizes 4,8,16
```

This calibrates batch latency on the current GPU, then compares FIFO batch-1
against ActServe using the same synthetic ViT-style policy and observation
stream. A batch-size sweep calibrates once, evaluates every candidate, and
recommends the ceiling with the most on-time actions before considering p95
latency and backend calls. It does not download model weights or use private
task data.

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

For a real external JSON policy endpoint, no application glue is required:

```bash
uv run actserve serve --backend-url http://127.0.0.1:9000/infer
```

The standalone server binds to `127.0.0.1` by default and supports front-end
and backend Bearer tokens supplied by environment-variable name. See
[`docs/HTTP_BACKEND.md`](docs/HTTP_BACKEND.md).

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

## Embodied.cpp integration

ActServe can sit in front of Embodied.cpp's serial ZeroMQ/Protobuf VLA server.
Embodied.cpp keeps responsibility for portable C++ model execution; ActServe
adds session coalescing, EDF ordering, admission control, response identity
validation, and adaptive end-to-end latency estimation.

Install the optional transport and follow the public integration guide:

```bash
uv sync --extra embodied-cpp
uv run actserve benchmark-serial
```

See [`docs/EMBODIED_CPP.md`](docs/EMBODIED_CPP.md). The serial benchmark is a
scheduler-only simulation and is not evidence of real-model speed or task
success. The guide also includes a live public-checkpoint benchmark that
measures latest-frame deadline success against the same unmodified server.

For decoupled action-chunk inference and execution, see
[`docs/ASYNC_ACTIONS.md`](docs/ASYNC_ACTIONS.md). The queue fails closed on
underrun and leaves robot-specific fallback commands outside the generic
runtime.

For dependency-free data-wait, compute, optimizer, and checkpoint timing, see
[`docs/TRAINING_PROFILER.md`](docs/TRAINING_PROFILER.md). It stores numeric
timings only and reports optimization hypotheses that still require isolated
workload validation.

For task adapters that share one loaded backbone, see
[`docs/ADAPTER_ROUTING.md`](docs/ADAPTER_ROUTING.md). Mixed-adapter batching is
opt-in and remains partitioned by backbone and input signature.

To put ActServe in front of an existing JSON policy service without sharing its
Python environment, use [`HttpJsonBackend`](docs/HTTP_BACKEND.md). It supports
dynamic batches while preserving request, session, model, and sequence identity.

## Roadmap

- Adaptive latency estimation and backpressure for HTTP policy backends.
- Session-aware vision-feature cache with explicit invalidation and parity tests.
- Public simulator benchmark with pinned model weights, seeds, and task-quality gates.
- gRPC transport for high-throughput binary observations.

See [`docs/POSITIONING.md`](docs/POSITIONING.md) for the deliberately narrow
comparison boundary and [`SECURITY.md`](SECURITY.md) before any physical-robot
integration.

Release history is in [`CHANGELOG.md`](CHANGELOG.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
