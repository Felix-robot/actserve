# LeRobot SmolVLA backend

`SmolVLABackend` loads a LeRobot SmolVLA policy in process and exposes it to
ActServe without application-specific batching code. Model work runs on one
dedicated worker thread, so scheduler deadlines, HTTP admission, metrics, and
new request handling remain responsive while the accelerator is busy.

## Install and start

Install the serving stack and LeRobot separately so the dependency-free
ActServe core never forces a particular PyTorch build:

```bash
pip install "actserve[server] @ git+https://github.com/Felix-robot/actserve.git"
pip install "lerobot[smolvla]>=0.6.1"
actserve serve-smolvla --device cuda
```

Use `--device mps` on Apple silicon or `--device auto` to select CUDA, then
MPS, then CPU. The default public model uses both weights and processors from
the tested revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`. A custom
`--model-id` does not inherit that revision; pass `--revision` explicitly when
reproducibility matters.

The service binds to `127.0.0.1:8080` and serves the same `/v1/actions`,
`/v1/metrics`, `/metrics`, `/healthz`, and `/readyz` routes as the generic
ActServe server. Set a public-facing model alias with `--served-model-name`.

## Observation contract

Each request is one independent observation. For the default checkpoint,
`observation` contains these fields:

```json
{
  "observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "observation.images.camera1": "nested float array with shape [3, 256, 256]",
  "observation.images.camera2": "nested float array with shape [3, 256, 256]",
  "observation.images.camera3": "nested float array with shape [3, 256, 256]",
  "task": "Move the public object to the public target."
}
```

Image tensors or arrays are channel-first float values with the exact shapes in
the loaded policy config. A single leading batch dimension of size one is also
accepted. Missing keys, wrong shapes, empty task text, a mismatched served model
name, or the wrong number of returned action chunks fails the request explicitly.

JSON image arrays are deliberately simple but bandwidth-heavy. Existing Python
robot loops should prefer the in-process API below. The HTTP route is useful for
local integration and correctness testing; it is not a replacement for the
planned binary transport.

## In-process integration

```python
import asyncio
import torch

from actserve import InferenceRequest, Scheduler, SmolVLABackend


async def main():
    backend = SmolVLABackend.from_pretrained(
        device="cuda",
        max_batch_size=4,
    )
    observation = {
        "observation.state": torch.zeros(6),
        "observation.images.camera1": torch.zeros(3, 256, 256),
        "observation.images.camera2": torch.zeros(3, 256, 256),
        "observation.images.camera3": torch.zeros(3, 256, 256),
        "task": "Move the public object to the public target.",
    }
    async with Scheduler(backend) as scheduler:
        outcome = await scheduler.submit(
            InferenceRequest.with_timeout(
                session_id="robot-1",
                model="lerobot/smolvla_base",
                observation=observation,
                timeout_ms=5_000,
                sequence_no=1,
                metadata={"input_signature": "smolvla-base-v1"},
            )
        )
    await backend.aclose()
    if outcome.action is not None:
        consume_action_chunk(outcome.action.actions)


asyncio.run(main())
```

The backend resets policy-local state before every dynamic batch and therefore
accepts only policies with `n_obs_steps == 1`. This prevents accidental history
sharing across robot sessions. A future stateful integration needs an explicit
per-session state contract.

## Deadline and capacity tuning

The backend learns a rolling p90 latency independently for each observed batch
size. It scales unseen sizes conservatively and exposes those estimates to the
scheduler. For fail-closed admission from startup, measure an isolated warmup
on the same model and hardware, then configure a floor:

```bash
actserve serve-smolvla \
  --device cuda \
  --max-batch-size 4 \
  --initial-backend-latency-ms 250 \
  --latency-safety-factor 1.15 \
  --drop-unserviceable-requests
```

Do not assume a larger batch is faster. Run
`benchmarks/lerobot_smolvla.py` on the intended backend and choose the smallest
ceiling that meets the workload's deadline and throughput requirements.

The server never sends actions to hardware. Authentication, asynchronous action
queueing, robot-specific fallback behavior, simulator parity, and physical
safety gates remain separate integration responsibilities. See
[`WORKFLOW_INTEGRATION.md`](WORKFLOW_INTEGRATION.md) before attaching any formal
or physical workflow.
