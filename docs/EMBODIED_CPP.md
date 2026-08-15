# Embodied.cpp integration

ActServe and Embodied.cpp operate at different layers:

- Embodied.cpp loads and executes VLA/WAM models in a portable C++ runtime.
- ActServe decides which perishable observation should reach that runtime.

The current Embodied.cpp VLA endpoint is a ZeroMQ `REQ/REP` service with one
`PredictRequest` and one `PredictResponse` per inference. ActServe therefore
uses `max_batch_size = 1` for this adapter. Its gain comes from avoiding stale
or predictably late work, not from claiming that a serial C++ server supports
tensor batching.

## Install

```bash
uv sync --extra embodied-cpp
```

Generate `vla_pb2.py` from the `serving/vla.proto` shipped by the Embodied.cpp
checkout, as its own evaluation client does. Keep model-specific image,
tokenizer, state, and noise preprocessing in a request builder:

```python
from actserve.integrations import EmbodiedCppVlaBackend
from actserve.scheduler import Scheduler, SchedulerConfig

import vla_pb2


def build_predict_request(request):
    message = vla_pb2.PredictRequest()
    observation = request.observation
    # Populate images, language tokens/text, state, and optional noise using
    # the preprocessing contract for the selected public model.
    populate_public_model_request(message, observation)
    return message


backend = EmbodiedCppVlaBackend(
    protobuf_module=vla_pb2,
    request_builder=build_predict_request,
    address="tcp://127.0.0.1:5555",
    initial_latency_ms=50,
)
config = SchedulerConfig(
    policy="edf",
    coalesce_sessions=True,
    drop_unserviceable_requests=True,
)

# Use both async context managers so the dedicated ZeroMQ worker closes cleanly.
async with backend:
    async with Scheduler(backend, config) as scheduler:
        outcome = await scheduler.submit(request)
```

The adapter replaces the wire-level numeric request id and validates the reply
before creating an `ActionChunk`. A mismatched reply fails closed rather than
routing an action to the wrong session.

## Safe validation ladder

1. Run `actserve benchmark-serial` without model weights or experiment data.
2. Start a separate Embodied.cpp server with a public checkpoint and dedicated
   port; replay public or synthetic observations.
3. Compare direct FIFO and ActServe with identical model, precision, GPU,
   request trace, warmup, and deadlines.
4. Require repeated improvement in on-time actions without action-value drift.
5. Only then consider a separately authorized private adapter or closed-loop
   experiment.

Do not attach the adapter to an active experiment merely because the protocol
test passes. Transport correctness, scheduling gain, model parity, and
closed-loop task gain are separate gates.
