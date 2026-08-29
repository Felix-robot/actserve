# Benchmarking contract

Performance claims must use the same model, weights, numerical mode, hardware,
observations, action schema, and simulator seeds for every scheduler.

Report at least:

- submitted observations and useful on-time actions;
- p50/p95/p99 end-to-end latency;
- deadline misses and expirations;
- observations replaced before dispatch;
- backend calls and mean/max batch size;
- peak device memory and task success when a real model is used.

Replacement is not success. It is useful load shedding and must be reported
separately. A faster runtime is not behavior-preserving until closed-loop task
success is measured.

The bundled benchmark uses a public simulated latency model. It validates
scheduler semantics and is not evidence of GPU or task-level superiority.

`actserve benchmark-serial` models a single-request runtime such as the current
Embodied.cpp VLA server. It isolates the benefit of stale-frame coalescing and
predicted-deadline-miss admission control, but does not execute Embodied.cpp or
support a real-model performance claim.

`actserve benchmark-cuda` adds a real CUDA execution benchmark with a public
synthetic ViT-style policy. It measures scheduler and batching behavior on real
hardware, but it is still not a closed-loop task-success comparison against a
production VLA/WAM runtime.

Pass `--batch-sizes 4,8,16` to calibrate once and compare several batch ceilings
under an identical workload. The recommendation prioritizes useful on-time
actions, then lower p95 latency, then fewer backend calls.

`benchmarks/lerobot_smolvla.py` is the real public-model integration benchmark.
It pins a SmolVLA revision, constructs deterministic synthetic observations,
uses explicit per-request diffusion noise, warms both measured batch shapes,
and counterbalances serial-first and ActServe-first trial order. Publish the raw
paired samples and action parity rather than only the best summary number.

The SmolVLA benchmark does not run a simulator or robot. Its result supports
only serving compatibility and performance for the exact hardware, software,
and workload recorded in the result. Select `max_batch_size` from a measurement
on the intended backend; do not assume dynamic batching is universally faster.
