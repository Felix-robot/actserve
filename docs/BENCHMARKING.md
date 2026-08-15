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

`actserve benchmark-cuda` adds a real CUDA execution benchmark with a public
synthetic ViT-style policy. It measures scheduler and batching behavior on real
hardware, but it is still not a closed-loop task-success comparison against a
production VLA/WAM runtime.

Pass `--batch-sizes 4,8,16` to calibrate once and compare several batch ceilings
under an identical workload. The recommendation prioritizes useful on-time
actions, then lower p95 latency, then fewer backend calls.
