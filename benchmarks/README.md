# Public benchmark results

## Shared-backbone adapter routing — 2026-08-16

Three dependency-free repetitions modeled four logical task adapters with two
simultaneous sessions each. Isolated routing could batch only within an adapter;
the explicitly capable shared-backbone arm batched all eight requests together.

| Arm | Backend calls | Mean wall time | Modeled memory |
| --- | ---: | ---: | ---: |
| Isolated model instances | 4 | 106.09 ms | 29,024 MB |
| Shared backbone + adapters | 1 | 42.79 ms | 8,024 MB |

Every arm completed all eight requests. The 72.35% memory reduction follows the
declared 7,000 MB backbone and 256 MB-per-adapter model; it is not measured GPU
memory. This result validates routing, batching, and accounting mechanics only,
not real-model latency, policy quality, or closed-loop task success.

Raw result: [`results/adapter_shared_backbone_20260816.json`](results/adapter_shared_backbone_20260816.json)

SHA-256: `4daa00a01e981fb38d041fa11da8c14e6ce221837505a01267eeb659fb4fbbd5`

Run it with:

```bash
actserve benchmark-adapters
```

## Asynchronous action refill timing — 2026-08-16

This dependency-free benchmark compares blocking chunk inference against
ActServe's low-watermark asynchronous refill. Each of three repetitions used
12 chunks of 10 actions, 80 ms synthetic inference, and a 20 ms action tick.

| Loop | Mean wall time | Mean idle ticks | Max steady idle ticks |
| --- | ---: | ---: | ---: |
| Blocking | 3601.35 ms | 48 | 44 |
| Async refill | 2664.27 ms | 3 | 0 |

All six arms executed all 120 requested actions. This validates inference and
action-execution overlap in a synthetic timing loop only; it is not evidence of
closed-loop task success, model quality, or physical-robot safety.

Raw result: [`results/async_action_queue_20260816.json`](results/async_action_queue_20260816.json)

SHA-256: `aaabf72e62a137d9b73ddf4330f03fe1fcc1eca017d83e2a1006cf430f08a6c8`

Run it with:

```bash
actserve benchmark-async
```

## Live Embodied.cpp pi0.5 overload comparison — 2026-08-16

This benchmark used Embodied.cpp commit `c5a96a2` with its public pi0.5 LIBERO
GGUF checkpoint on a shared NVIDIA A800-SXM4-80GB. Both arms used the same
loaded model, precision, synthetic observations, fixed action noise, serial
ZeroMQ endpoint, and four-session arrival trace.

Across three repetitions, direct FIFO executed all 16 requests and delivered
zero of the four latest session frames before their deadlines. ActServe
executed five requests, replaced 11 stale pending frames, and delivered all
four latest frames on time in every repetition. Action deltas stayed within a
tolerance calibrated from the unmodified server's own repeated-request CUDA
variation; every run recorded zero parity failures.

| Run | Direct latest on time | ActServe latest on time | Direct calls | ActServe calls | Replaced |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0/4 | 4/4 | 16 | 5 | 11 |
| 2 | 0/4 | 4/4 | 16 | 5 | 11 |
| 3 | 0/4 | 4/4 | 16 | 5 | 11 |

Raw result: [`results/a800_embodied_cpp_pi05_live_20260816.json`](results/a800_embodied_cpp_pi05_live_20260816.json)

SHA-256: `149164bb13fba7472e7f909206b5cc53a817489eb7d18e8993e5ed2863a2a7f4`

This is real-model evidence for overload freshness and avoided stale compute.
It does not measure policy quality or closed-loop task success. The GPU was
shared with pre-existing workloads, but both benchmark arms ran against the
same isolated server without changing those workloads.

## Serial embodied-runtime scheduling simulation — 2026-08-15

This dependency-free benchmark models the current Embodied.cpp VLA serving
contract as a single-request serial backend. It does not load Embodied.cpp,
model weights, simulator observations, or private data.

Five repeated runs used 8 sessions, 12 observations/session, 30 Hz/session, an
80 ms deadline, and 15 ms serial backend latency:

- direct FIFO: 15/96 on-time actions in every run and 12 late dispatched
  actions in every run;
- ActServe + serial runtime: 26--27/96 on-time actions and zero late dispatched
  actions in every run;
- p95 end-to-end latency ranged from 85.42--86.05 ms for direct FIFO and
  51.28--59.01 ms for the combined control plane.

The apparent useful-action rate counts all incoming observations, including the
65 stale frames that ActServe explicitly replaced. This result validates the
scheduler mechanism under a serial overload model only. A superiority claim
still requires a public Embodied.cpp checkpoint and identical real requests.

Raw summary: [`results/serial_runtime_stability_20260815.json`](results/serial_runtime_stability_20260815.json)

SHA-256: `e29f9145ce02144c5844b3902a10007efcb80f070e2ead656dc913ba76b4a6b8`

Run it with:

```bash
actserve benchmark-serial
```

## A100 synthetic vision policy — 2026-08-15

This benchmark was executed on one NVIDIA A100-SXM4-80GB with PyTorch
2.9.0+cu126. It used the public synthetic ViT-style policy built into
`actserve benchmark-cuda`:

- image size: 224×224;
- hidden width: 384;
- transformer layers: 6;
- sessions: 16;
- observations per session: 20;
- observation rate: 30 Hz per session;
- deadline: 100 ms;
- maximum batch size: 8.

| Scheduler | On-time actions | Backend calls | Mean batch | p95 | p99 | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIFO batch-1 | 320/320 | 320 | 1.0 | 42.51 ms | 45.26 ms | 0.877 s |
| ActServe | 320/320 | 40 | 8.0 | 17.54 ms | 18.42 ms | 0.689 s |

Raw result: [`results/a100_synthetic_vit_20260815.json`](results/a100_synthetic_vit_20260815.json)

SHA-256:
`e2dec3d184da9c4c7ac0ae0d0b3ff4f651b59c14600c34f14b7badd6d90ebc21`

This establishes a real-GPU serving and batching result, not superiority over
another embodied runtime. It uses a synthetic policy and does not measure
closed-loop task success.

## A100 deadline-pressure run — 2026-08-15

This run used the same public synthetic policy and GPU with 32 sessions, 30
observations per session, 60 Hz per session, and a 30 ms deadline.

| Scheduler | On-time actions | Expired | Deadline missed | Backend calls | Mean batch | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIFO batch-1 | 291/960 | 636 | 33 | 324 | 1.0 | 31.43 ms |
| ActServe | 960/960 | 0 | 0 | 120 | 8.0 | 16.42 ms |

Raw result:
[`results/a100_stress_32x60hz_30ms_20260815.json`](results/a100_stress_32x60hz_30ms_20260815.json)

SHA-256:
`da257dfcd732b830021da3fabcf626fa67bdc4dd9cc19626105e16a441cb61e2`

This stress test shows deadline-aware batching under overload. It remains a
synthetic-policy serving benchmark and is not evidence of closed-loop task
success or superiority over another embodied runtime.

## v0.3 automatic batch sweep — 2026-08-15

The v0.3 sweep calibrated batch sizes 1–16 once, then evaluated candidate
ceilings 4, 8, and 16 under the same 32-session, 60 Hz, 20 ms workload.

| Batch ceiling | On-time actions | Backend calls | p95 |
| ---: | ---: | ---: | ---: |
| 4 | 373/640 | 117 | 20.02 ms |
| 8 | 640/640 | 80 | 17.33 ms |
| 16 | 640/640 | 40 | 12.43 ms |

ActServe recommended 16 by its documented ordering: most on-time actions,
then lower p95 latency, then fewer backend calls.

Raw result:
[`results/a100_v03_tune_4_8_16_20260815.json`](results/a100_v03_tune_4_8_16_20260815.json)

SHA-256:
`75f61466e8ac415c0be5f652eec56fc7a501cb707997f9d732876b805ffa9624`
