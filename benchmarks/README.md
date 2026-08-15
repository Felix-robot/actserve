# Public benchmark results

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
