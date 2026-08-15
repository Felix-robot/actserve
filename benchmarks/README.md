# Public benchmark results

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
