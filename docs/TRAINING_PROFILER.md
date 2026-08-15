# Training profiler

ActServe's dependency-free training profiler measures time spent waiting for
data, running forward/backward computation, updating parameters, and writing
checkpoints. It retains numeric timings only—never batches, paths, model state,
prompts, images, actions, or checkpoints.

```python
from actserve.training_profiler import TrainingProfiler

profiler = TrainingProfiler()

for _ in range(train_steps):
    with profiler.step(samples=batch_size) as step:
        with step.phase("data_wait"):
            batch = next(loader)
        with step.phase("forward"):
            loss = model(batch)
        with step.phase("backward"):
            loss.backward()
        with step.phase("optimizer"):
            optimizer.step()

print(profiler.snapshot().as_dict())
```

Durations measured by an existing framework profiler can instead be supplied
with `record_step()`. Recommendations are hypotheses to benchmark, not automatic
changes or claims of speedup. Run it first in a public or isolated workload; do
not attach it to a shared formal experiment without the experiment owner's
approval.

## Select from measured trials

After running isolated candidates, record only public tunables and numeric
results using the schema in `examples/training_trials.json`, then run:

```bash
actserve tune-training examples/training_trials.json --min-memory-headroom-mb 1024
```

OOM and failed trials are excluded, as are candidates below the requested
memory margin or above an optional p95 limit. The remaining candidate with the
highest measured throughput is selected. ActServe prints the decision but does
not edit a config, launch a job, or stop a process.
