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
