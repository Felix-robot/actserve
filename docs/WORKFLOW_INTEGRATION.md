# Formal workflow integration gate

ActServe v0.7 can be evaluated beside an existing embodied-AI workflow, but it
must not be inserted into a running formal experiment merely because public
synthetic benchmarks pass. Use the following gate for private integrations.

## Boundaries

- Keep private adapters, checkpoints, observations, prompts, actions, task
  names, seeds, and result paths outside this repository.
- Do not attach to, signal, restart, pause, or reconfigure an existing process.
- Obtain fresh, explicit authorization from the workflow owner before each move
  from offline analysis to shadow execution or from shadow execution to active
  serving.
- Treat throughput and latency gains as engineering evidence, never as policy
  quality or task-success evidence.

## Staged entry

1. **Offline report:** feed copied numeric timings into the training profiler,
   tuner, or adapter planner. No process attachment and no model data.
2. **Isolated replay:** run the same public or approved replay inputs through
   baseline and ActServe with immutable configs. Validate action identity and
   parity before comparing speed.
3. **Shadow canary:** on a newly launched, owner-approved job, observe or mirror
   requests without controlling a robot or becoming the scientific authority.
4. **Active canary:** route a bounded, reversible fraction only after the owner
   accepts the preregistered gates and rollback procedure.
5. **Workflow default:** promote only after repeated evidence across the
   workload's required seeds, hardware, and failure cases.

## Preregistered engineering gates

Choose numeric thresholds before collecting the integration result. At minimum:

- identical model/checkpoint, precision, inputs, action interface, hardware
  allocation, and workload order for baseline and candidate;
- zero request/session/sequence/model identity mismatches;
- zero new OOMs, deadlocks, silent fallbacks, or dropped required outputs;
- a minimum GPU-memory headroom and an automatic fail-closed response when the
  action queue underruns;
- a material improvement in the declared primary engineering metric, such as
  samples/second, useful actions/second, p95 latency, stale compute, or peak
  memory;
- no regression beyond a declared tolerance in task success or training
  quality, measured by the workflow's existing scientific authority.

## Rollback package

Before an active canary, preserve the baseline launch command/config, record the
ActServe version and adapter version, define the stop condition, and verify that
returning to the baseline does not require deleting or rewriting experiment
artifacts. ActServe itself must not terminate unrelated or pre-existing jobs.

Passing this document's checklist means an integration is ready to evaluate. It
does not by itself authorize the evaluation or prove a gain on a private task.
