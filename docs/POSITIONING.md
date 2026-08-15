# Positioning

ActServe is a control plane for continuous embodied inference. It is not a
new tensor runtime and does not reimplement model kernels.

Portable runtimes can remain the execution plane beneath an ActServe
backend. The v0.1 differentiation is intentionally narrow and measurable:

- multiple persistent robot sessions;
- per-observation deadlines and EDF scheduling;
- explicit stale-frame coalescing;
- latency-aware microbatch admission;
- fail-closed action identity routing;
- public scheduling and replay metrics.

The project should claim superiority only on a published benchmark axis where
the same model, hardware, input stream, action interface, and behavior outcome
are compared. Until then, it is accurate to describe these as implemented
serving semantics, not as proven GPU or task-level superiority.
