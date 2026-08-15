# Shared-backbone adapter routing

`AdapterBackend` maps public logical model IDs to a backbone, adapter, and input
signature. This lets one loaded VLA backbone serve multiple task adapters while
the scheduler retains per-session deadlines and action identity checks.

Cross-adapter batching is disabled by default. Enable `mixed_adapter_batch=True`
only when the underlying runtime can select a different adapter for each batch
item. Requests are never batched across different backbones or input signatures.

```python
backend = AdapterBackend(
    infer_routed_batch,
    [
        AdapterRoute("pick", "shared-vla", "pick-lora"),
        AdapterRoute("place", "shared-vla", "place-lora"),
    ],
    mixed_adapter_batch=True,
)
```

The callback receives `RoutedRequest` objects and remains responsible for
loading the declared backbone/adapters and returning actions whose request,
session, sequence, and logical model identities match. ActServe stores routing
metadata only; it does not publish or inspect weights, observations, or actions.
