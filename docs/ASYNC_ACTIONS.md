# Asynchronous action queues

`AsyncActionQueue` decouples action-chunk inference from action execution. A
robot-specific client consumes individual actions while a producer claims the
single refill slot whenever the queue reaches its configured low watermark.

```python
from actserve import ActionQueueConfig, AsyncActionQueue

queue = AsyncActionQueue(
    session_id="robot-1",
    model="public-policy",
    config=ActionQueueConfig(low_watermark=5),
)

async def refill_loop():
    while await queue.acquire_refill():
        outcome = await request_latest_observation()
        await queue.finish_refill(outcome.action)

async def control_tick():
    queued = await queue.get(timeout_ms=20)
    send_robot_specific_action(queued.action)
```

The default `replace` handoff discards unexecuted actions from an older chunk
when a newer observation produces a chunk. `append` is available for policies
whose chunks must finish in order. Neither mode blends overlapping chunks;
model-aware RTC adapters should perform that operation before publishing the
new chunk.

An empty queue raises `ActionQueueEmpty`. ActServe deliberately does not invent
a hold, zero, or stop command because safe fallback semantics depend on the
robot and controller. Physical execution must define that policy outside this
generic package.
