import asyncio

import pytest

from actserve.action_queue import (
    ActionQueueClosed,
    ActionQueueConfig,
    ActionQueueEmpty,
    AsyncActionQueue,
)
from actserve.profiler import StageProfiler
from actserve.types import ActionChunk


def chunk(sequence: int, actions: list[int], *, session: str = "robot") -> ActionChunk:
    return ActionChunk(
        request_id=f"request-{sequence}",
        session_id=session,
        sequence_no=sequence,
        actions=actions,
        model="policy",
    )


async def test_new_chunk_replaces_unexecuted_actions_by_default() -> None:
    queue = AsyncActionQueue(session_id="robot", model="policy")
    assert await queue.put(chunk(1, [10, 11, 12]))
    first = await queue.get()
    assert first.action == 10
    assert await queue.put(chunk(2, [20, 21]))
    assert (await queue.get()).action == 20
    snapshot = await queue.snapshot()
    assert snapshot.replaced_actions == 2
    assert snapshot.remaining_actions == 1


async def test_append_policy_preserves_previous_tail() -> None:
    queue = AsyncActionQueue(
        session_id="robot",
        model="policy",
        config=ActionQueueConfig(handoff_policy="append"),
    )
    await queue.put(chunk(1, [10, 11]))
    await queue.put(chunk(2, [20]))
    assert [(await queue.get()).action for _ in range(3)] == [10, 11, 20]


async def test_stale_and_misrouted_chunks_are_rejected() -> None:
    queue = AsyncActionQueue(session_id="robot", model="policy")
    assert await queue.put(chunk(2, [20]))
    assert not await queue.put(chunk(1, [10]))
    assert not await queue.put(chunk(3, [30], session="other"))
    assert (await queue.snapshot()).rejected_chunks == 2


async def test_low_watermark_claims_only_one_refill_slot() -> None:
    queue = AsyncActionQueue(
        session_id="robot",
        model="policy",
        config=ActionQueueConfig(low_watermark=1),
    )
    assert await queue.acquire_refill()
    waiting = asyncio.create_task(queue.acquire_refill())
    await asyncio.sleep(0)
    assert not waiting.done()
    assert await queue.finish_refill(chunk(1, [1, 2, 3]))
    await queue.get()
    await queue.get()
    assert await asyncio.wait_for(waiting, 0.1)
    await queue.finish_refill(chunk(2, [4, 5]))


async def test_empty_queue_times_out_without_inventing_fallback() -> None:
    profiler = StageProfiler()
    queue = AsyncActionQueue(session_id="robot", model="policy", profiler=profiler)
    with pytest.raises(ActionQueueEmpty):
        await queue.get(timeout_ms=1)
    assert (await queue.snapshot()).underruns == 1
    assert profiler.snapshot().get("action_queue.underrun", "events") is not None


async def test_close_wakes_waiters() -> None:
    queue = AsyncActionQueue(session_id="robot", model="policy")
    waiting = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    await queue.close()
    with pytest.raises(ActionQueueClosed):
        await waiting
