from __future__ import annotations

import argparse
import asyncio

from actserve.async_benchmark import benchmark


def test_async_refill_reduces_control_loop_idle_time() -> None:
    result = asyncio.run(
        benchmark(
            argparse.Namespace(
                chunks=3,
                chunk_size=10,
                inference_ms=6,
                tick_ms=2,
                low_watermark=5,
            )
        )
    )

    synchronous = result["synchronous"]
    asynchronous = result["asynchronous"]
    assert isinstance(synchronous, dict)
    assert isinstance(asynchronous, dict)
    assert synchronous["actions_executed"] == 30
    assert asynchronous["actions_executed"] == 30
    assert asynchronous["steady_idle_ticks"] < synchronous["steady_idle_ticks"]
    assert asynchronous["total_idle_ticks"] < synchronous["total_idle_ticks"]
