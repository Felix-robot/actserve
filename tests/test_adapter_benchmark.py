from __future__ import annotations

import argparse
import asyncio

from actserve.adapter_benchmark import benchmark


def test_shared_backbone_benchmark_reduces_calls_and_modeled_memory() -> None:
    result = asyncio.run(
        benchmark(
            argparse.Namespace(
                adapters=3,
                sessions_per_adapter=2,
                max_batch_size=6,
                fixed_ms=1,
                per_item_ms=0.1,
                backbone_mb=1000,
                adapter_mb=100,
            )
        )
    )
    isolated = result["isolated_models"]
    shared = result["shared_backbone"]
    assert isinstance(isolated, dict)
    assert isinstance(shared, dict)
    assert isolated["completed"] == shared["completed"] == 6
    assert isolated["backend_calls"] == 3
    assert shared["backend_calls"] == 1
    assert shared["modeled_memory_mb"] < isolated["modeled_memory_mb"]
