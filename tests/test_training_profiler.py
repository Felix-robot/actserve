from __future__ import annotations

import json

import pytest

from actserve.training_profiler import TrainingProfiler


def test_training_profile_finds_data_bottleneck_without_retaining_batches() -> None:
    profiler = TrainingProfiler()
    for _ in range(3):
        profiler.record_step(
            {
                "data_wait": 40,
                "forward": 20,
                "backward": 25,
                "optimizer": 5,
                "checkpoint": 10,
            },
            samples=8,
        )

    snapshot = profiler.snapshot()
    payload = snapshot.as_dict()
    assert snapshot.steps == 3
    assert snapshot.samples == 24
    assert snapshot.bottleneck == "data_wait"
    assert snapshot.samples_per_second == 80
    assert any("prefetching" in item for item in snapshot.recommendations)
    assert "batch" not in json.dumps(payload)


def test_training_step_context_records_named_phases() -> None:
    profiler = TrainingProfiler()
    with profiler.step(samples=2) as step:
        step.record("forward", 2)
        with step.phase("optimizer"):
            pass

    snapshot = profiler.snapshot()
    assert snapshot.steps == 1
    assert {phase.name for phase in snapshot.phases} == {"forward", "optimizer"}


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_training_profiler_rejects_invalid_durations(duration: float) -> None:
    profiler = TrainingProfiler()
    with pytest.raises(ValueError):
        profiler.record_step({"forward": duration})
