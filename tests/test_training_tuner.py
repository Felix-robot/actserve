from __future__ import annotations

import json

from actserve.training_tuner import TrainingTrial, generate_sweep, main, select_training_trial


def test_generate_sweep_is_deterministic_and_bounded() -> None:
    sweep = generate_sweep(
        {"precision": "bf16"},
        {"workers": [4, 8], "microbatch": [2, 4]},
        max_trials=3,
    )
    assert sweep == [
        {"precision": "bf16", "microbatch": 2, "workers": 4},
        {"precision": "bf16", "microbatch": 2, "workers": 8},
        {"precision": "bf16", "microbatch": 4, "workers": 4},
    ]


def test_tuner_rejects_oom_and_low_headroom_before_maximizing_throughput() -> None:
    trials = [
        TrainingTrial({"microbatch": 8}, 120, 70, 300, "ok"),
        TrainingTrial({"microbatch": 6}, 110, 75, 1600, "ok"),
        TrainingTrial({"microbatch": 10}, 150, 0, 0, "oom"),
    ]
    decision = select_training_trial(trials, min_memory_headroom_mb=1024)
    assert decision.selected is trials[1]
    assert decision.rejected_memory == 1
    assert decision.rejected_status == 1


def test_tune_training_cli_emits_machine_readable_decision(tmp_path, capsys) -> None:
    input_path = tmp_path / "trials.json"
    input_path.write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "config": {"workers": 8},
                        "samples_per_second": 42,
                        "p95_step_ms": 20,
                        "memory_headroom_mb": 2048,
                        "status": "ok",
                    }
                ]
            }
        )
    )
    assert main([str(input_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["selected"]["config"] == {"workers": 8}
    assert output["scope"].startswith("offline trial selection")
