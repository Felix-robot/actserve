from __future__ import annotations

import json

import pytest

from actserve.adapter_planner import AdapterDemand, main, plan_adapter_residency


def test_residency_planner_finds_global_capacity_optimum() -> None:
    demands = [
        AdapterDemand("large", 6, 12, 1),
        AdapterDemand("small-a", 5, 10, 1),
        AdapterDemand("small-b", 5, 10, 1),
    ]
    plan = plan_adapter_residency(demands, budget_mb=10)
    assert [item.adapter for item in plan.selected] == ["small-a", "small-b"]
    assert plan.used_mb == 10
    assert plan.avoided_load_ms_per_minute == 20


def test_residency_planner_rejects_duplicate_adapter_ids() -> None:
    demands = [AdapterDemand("same", 1, 1, 1), AdapterDemand("same", 2, 2, 2)]
    with pytest.raises(ValueError, match="unique"):
        plan_adapter_residency(demands, budget_mb=10)


def test_plan_adapters_cli_is_machine_readable(tmp_path, capsys) -> None:
    input_path = tmp_path / "adapters.json"
    input_path.write_text(
        json.dumps(
            {
                "adapters": [
                    {
                        "adapter": "hot-task",
                        "size_mb": 256,
                        "requests_per_minute": 30,
                        "cold_load_ms": 80,
                    }
                ]
            }
        )
    )
    assert main([str(input_path), "--budget-mb", "512"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected"][0]["adapter"] == "hot-task"
    assert payload["scope"].startswith("offline residency plan")
