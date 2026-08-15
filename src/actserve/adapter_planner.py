from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AdapterDemand:
    adapter: str
    size_mb: float
    requests_per_minute: float
    cold_load_ms: float

    def __post_init__(self) -> None:
        if not self.adapter:
            raise ValueError("adapter must be non-empty")
        metrics = (self.size_mb, self.requests_per_minute, self.cold_load_ms)
        if any(not math.isfinite(value) or value < 0 for value in metrics):
            raise ValueError("adapter demand metrics must be finite and non-negative")

    @property
    def avoided_load_ms_per_minute(self) -> float:
        return self.requests_per_minute * self.cold_load_ms


@dataclass(frozen=True, slots=True)
class AdapterResidencyPlan:
    budget_mb: float
    selected: tuple[AdapterDemand, ...]
    used_mb: float
    avoided_load_ms_per_minute: float
    excluded: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "actserve.adapter_residency.v1",
            "scope": "offline residency plan only; no GPU state is changed",
            "budget_mb": self.budget_mb,
            "used_mb": self.used_mb,
            "avoided_load_ms_per_minute": self.avoided_load_ms_per_minute,
            "selected": [asdict(item) for item in self.selected],
            "excluded": list(self.excluded),
        }


def plan_adapter_residency(
    demands: Sequence[AdapterDemand], *, budget_mb: float
) -> AdapterResidencyPlan:
    """Solve adapter residency as a deterministic integer-MiB knapsack."""

    if not math.isfinite(budget_mb) or budget_mb < 0:
        raise ValueError("budget_mb must be finite and non-negative")
    if len({demand.adapter for demand in demands}) != len(demands):
        raise ValueError("adapter demand entries must be unique")

    capacity = math.floor(budget_mb)
    ordered = sorted(demands, key=lambda demand: demand.adapter)
    # capacity -> (benefit, selected indexes); integer MiB rounds each adapter up
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, demand in enumerate(ordered):
        weight = math.ceil(demand.size_mb)
        additions = dict(states)
        for used, (benefit, indexes) in states.items():
            candidate_used = used + weight
            if candidate_used > capacity:
                continue
            candidate = (benefit + demand.avoided_load_ms_per_minute, indexes + (index,))
            incumbent = additions.get(candidate_used)
            if incumbent is None or _is_better(candidate, incumbent):
                additions[candidate_used] = candidate
        states = additions

    _best_used, (best_benefit, best_indexes) = max(
        states.items(),
        key=lambda item: (item[1][0], -item[0], tuple(-index for index in item[1][1])),
    )
    selected = tuple(ordered[index] for index in best_indexes)
    selected_names = {demand.adapter for demand in selected}
    return AdapterResidencyPlan(
        budget_mb=budget_mb,
        selected=selected,
        used_mb=sum(demand.size_mb for demand in selected),
        avoided_load_ms_per_minute=best_benefit,
        excluded=tuple(
            demand.adapter for demand in ordered if demand.adapter not in selected_names
        ),
    )


def _is_better(
    candidate: tuple[float, tuple[int, ...]], incumbent: tuple[float, tuple[int, ...]]
) -> bool:
    return candidate[0] > incumbent[0] or (
        candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actserve plan-adapters",
        description="Plan adapter residency within a GPU memory budget.",
    )
    parser.add_argument("input", type=Path, help="JSON file containing an adapters array")
    parser.add_argument("--budget-mb", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.input.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("adapters"), list):
        raise ValueError("input must be a JSON object with an adapters array")
    demands = [AdapterDemand(**item) for item in payload["adapters"]]
    plan = plan_adapter_residency(demands, budget_mb=args.budget_mb)
    print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
