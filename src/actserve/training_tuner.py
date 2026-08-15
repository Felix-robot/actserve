from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Scalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class TrainingTrial:
    config: Mapping[str, Scalar]
    samples_per_second: float
    p95_step_ms: float
    memory_headroom_mb: float
    status: Literal["ok", "oom", "failed"] = "ok"

    def __post_init__(self) -> None:
        if self.status not in {"ok", "oom", "failed"}:
            raise ValueError("status must be ok, oom, or failed")
        metrics = (self.samples_per_second, self.p95_step_ms, self.memory_headroom_mb)
        if any(not math.isfinite(value) for value in metrics):
            raise ValueError("trial metrics must be finite")
        if self.samples_per_second < 0 or self.p95_step_ms < 0:
            raise ValueError("throughput and latency must be non-negative")
        for key, value in self.config.items():
            if not key or not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("config must contain non-empty keys and scalar values")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TuningDecision:
    selected: TrainingTrial | None
    eligible_trials: int
    rejected_status: int
    rejected_memory: int
    rejected_latency: int
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "actserve.training_tuning.v1",
            "scope": "offline trial selection only; no training configuration is changed",
            "selected": self.selected.as_dict() if self.selected is not None else None,
            "eligible_trials": self.eligible_trials,
            "rejected": {
                "status": self.rejected_status,
                "memory": self.rejected_memory,
                "latency": self.rejected_latency,
            },
            "rationale": self.rationale,
        }


def generate_sweep(
    baseline: Mapping[str, Scalar],
    axes: Mapping[str, Sequence[Scalar]],
    *,
    max_trials: int = 32,
) -> list[dict[str, Scalar]]:
    """Generate a deterministic, bounded Cartesian sweep of public tunables."""

    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    names = sorted(axes)
    if any(not name or not axes[name] for name in names):
        raise ValueError("sweep axes must have non-empty names and values")
    candidates = []
    combinations = itertools.product(*(axes[name] for name in names))
    for values in combinations:
        candidate = dict(baseline)
        candidate.update(zip(names, values, strict=True))
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= max_trials:
            break
    return candidates


def select_training_trial(
    trials: Sequence[TrainingTrial],
    *,
    min_memory_headroom_mb: float = 1024,
    max_p95_step_ms: float | None = None,
) -> TuningDecision:
    if min_memory_headroom_mb < 0:
        raise ValueError("min_memory_headroom_mb must be non-negative")
    if max_p95_step_ms is not None and max_p95_step_ms <= 0:
        raise ValueError("max_p95_step_ms must be positive")

    eligible = []
    rejected_status = 0
    rejected_memory = 0
    rejected_latency = 0
    for trial in trials:
        if trial.status != "ok":
            rejected_status += 1
        elif trial.memory_headroom_mb < min_memory_headroom_mb:
            rejected_memory += 1
        elif max_p95_step_ms is not None and trial.p95_step_ms > max_p95_step_ms:
            rejected_latency += 1
        else:
            eligible.append(trial)

    selected = max(
        eligible,
        key=lambda trial: (
            trial.samples_per_second,
            -trial.p95_step_ms,
            trial.memory_headroom_mb,
            json.dumps(dict(trial.config), sort_keys=True),
        ),
        default=None,
    )
    rationale = (
        "highest measured samples_per_second among eligible trials; ties prefer lower "
        "p95_step_ms, then greater memory_headroom_mb"
        if selected is not None
        else "no trial satisfied status, memory, and latency constraints"
    )
    return TuningDecision(
        selected=selected,
        eligible_trials=len(eligible),
        rejected_status=rejected_status,
        rejected_memory=rejected_memory,
        rejected_latency=rejected_latency,
        rationale=rationale,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actserve tune-training",
        description="Select a training configuration from measured isolated trials.",
    )
    parser.add_argument("input", type=Path, help="JSON file containing a trials array")
    parser.add_argument("--min-memory-headroom-mb", type=float, default=1024)
    parser.add_argument("--max-p95-step-ms", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.input.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("trials"), list):
        raise ValueError("input must be a JSON object with a trials array")
    trials = [TrainingTrial(**item) for item in payload["trials"]]
    decision = select_training_trial(
        trials,
        min_memory_headroom_mb=args.min_memory_headroom_mb,
        max_p95_step_ms=args.max_p95_step_ms,
    )
    print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
    return 0 if decision.selected is not None else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
