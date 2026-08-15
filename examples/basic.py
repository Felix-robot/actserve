import asyncio

from actserve.backend import SimulatedBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.types import InferenceRequest


async def main() -> None:
    backend = SimulatedBackend(fixed_ms=10, per_item_ms=1, max_batch_size=4)
    async with Scheduler(backend, SchedulerConfig()) as scheduler:
        outcomes = await asyncio.gather(
            *[
                scheduler.submit(
                    InferenceRequest.with_timeout(
                        session_id=f"robot-{index}",
                        model="demo-vla",
                        observation={"frame": 0},
                        timeout_ms=100,
                        sequence_no=0,
                    )
                )
                for index in range(4)
            ]
        )
    for outcome in outcomes:
        print(outcome.request.session_id, outcome.status.value, outcome.end_to_end_ms)


if __name__ == "__main__":
    asyncio.run(main())
